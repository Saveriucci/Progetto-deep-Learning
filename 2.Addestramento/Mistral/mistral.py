import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# -----------------------
# CONFIGURAZIONE
# -----------------------
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_PATH = r"C:\Users\tomas\Desktop\universita\Magistrale\Secondo Anno\Primo Semestre\Deep Learning\Progetto\2.Addestramento\Training_Dataset_Clean.csv"
OUTPUT_DIR = r"C:\Users\tomas\Desktop\universita\Magistrale\Secondo Anno\Primo Semestre\Deep Learning\Progetto\2.Addestramento\Mistral\mistral-recipe-json-model"

MAX_LENGTH = 192
TEST_SIZE = 0.1
SEED = 42

SYSTEM = (
    "You are an information extraction engine.\n"
    "Input: a raw recipe written in natural language.\n"
    "Output: ONLY valid JSON with EXACTLY these keys: title, ingredients, steps.\n"
    "No extra text.\n\n"
    "Extraction rules:\n"
    "- title: use an explicit title if present; otherwise infer a short, non-empty title from the recipe; if impossible use \"\".\n"
    "- ingredients: list ONLY ingredients mentioned in the text. Keep quantities if present (e.g., \"200g spaghetti\").\n"
    "- steps: ordered list of cooking actions derived from the text, split into multiple short imperative steps.\n"
    "- Use double quotes for all strings. No trailing commas. Valid JSON.\n"
    "- Do not add any keys besides title, ingredients, steps.\n"
    "Start your answer with '{' and end with '}'.\n\n"
    "Do NOT wrap the JSON in markdown code fences (no ```json and no ```).\n"
    "Output must be plain JSON text only.\n\n"
    "End the output immediately after the closing '}'.\n\n"
    "JSON format:\n"
    "{\n"
    "  \"title\": \"...\",\n"
    "  \"ingredients\": [\"...\"],\n"
    "  \"steps\": [\"...\"]\n"
    "}\n"
)

# -----------------------
# 1) Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

# -----------------------
# 2) QLoRA: 4-bit load
# -----------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",           # IMPORTANTISSIMO con modelli 4-bit
    trust_remote_code=True,
)

# prepara modello per k-bit training (LayerNorm, input grads, ecc.)
model = prepare_model_for_kbit_training(model)

# riduce memoria attivazioni
model.gradient_checkpointing_enable()
model.config.use_cache = False

# -----------------------
# 3) LoRA (r invariato = 16)
# -----------------------
peft_config = LoraConfig(
    r=16,  # <-- come richiesto
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# -----------------------
# 4) Dataset
# -----------------------
dataset = load_dataset("csv", data_files=DATASET_PATH)["train"]

def sanitize(examples):
    return {
        "text": ["" if x is None else str(x) for x in examples["text"]],
        "json": ["" if x is None else str(x) for x in examples["json"]],
    }

dataset = dataset.map(sanitize, batched=True)
dataset = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)

# -----------------------
# 5) Tokenizzazione + masking
# -----------------------
def tokenize_and_mask(examples):
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    for user_text, target_json in zip(examples["text"], examples["json"]):
        prefix_messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
        ]
        prefix_str = tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        full_messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": target_json},
        ]
        full_str = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        prefix_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]

        enc = tokenizer(
            full_str,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            add_special_tokens=False
        )

        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        labels = input_ids.copy()
        cut = min(len(prefix_ids), MAX_LENGTH)
        labels[:cut] = [-100] * cut

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attn)
        labels_batch.append(labels)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch,
    }

tokenized = dataset.map(
    tokenize_and_mask,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# -----------------------
# 6) TrainingArguments (FIX: place_model_on_device=False)
# -----------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.03,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    report_to="none",

    fp16=(torch.cuda.is_available() and not use_bf16),
    bf16=(torch.cuda.is_available() and use_bf16),

    optim="paged_adamw_8bit",
    group_by_length=True,
)

# -----------------------
# 7) Trainer
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=default_data_collator,
)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Inizio addestramento...")
trainer.train()

# -----------------------
# 8) Salvataggio
# -----------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modello salvato in: {OUTPUT_DIR}")
