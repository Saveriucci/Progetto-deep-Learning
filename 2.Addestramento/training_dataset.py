import re
import ast
import json
import pandas as pd

# Percorsi
CSV_PATH = r"C:\Users\tomas\Desktop\universita\Magistrale\Secondo Anno\Primo Semestre\Deep Learning\Progetto\Dataset\RecipeNLG_dataset.csv"
TXT_PATH = r"C:\Users\tomas\Desktop\universita\Magistrale\Secondo Anno\Primo Semestre\Deep Learning\Progetto\Dataset\Dataset_testuale.txt"
OUT_CSV  = r"C:\Users\tomas\Desktop\universita\Magistrale\Secondo Anno\Primo Semestre\Deep Learning\Progetto\Dataset\Training_Dataset_Clean.csv"

# --- 1) Caricamento Dataset Strutturato ---
print("Caricamento RecipeNLG...")
ds = pd.read_csv(CSV_PATH, engine="python", on_bad_lines="skip", dtype=str, keep_default_na=False)
if "Unnamed: 0" in ds.columns: ds.rename(columns={"Unnamed: 0": "id"}, inplace=True)

# --- 2) Parsing Dataset Testuale (Logica anti-fusione) ---
print("Parsing Dataset_testuale.txt...")
with open(TXT_PATH, "r", encoding="utf-8", errors="replace") as f:
    raw_txt = f.read()

# Questa regex garantisce che il testo venga preso integralmente dall'ID X all'ID Y
recipe_pattern = re.compile(r"i?ID\s+(\d+)\s*[—-]*\s*(.*?)(?=i?ID\s+\d+|$)", re.DOTALL | re.IGNORECASE)
id_to_text = {int(m[0]): m[1].strip() for m in recipe_pattern.findall(raw_txt)}

# --- 3) Selezione Intervallo Fisso (50 -> 900) ---
# Tagliamo i primi 50 e gli ultimi 100 dalle prime 1000 righe
df_subset = ds.iloc[50:900].copy()

# --- 4) Definizione Blacklist (Ricette corrotte nell'intervallo 50-900) ---
ID_DA_ESCLUDERE = [
    25, 261, 380, 383, 585, 619, 674, 844, 851, 899, 902, 993, 994, 999
]

# --- 5) Creazione Dataset Finale ---
final_data = []

print("Pulizia e validazione finale...")
for _, row in df_subset.iterrows():
    rid = int(row["id"])
    
    # Se l'ID è nella blacklist o non ha testo nel TXT, lo saltiamo
    if rid in ID_DA_ESCLUDERE or rid not in id_to_text:
        continue
    
    # Prepariamo il JSON
    try:
        ing = ast.literal_eval(row["ingredients"])
        steps = ast.literal_eval(row["directions"])
    except:
        continue # Salta se il formato lista è rotto nel sorgente
        
    json_str = json.dumps({
        "title": row["title"].strip(),
        "ingredients": ing,
        "directions": steps
    }, ensure_ascii=False)
    
    final_data.append({
        "id": rid,
        "text": id_to_text[rid],
        "json": json_str
    })

# Salvataggio
df_final = pd.DataFrame(final_data)
df_final.to_csv(OUT_CSV, index=False, encoding="utf-8")

# --- Report ---
print("-" * 30)
print(f"PROCESSO COMPLETATO")
print(f"Ricette originali nel range 50-900: 850")
print(f"Ricette rimosse perché corrotte/mancanti: {850 - len(df_final)}")
print(f"Numero finale record salvati: {len(df_final)}")