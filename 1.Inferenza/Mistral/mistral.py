import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList

class StopOnCompleteJson(StoppingCriteria):
    def __init__(self, tokenizer, start_len: int):
        self.tokenizer = tokenizer
        self.start_len = start_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # decodifica solo la parte generata
        gen_ids = input_ids[0][self.start_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        start = text.find("{")
        if start == -1:
            return False

        depth = 0
        in_str = False
        esc = False

        # percorre dal primo '{' e chiude al matching '}'
        for ch in text[start:]:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return True  # JSON completo

        return False

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA available:", torch.cuda.is_available())

def strip_code_fences(s: str) -> str:
    # rimuove eventuali ```json ... ``` che il modello a volte aggiunge
    s = re.sub(r"```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("```", "")
    return s.strip()

def extract_first_json_block(s: str):
    """
    Estrae il primo blocco JSON completo { ... }.
    Gestisce correttamente { } dentro stringhe JSON (tra " ").
    """
    s = strip_code_fences(s)
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

    return None

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

# Fix rope_scaling edge-cases that can break loading in some environments
rs = getattr(config, "rope_scaling", None)
if isinstance(rs, dict):
    if "type" not in rs and "rope_type" in rs:
        rs["type"] = rs["rope_type"]
    if rs.get("type") in (None, "default", "linear", "dynamic"):
        config.rope_scaling = None

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    attn_implementation="sdpa",
)
model.eval()
if device == "cpu":
    model = model.to(device)

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

RETRY_SYSTEM = (
    SYSTEM
    + "\nIMPORTANT:\n"
      "- Your previous output was invalid or cut off.\n"
      "- Return ONLY ONE complete JSON object.\n"
      "- Ensure it is valid JSON and fully closed.\n"
      "- Do not add any extra commentary.\n"
      "- End immediately after the final '}'.\n"
)

RECIPES = [
    "No-Bake Nut Cookies. In a heavy two-quart saucepan, combine 1 cup firmly packed brown sugar, 1/2 cup evaporated milk, 2 tablespoons butter or margarine, and 1/2 cup broken pecans. Cook over medium heat, stirring constantly, until the mixture bubbles across the surface. Continue boiling and stirring for 5 minutes. Remove from heat and stir in 1/2 teaspoon vanilla, then add 3 1/2 cups bite-size shredded rice biscuits and mix thoroughly. Using two teaspoons, drop the mixture onto wax paper to form about 30 clusters. Let stand for approximately 30 minutes, until firm.",
    "Jewell Ball’s Chicken. Spread 1 small jar of chipped beef, cut into pieces, over the bottom of a baking dish. Place 4 boneless chicken breasts on top. In a bowl, mix together 1 can cream of mushroom soup and 1 carton sour cream, then pour the mixture evenly over the chicken. Bake uncovered at 275°F for 3 hours.",
    "Creamy Corn. Place 2 packages (16 oz each) frozen corn into a slow cooker along with 1 package (8 oz) cream cheese cut into cubes, 1/3 cup butter cut into pieces, 1/2 teaspoon garlic powder, 1/2 teaspoon salt, and 1/4 teaspoon pepper. Cover and cook on low heat for about 4 hours, until heated through and creamy, stirring before serving.",
    "Chicken Funny. Boil 1 large whole chicken until fully cooked, then debone and cut into bite-size pieces. Place the chicken in a medium square casserole dish. Pour over 2 cans (10 1/2 oz each) chicken gravy and 1 can cream of mushroom soup, spreading evenly. Prepare 1 box (6 oz) Stove Top stuffing according to package directions, keeping it slightly dry, and spread it over the chicken mixture. Sprinkle 4 oz shredded cheese on top and bake at 350°F for about 20 minutes, until bubbly and lightly golden.",
    "Reese’s Cups Candy. Mix together 1 cup peanut butter, 3/4 cup graham cracker crumbs, 1 cup melted butter, and 1 pound powdered sugar until smooth. Press the mixture firmly into an ungreased 13 x 9-inch pan. Melt 1 large package of chocolate chips and spread evenly over the top. Refrigerate for about 20 minutes, then cut into pieces before the chocolate fully hardens. Store in the refrigerator.",
    "Cheeseburger Potato Soup. Wash 6 baking potatoes and prick them several times with a fork. Microwave with a damp paper towel on high for 6 to 8 minutes, until tender, then let cool slightly. Cut in half lengthwise, scoop out the pulp, and discard the skins. Brown 1 pound extra-lean ground beef in a skillet, drain, and set aside. In a large kettle, melt 2/3 cup butter or margarine over low heat, add flour and stir until smooth, then cook for 1 minute. Gradually add 6 cups milk, stirring constantly until thickened and bubbly. Stir in the potatoes, ground beef, 3/4 teaspoon salt, 1/2 teaspoon pepper, 1 cup shredded Cheddar cheese, 2 tablespoons chopped green onion, and 1/2 cup cooked crumbled bacon. Heat gently without boiling. Stir in 1 carton (8 oz) sour cream if desired. Garnish with remaining cheese, bacon, and green onions.",
    "Rhubarb Coffee Cake. Cream together 1 1/2 cups sugar and 1/2 cup butter. Add 1 egg and beat well. Alternately add 1 cup buttermilk and a mixture of 2 cups flour, 1/2 teaspoon salt, and 1 teaspoon soda. Mix thoroughly, then stir in 2 cups finely chopped rhubarb and 1 teaspoon vanilla. Pour into a greased 9 x 13-inch pan and add topping as desired before baking.",
    "Scalloped Corn. Mix together 1 can cream-style corn, 1 can whole kernel corn, about 20 crushed saltine crackers, 1 beaten egg, 2 teaspoons melted butter, and pepper to taste. Transfer to a buttered baking dish, dot the top with the remaining 4 teaspoons butter, and bake at 350°F for 1 hour.",
    "Nolan’s Pepper Steak. Roll 1 1/2 pounds round steak cut into strips in flour. Brown in a skillet, then season with salt and pepper. Combine the reserved tomato liquid from 1 can of tomatoes with 1 3/4 cups water and 1/2 cup onions. Add the browned steak, cover, and simmer for 1 hour and 15 minutes. Uncover, stir in 1 1/2 tablespoons Worcestershire sauce, then add the tomatoes and 2 diced green peppers. Simmer for 5 minutes and serve over hot cooked rice.",
    "Millionaire Pie. Empty 1 large container Cool Whip into a bowl. Drain 1 large can crushed pineapple and mix it into the whipped topping. Add 1 can condensed milk, then squeeze the juice from 3 lemons, removing seeds, and stir it in. Chop 1 cup pecans and fold into the mixture. Pour into 2 graham cracker crusts, cover, and chill overnight before serving.",
    "Double Cherry Delight. Drain 1 can (17 oz) dark sweet pitted cherries, reserving the syrup, and cut the cherries in half. Add 1/2 cup ginger ale to the syrup and enough water to make 1 1/2 cups liquid total. Dissolve 1 package (6 oz) cherry-flavored gelatin in 2 cups boiling water. Stir in the measured liquid and add 1/8 teaspoon almond extract. Chill until very thick. Fold in 1 cup miniature marshmallows and the cherries, then spoon into a 6-cup mold. Chill until firm, at least 4 hours or overnight.",
    "Buckeye Candy. Mix together 1 box powdered sugar, 8 oz soft butter, and 1 jar (8 oz) peanut butter until smooth. Roll the mixture into small balls and place on a cookie sheet. Freeze for at least 30 minutes. Melt 12 oz chocolate chips together with paraffin in a double boiler. Using a toothpick, dip each ball about three-quarters of the way into the chocolate mixture, leaving the top exposed.",
    "Quick Barbecue Wings. Clean chicken wings in the quantity desired. Coat lightly with flour and fry until fully cooked. Transfer the wings to a microwave-safe bowl, add barbecue sauce to taste, and stir well. Microwave on high for 4 minutes, stirring once during cooking.",
    "Taco Salad Chip Dip. Mix together 8 oz taco sauce, 8 oz sour cream, and 8 oz cream cheese until smooth. Spread the mixture onto a pizza pan. Brown 1 lb ground beef, drain excess fat, and mix with 1 package taco seasoning. Spread the meat over the cream mixture. Layer with shredded lettuce, 1 diced tomato, 1 diced onion, 1 diced green pepper, and shredded Colby or Cheddar cheese. Serve with chips.",
    "Pink Stuff Frozen Dessert. Mix together 1 can cherry or strawberry pie filling, 1 can crushed pineapple drained, 1 can sweetened condensed milk, 1 container (8 oz) thawed whipped topping, and 2 tablespoons lemon juice. Pour into a 9 x 13-inch pan and freeze until firm. Let stand about 15 minutes before serving.",
    "Fresh Strawberry Pie. Combine 1 1/2 cups water, 4 tablespoons cornstarch, 1 cup sugar, and 1/8 teaspoon salt in a saucepan. Bring to a boil, stirring constantly, until thick and clear. Remove from heat and stir in 4 tablespoons strawberry gelatin. Cool slightly without letting it set. Layer 1 quart cleaned strawberries in a baked pie shell, pour glaze over them, refrigerate, and serve with whipped cream.",
    "Easy German Chocolate Cake. Prepare 1/2 package chocolate fudge cake mix and 1/2 package white cake mix according to package directions, adding 1/4 cup Wesson oil. Pour into a 9 x 13-inch pan and bake as directed until done.",
    "Broccoli Salad. Trim and wash 1 large head broccoli, then cut into bite-size pieces. Add 10 slices cooked, crumbled bacon, 5 sliced green onions or 1/4 cup chopped red onion, and 1/2 cup raisins. In a separate bowl, mix 1 cup mayonnaise, 2 tablespoons vinegar, and 1/4 cup sugar. Toss dressing with broccoli mixture and refrigerate for 2 to 3 hours before serving.",
    "Strawberry Whatever. Dissolve 1 large package strawberry gelatin in 1 1/2 cups boiling water. Add 1 lb frozen strawberries in juice, 1 can crushed pineapple, 3 mashed ripe bananas, and 1 cup chopped pecans. Spread half the mixture in a pan and freeze for 30 minutes. Spread 1 pint sour cream over the top, then add remaining mixture. Chill until set.",
    "Eggless Milkless Applesauce Cake. Cream together 1/2 cup shortening and 3/4 cup sugar. Add 1 1/2 cups applesauce, 1 cup raisins, and 1 cup nuts. Sift together 2 cups flour, 3 teaspoons soda, and 1 teaspoon each cinnamon, cloves, and nutmeg, then add to the mixture. Pour into a greased pan and bake at 350°F for 45 to 60 minutes.",
    "Grandma Hanrath’s Banana Bread. Cream 1 cup sugar and 1/2 cup shortening. Add 2 eggs one at a time, then stir in 1 teaspoon salt and 2 teaspoons soda. Add 2 chopped overripe bananas, 2 cups flour, and 1/4 cup nuts. Mix well and bake in a loaf pan at 350°F for 1 hour.",
    "Chocolate Frango Mints. Mix together 1 package devil’s food cake mix, 1 package instant chocolate fudge pudding, 8 oz sour cream, 3/4 cup water, 1/2 cup Wesson oil, and 4 eggs. Beat for 5 minutes, scraping bowl often. Fold in 6 oz chopped Frango mints and bake at 350°F for 35 to 40 minutes.",
    "Cuddy Farms Marinated Turkey. Mix together 2 cups 7-Up or Sprite, 1 cup vegetable oil, 1 cup soy sauce, and garlic salt to taste. Cut a whole turkey breast into hand-sized pieces and pour marinade over the meat. Refrigerate for at least 8 hours, up to 48 hours, before cooking.",
    "Spaghetti Sauce to Can. Cook chopped peppers and onions in 1 cup oil for 30 minutes. Cook 1/2 bushel tomatoes and 1/4 cup minced garlic, then mill or process. Combine tomatoes with 6 cans tomato paste, 1 1/2 cups sugar, 1/2 cup salt, 1 tablespoon basil, 2 tablespoons oregano, and 1 teaspoon Italian seasoning. Cook 1 hour and seal in jars.",
    "Prize-Winning Meat Loaf. Mix together 1 1/2 lb ground beef, 1 cup tomato juice, 3/4 cup oats, 1 beaten egg, 1/4 cup chopped onion, 1/4 teaspoon pepper, and 1 1/2 teaspoons salt. Press into a loaf pan and bake at 350°F for 1 hour. Let rest 5 minutes before slicing.",
    "Smothered Round Steak. Dredge 2 lb round steak in flour and brown briefly in 1/2 cup vegetable oil. Add chopped onions, celery, bell peppers, beef stock, salt, black pepper, white pepper, and red pepper. Cover and cook slowly over low heat until tender.",
    "Corral Barbecued Beef Steak Strips. Brown 2 lb sliced round steak in 2 tablespoons oil. Combine 1 can tomato sauce, 1/3 cup water, 2 tablespoons brown sugar, 1 tablespoon mustard, and 1 tablespoon Worcestershire sauce. Add sauce and sliced onion to meat and simmer 30 minutes until tender.",
    "Easy Fudge. Heat 1 can sweetened condensed milk with 1 package semi-sweet chocolate chips and optional unsweetened chocolate over low heat, stirring until smooth. Remove from heat, stir in chopped nuts and 1 teaspoon vanilla, then spread into a buttered pan.",
    "Taco-Filled Green Pepper. Brown 1 lb ground beef and mix with taco seasoning, kidney beans, and 1 cup salsa. Cut 4 green peppers in half and boil for 3 minutes. Fill peppers with meat mixture and bake covered at 350°F for 15 to 20 minutes. Top with tomato, cheese, and sour cream.",
    "One Hour Rolls. Heat 1 cup milk, 2 tablespoons sugar, 1 tablespoon salt, and 3 tablespoons oil until boiling, then cool to lukewarm. Add 1 package yeast and mix into 2 cups flour. Roll dough, cut, let rise 1 hour, and bake at 350°F for 20 minutes.",
    "Punch Bowl Fruit Salad. Combine 2 cans sliced peaches, 2 cans fruit cocktail, 1 can diced pineapple with juice, 1 quart strawberries, 4 bananas, and assorted apples and grapes. Mix well and chill before serving.",
    "Summer Chicken. Marinate chicken cutlets in 1/2 cup oil, 1/3 cup red vinegar, 2 tablespoons oregano, and 2 tablespoons garlic salt. Cook as desired.",
    "Cherry Pizza. Spread 2 cans cherry pie filling in a greased baking pan. Sprinkle dry white cake mix over cherries, top with ground nuts and melted butter, then bake at 350°F for 45 minutes.",
    "Potato and Cheese Pie. Beat 3 eggs with 1 teaspoon salt and 1/4 teaspoon pepper. Stir in 2 cups half-and-half, 3 cups shredded potatoes, 1 cup Cheddar cheese, and 1/3 cup green onions. Bake at 400°F for 35 to 40 minutes.",
    "Tuna Macaroni Casserole. Prepare 1 box macaroni and cheese as directed. Stir in 1 can drained tuna, 1 jar pimentos, and 1 chopped onion. Serve hot or cold.",
    "Artichoke Dip. Drain and chop 2 cans artichoke hearts and mix with 1 cup mayonnaise and 1 cup Parmesan cheese. Bake uncovered at 350°F for 20 to 30 minutes.",
    "Broccoli Dip for Crackers. Mix 16 oz sour cream, 1 package dry vegetable soup mix, 10 oz thawed chopped broccoli, and half of 6 oz grated Cheddar cheese. Top with remaining cheese and bake at 350°F for 30 minutes.",
    "Summer Spaghetti. Cook 1 lb thin spaghetti, drain, and toss with butter. Marinate overnight with Salad Supreme seasoning and Italian dressing. Add vegetables before serving.",
    "Pear-Lime Salad. Drain 1 can pear halves and reserve juice. Boil juice and dissolve lime gelatin in it. Cool slightly, then mix with softened cream cheese and lemon yogurt. Stir in pears and chill until set.",
    "Watermelon Rind Pickles. Cut 7 lb watermelon rind into cubes and parboil until tender. Boil 7 cups sugar with 2 cups vinegar, 1/2 teaspoon oil of cloves, and 1/2 teaspoon oil of cinnamon. Pour over rind, soak overnight, reheat twice, then seal in jars.",
    "Chicken Stew. Remove meat from 3 lb boiled chicken and reserve the broth. In a large pot combine the chicken, 4 diced medium potatoes, 2 chopped onions, 1 can (16 oz) creamed corn, 1 can English peas, 1 can field peas, 1 can butter beans, 1 can tomatoes, 1 can (46 oz) tomato juice, and 1 small box macaroni. Add 1 tablespoon salt, 1 tablespoon black pepper, and 1 tablespoon sugar. Cook over medium heat until all vegetables are tender.",
    "Quick Coffee Cake. Cream together 3 tablespoons softened butter or margarine and 1/2 cup granulated sugar. Add 1 egg and beat until creamy. Mix 1 cup flour, 1 1/4 teaspoons baking powder, and 1/4 teaspoon salt, then add alternately with 1/3 cup milk. Stir in 1/4 teaspoon vanilla. Spread batter in a greased 8-inch square pan. Mix 1/3 cup packed brown sugar with 1/2 teaspoon cinnamon, sprinkle on top, dot with 2 tablespoons butter, and bake at 350°F for 25 to 30 minutes.",
    "Angel Biscuits. Mix together 5 cups flour, 3 tablespoons sugar, 4 teaspoons baking powder, and 1 teaspoon soda. Cut in 1 1/2 sticks margarine. Dissolve 1 1/2 packages dry yeast in 1/2 cup warm water, stir into 2 cups buttermilk, then combine with dry ingredients. Cover and chill before baking as biscuits.",
    "Quick Peppermint Puffs. Dip 8 marshmallows in 2 tablespoons melted margarine, then roll in 1/4 cup crushed peppermint candy. Wrap each marshmallow completely with crescent roll dough from 1 can. Dip again in margarine, place in greased muffin tin, and bake at 375°F for 10 to 15 minutes.",
    "Fruit Pizza. Mix together 1 box yellow cake mix, 1/4 cup water, 1/4 cup brown sugar, 1/2 cup soft butter, 2 eggs, and 1/2 cup chopped nuts. Spread into greased pizza pans and bake at 350°F for 15 to 20 minutes.",
    "Chocolate Mint Bars. Mix together 1 cup flour, 1 cup sugar, 1/2 cup softened oleo, 4 eggs, and 1 can chocolate syrup. Pour into a 9 x 13-inch pan and bake at 350°F for 25 to 30 minutes, until the top springs back lightly when touched.",
    "Beer Bread. Stir together 3 cups self-rising flour, 1 tablespoon sugar, and 1 can (12 oz) beer. Pour into a greased loaf pan and bake at 425°F for 50 minutes. Drizzle melted butter over the top if desired.",
    "Chicken Divan. Melt 1/4 cup margarine in a skillet and sauté 1/4 cup chopped onion and celery to taste. Remove from heat and stir in 1/4 cup flour and 1/8 teaspoon curry powder. Drain 1 can mushrooms and reserve juice, adding water to make 1/2 cup liquid. Stir liquid and 1 can cream of celery soup into mixture and cook until thick. Arrange chopped broccoli and cubed cooked chicken in a baking dish, spoon sauce over, top with shredded Monterey Jack cheese, and bake at 350°F until cheese melts.",
    "Mexican Cookie Rings. Sift together 1 1/2 cups flour, 1/2 teaspoon baking powder, and 1/2 teaspoon salt. Cream 1/2 cup butter with 2/3 cup sugar, add 3 egg yolks and 1 teaspoon vanilla. Mix in dry ingredients, shape dough into rings, dip into multi-colored candies, and bake at 375°F for 10 to 12 minutes.",
    "Vegetable-Burger Soup. Brown 1/2 lb ground beef lightly in a soup pot and drain excess fat. Add 2 cups water, 1 can stewed tomatoes, 1 can tomato sauce, 1 package onion soup mix, 1 package frozen mixed vegetables, 1 teaspoon sugar, and simmer for 20 minutes.",
]


def generate_once(recipe_text: str, max_new_tokens: int, system_text: str):
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": "Recipe text:\n" + recipe_text},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start_len = inputs["input_ids"].shape[1]

    stopping = StoppingCriteriaList([StopOnCompleteJson(tokenizer, start_len)])

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,   # <-- STOP qui
        )

    gen_ids = out[0][start_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

def try_parse(decoded: str):
    json_str = extract_first_json_block(decoded)
    if json_str is None:
        return None, "no_json", None

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return None, "invalid_json", json_str

    if set(obj.keys()) != {"title", "ingredients", "steps"}:
        return None, "bad_keys", json_str

    return obj, None, json_str

def run_extraction(recipes):
    results = []
    failures = []

    for i, recipe_text in enumerate(recipes):
        print(f"\n=== Processing recipe {i+1}/{len(recipes)} ===")

        # 1) primo tentativo
        decoded = generate_once(recipe_text, max_new_tokens=300, system_text=SYSTEM)
        obj, err, json_str = try_parse(decoded)

        # 2) retry “vero” se serve
        if obj is None:
            decoded_retry = generate_once(recipe_text, max_new_tokens=600, system_text=RETRY_SYSTEM)
            decoded = decoded_retry
            obj, err, json_str = try_parse(decoded)

        if obj is None:
            print("❌ Failed:", err)
            failures.append({
                "index": i,
                "reason": err,
                "recipe_text": recipe_text,
                "raw": decoded,
                "json_str": json_str,
            })
            continue

        results.append(obj)
        print("✅ OK:", obj.get("title", ""))

    return results, failures

if __name__ == "__main__":
    results, failures = run_extraction(RECIPES)

    with open("recipes_extracted.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open("recipes_failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print("Extracted:", len(results), "/", len(RECIPES))
    print("Failures:", len(failures))
    print("Saved: recipes_extracted.json and recipes_failures.json")