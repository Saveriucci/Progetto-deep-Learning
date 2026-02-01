import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =======================
# CONFIG
# =======================
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# cartella dove hai salvato adapter + tokenizer con save_pretrained(OUTPUT_DIR)
ADAPTER_DIR = r"C:\Users\tomas\Desktop\universita\Magistrale\Secondo Anno\Primo Semestre\Deep Learning\Progetto\2.Addestramento\Qwen2.5\qwen-recipe-json-model"

OUT_DIR = r"C:\Users\tomas\Desktop\universita\Magistrale\Secondo Anno\Primo Semestre\Deep Learning\Progetto\3.Valutazione\Qwen2.5"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_JSON = os.path.join(OUT_DIR, "recipes_extracted.json")
OUT_FAIL = os.path.join(OUT_DIR, "recipes_failures.json")
OUT_JSONL = os.path.join(OUT_DIR, "recipes_extracted.jsonl")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA available:", torch.cuda.is_available())

# =======================
# TEXT HELPERS
# =======================
def strip_code_fences(s: str) -> str:
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

def try_parse(decoded: str):
    json_str = extract_first_json_block(decoded)
    if json_str is None:
        return None, "no_json", None

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"invalid_json: {e}", json_str

    # alias: directions -> steps
    if "steps" not in obj and "directions" in obj:
        obj["steps"] = obj.pop("directions")

    # prendi SOLO le chiavi che ti servono (ignora eventuali extra)
    obj = {
        "title": obj.get("title", ""),
        "ingredients": obj.get("ingredients", []),
        "steps": obj.get("steps", []),
    }

    # sanity check tipi
    if not isinstance(obj["title"], str) or not isinstance(obj["ingredients"], list) or not isinstance(obj["steps"], list):
        return None, "bad_types", json_str

    return obj, None, json_str

# =======================
# LOAD TOKENIZER + MODEL + ADAPTER
# =======================
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


dtype = torch.float16 if device == "cuda" else torch.float32

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    attn_implementation="eager",
)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

if device == "cpu":
    model = model.to(device)

# =======================
# PROMPT
# =======================
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

SYSTEM += (
    "\nIMPORTANT:\n"
    "- Output MUST start with '{' as the very first character.\n"
    "- Output MUST be exactly one JSON object and nothing else.\n"
    "Example (format only):\n"
    "{\"title\":\"x\",\"ingredients\":[\"x\"],\"steps\":[\"x\"]}\n"
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

# =======================
# INCOLLA QUI LE TUE RICETTE
# =======================
RECIPES = [
    # --- LE PRIME 50 RICETTE ---
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
    "Corral Barbecued Beef Steak Strips. Brown 2 lb sliced round steak in 2 tablespoons oil. Combine 1 can tomato sauce, 1/3 cup water, 2 tablespoons brown sugar, 1 tablespoon military, and 1 tablespoon Worcestershire sauce. Add sauce and sliced onion to meat and simmer 30 minutes until tender.",
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

    # --- LE 100 NUOVE RICETTE (900-1000) ---
    "Corn Pudding. Mix 3/4 c. sugar, 2 Tbsp. flour, 1/2 tsp. salt, 2 eggs, 1 1/4 c. milk, 1/2 stick melted butter, and 1 1/2 pt. drained corn, pour into a baking dish, and bake at 375°F for 1 hour.",
    "Chicken Cordon Blue. Pound 3 chicken breasts (halves) until thin, lay 3 slices Swiss or Jack cheese and 3 slices cooked ham on the breasts, roll up and secure with toothpicks; mix 3 Tbsp. flour, 1 tsp. salt, pepper to taste, and 1 tsp. paprika in a plastic bag, add the rolls and shake to coat, then melt 3 Tbsp. butter in a frying pan and brown on all sides, add 1/2 c. white wine and 1 chicken bouillon cube, cover and simmer 30 minutes; remove chicken, add 1 c. whipping cream to the pan and cook on medium-high for 3 minutes until smooth and creamy, then pour over chicken (serves 3).",
    "Corn Casserole. In a large bowl, mix 1 can creamed corn, 1 can whole corn, 1 stick margarine (melted), and 1 box Jiffy corn bread mix (dry) until well combined, then stir in about 3/4 of the 1 (16 oz.) container fat free sour cream and mix well.",
    "Poppy Seed Bread. Mix 1 c. boiling water with 2 Tbsp. poppy seed and set aside; mix 1 pkg. Betty Crocker butter pecan cake mix, 1 small pkg. instant coconut pudding, 1/2 c. oil, and 4 eggs, then add the seed-and-water mixture, pour into 2 greased loaf pans, and bake at 325° for 1 hour (freezes well).",
    "Buttermilk Brownie Cake. In a saucepan combine 1 c. butter, 1/3 c. cocoa, and 1 cup water, bring to a boil stirring constantly, then remove from heat; in a large bowl sift together 2 c. flour, 2 c. sugar, 1 tsp. soda, and 1/2 tsp. salt, stir in 2 slightly beaten eggs, 1/2 c. buttermilk, and 1 1/2 tsp. vanilla, add the cocoa mixture and mix until blended, pour into a greased 15 x 10 x 1-inch pan and bake at 375° for 20 minutes, then immediately pour frosting over brownies and spread evenly.",
    "Seven Layer Salad. Layer lettuce, tomatoes, boiled eggs (cut), grated cheese, 2 cans English peas (drained), Miracle Whip mayonnaise, and bacon bits in the order listed, cover with Miracle Whip mayonnaise, sprinkle bacon bits on top, and refrigerate.",
    "Taco Salad. Combine 3 lb. ground beef, 2 c. chopped onions, 1 c. chopped celery, 1 c. chopped green bell pepper, 3 Tbsp. cumin seed, 3 cloves garlic (minced), and 4 tsp. chili powder in a crock-pot and mix well, add salt and pepper to taste, and cook on low 3 to 4 hours; heat 8 oz. Velveeta cheese (shredded) with 1 (10 oz.) can Ro-Tel tomatoes in a saucepan until melted, stirring constantly, then layer 18 c. chopped lettuce, the beef mixture, 8 chopped tomatoes, the cheese sauce, and 1 (16 oz.) pkg. crushed corn chips on individual plates (yield: 12 servings).",
    "Pizza Dippers. Grease 1 large cookie sheet, unroll 10 oz. can Pillsbury refrigerator pizza crust, cut the rectangle in half crosswise and then into 8 (1-inch) strips (16 total), place strips apart on the sheet, brush with 2 Tbsp. margarine, sprinkle with 1 1/2 tsp. pizza seasonings, bake at 350° for 11 to 17 minutes, sprinkle with 4 oz. (1 c.) shredded Mozzarella cheese and return to oven to melt, then serve immediately with 8 oz. warm pizza sauce for dipping (makes 16 servings).",
    "Madge’s Refrigerator Rolls. Sprinkle 1 Tbsp. sugar over 1 pkg. yeast in 1/2 c. lukewarm water and let foam; scald 1 pt. milk with 1/2 c. sugar, 1/2 c. shortening or oleo, and 1 Tbsp. salt, cool, then add the yeast, fold in 2 egg whites beaten stiff, stir in 6 c. sifted flour (dough should be soft, do not add more flour), cover and let rise until doubled, stir down, refrigerate covered overnight, then bake in muffin tins using two small balls dipped in melted oleo to form one roll.",
    "Super Sundae Pie. Stir 2 pt. (1 qt.) vanilla ice cream slightly softened until smooth, spoon into 1 (6 oz.) graham cracker pie crust and cover, freeze several hours until hard, then let stand 10 minutes to soften; spoon 1 jar strawberry topping onto the serving plate, set a slice of pie on top, garnish with 1 bottle chocolate sundae syrup plus 1/2 c. nuts, whipped cream, and maraschino cherries as desired, then return to freezer.",
    "Cranberry-Orange Relish For Poultry. Combine 2 oranges (peeled and sectioned), 1 (16 oz.) can whole cranberry sauce, 1 (16 oz.) can pear halves (drained and chopped), 1 chopped unpeeled apple, 1 c. sugar, 1/2 c. raisins, 1/4 c. chopped pecans, 1 Tbsp. vinegar, 1/2 tsp. ground cinnamon, and 1/2 tsp. ground ginger in a saucepan, bring to a boil over medium heat, then reduce heat and simmer until thick; cool and refrigerate.",
    "Swedish Heirloom Cookies. Cream 1/2 c. shortening with 1/2 c. butter (softened) until light and fluffy, add 1 c. sifted powdered sugar and 1/2 tsp. salt and mix well, stir in 2 c. all-purpose flour, then stir in 1 Tbsp. water, 1 Tbsp. vanilla extract, and 1 1/4 c. ground almonds; shape into 1-inch balls, place on ungreased baking sheets and flatten slightly, bake at 325° for 12 to 15 minutes, then dredge warm cookies in powdered sugar (yields about 8 dozen).",
    "Garden Cabbage Salad. Combine 1 small cabbage (finely shredded), 1 bunch green onions (chopped), 1 shredded carrot, 1 small green pepper (finely chopped), 6 radishes (thinly sliced), and 1 cucumber (finely chopped), cover and chill, then toss with Dressing just before serving (yields 8 to 10 servings).",
    "Kentucky Pecan Pie. Mix 1 c. white corn syrup, 1 c. brown sugar, 1/2 tsp. salt, 1/3 c. melted butter, 1 tsp. vanilla, 3 whole eggs (slightly beaten), and 1 c. pecans (finely chopped), bake 45 minutes at 350°, and decorate with whole pecan halves.",
    "London Broil. Place 1 flank steak (about 1 1/2 lb.) in a shallow pan, combine 1 c. salad oil with 1 Tbsp. vinegar and 1 clove garlic (minced) and pour over, cover and refrigerate 3 hours or overnight turning several times; broil 3 inches from heat about 5 minutes, season with salt and pepper, turn and broil 3 minutes, season again, then carve very thin slices diagonally across the grain (serves 4 to 5).",
    "Hello Dolly Bars. Melt 1 stick of Oleo, mix with 1 c. Graham Cracker Crumbs and spread in the bottom of a greased 9 x 13-inch pan, then layer 1 c. Chocolate Chips, 1 c. Butterscotch Chips, 1 c. Walnuts chopped (optional), and 1 c. Coconut (shedded), pour 1 c. Condensed Milk over top, bake at 350° for 30-35 minutes, cool, and cut into bars.",
    "Macaroni And Cheese Casserole. Mix 1 lb. sharp cheese (grated), 8 oz. macaroni (cooked and drained), 1 can condensed cream of mushroom soup, 2 oz. jar pimento, 1 c. mayonnaise, and 1/2 onion (grated), then before baking cover with crumbled cheese crackers and dot with butter, and bake at 325° for 40 minutes.",
    "Betsy’s Chocolate Fudge Cake. Melt 3 sq. unsweetened chocolate over hot water and cool; grease and flour 2 to 3 (9-inch) cake pans; sift together 2 1/4 c. sifted flour, 2 tsp. baking powder, and 1/2 tsp. salt, beat 1/2 c. butter until soft, add 2 1/4 c. packed light brown sugar and 3 eggs and beat at high speed until light and fluffy (about 5 minutes), beat in 1 1/2 tsp. vanilla and the chocolate, stir in dry ingredients alternating with 1 c. sour cream, beating well after each addition, stir in 1 c. boiling water (batter will be thin), pour into pans and bake at 350° for 35 minutes, cool 10 minutes in pans then turn out and cool completely before icing with Chocolate Fudge Frosting.",
    "Cream Cheese Squares. Mix 1 box yellow cake mix with 1 egg and 1 stick margarine, press into a 9 x 12-inch sheet cake pan, then mix and cream together 1 box 10x sugar, 2 eggs, and 8 oz. cream cheese (softened), spread over the base, bake at 350° for 40 minutes, and cut when cool.",
    "Beef Barbecue. Put 3 to 5 lb. boneless chuck roast in a pot, add 1 c. ketchup, 1/2 c. water, 1/2 tsp. mustard, 4 Tbsp. lemon juice, 2 Tbsp. vinegar, and 1 onion (finely chopped), then cook on low to medium heat 3 to 4 hours (on top of stove) until the meat will shred, adding extra water and ketchup if liquid disappears.",
    "Chicken Stroganoff. Slice 1 lb. chicken breasts into thin strips; in a skillet over medium heat heat 1 Tbsp. of the 2 Tbsp. vegetable oil, add 2 medium onions and sauté 2 minutes, add 1/2 lb. mushrooms (2 c.) thinly sliced and sauté 3 minutes, remove vegetables and reserve; heat remaining oil, add chicken and 1 tsp. minced garlic and cook as for stir-fry, return vegetables, add 1/2 c. chicken broth and 1/8 tsp. pepper, then mix 1 Tbsp. cornstarch with 2 Tbsp. water and stir into skillet, bring to a boil and boil 1 minute stirring constantly, remove from heat and stir in 1 (8 oz.) pkg. yogurt, heat gently over low (do not boil), and serve over hot cooked noodles.",
    "Cream Pie And Pudding (Microwave). Combine 3/4 c. sugar, 3/8 c. flour, and 1/4 tsp. salt in a deep 2-quart glass bowl, add 2 eggs or 3 egg yolks and 2 c. milk, beat with a whisk until smooth, then microwave at full power for 8 minutes stirring 3 times, and proceed as the recipe directs.",
    "Crispix Cookies. Melt 1 (12 oz.) pkg. butterscotch morsels with 1 (6 oz.) pkg. semi-sweet chocolate morsels in a large saucepan over low heat, stirring constantly until smooth, remove from heat and add 1 1/2 c. salted peanuts and 4 c. Kellogg's Crispix cereal, stir gently until well coated, drop by teaspoons onto waxed paper, and chill until firm.",
    "Rum Balls. Combine 3 c. finely crushed vanilla wafers with 1 c. finely chopped nuts (any kind) and 1/2 cup of the 1 1/2 c. powdered sugar in a large bowl, melt 1 c. semi-sweet chocolate with 1/2 c. light corn syrup, stir in 1/3 c. rum until smooth, pour over the crumb mixture and mix thoroughly, roll into 1-inch balls, place on wax paper, roll each ball in the remaining 1 c. powdered sugar, and store airtight for at least 3 days before serving.",
    "Millionaire Cake. Beat together 2 c. flour, 2 c. sugar, 2 tsp. soda, 2 eggs, 1 c. coconut, 1 c. chopped nuts, 1 tsp. vanilla, 1 tsp. salt, and a 20 oz. can crushed pineapple and juice, then bake in a 9 x 13-inch pan at 350° for 45 to 55 minutes.",
    "Hawaiian Carrots. Scrape and cut 1 lb. pkg. carrots into small pieces, cook and drain and salt, drain 1 (No. 2) can and 1 small can crushed pineapple, then cook 1/2 cup pineapple juice with 3 Tbsp. cornstarch, 1/2 stick oleo, and 1/2 c. brown sugar until thick, add carrots and pineapple and simmer 10 minutes (better if made a day ahead).",
    "Ginny’s Salad. Wash, dry, and break 3 to 4 bunches red leaf lettuce into bite-size pieces, toss with 12 oz. shredded Mozzarella cheese and 6 to 8 oz. chopped walnuts, then mix with 2 pkg. Good Seasons Italian dressing mixed according to directions just before serving.",
    "Onion Soup. Cook 2 large onions (thinly sliced) in 1 Tbsp. butter and 1 Tbsp. vegetable oil for 15 minutes stirring occasionally, add 1/4 tsp. sugar and cook 15 minutes until onions are brown, sprinkle with 2 Tbsp. flour and blend in 6 c. beef broth plus 1/4 c. white wine and seasonings, simmer 30 minutes; lightly toast 4 to 8 slices French bread in a 350° oven for 20 minutes, baste each slice with 1/2 teaspoon of the 2 tsp. vegetable oil and garlic, serve soup with toast floating on top, sprinkle with 1 to 2 c. grated Swiss cheese, bake 15 minutes, and serve hot (makes 4 to 8 servings).",
    "Banana Walnut Cake. Cream 3/4 c. shortening (butter) and gradually add 2 1/4 c. sugar, creaming until light and fluffy, add 3 eggs (beaten well), sift 3 c. cake flour before measuring then add 3 tsp. baking powder, 3/4 tsp. soda, and 3/4 tsp. salt, then add 6 Tbsp. sour milk alternating with the flour mixture, beating after each addition, add 1 1/2 tsp. vanilla, add 1 1/2 c. banana pulp and beat well, stir in 1 1/2 c. chopped walnuts, pour into a greased Bundt pan, and bake at 350° for 50 to 60 minutes.",
    "Fish And Broccoli Roll-Ups. Cut 16 oz. flounder or haddock into strips, cut 4 c. broccoli into small pieces, alternate broccoli pieces and rolled fish strips onto kabob spears adding 8 lemon wedges to each end, baste with 1/2 c. salad dressing, and broil 8 to 10 minutes.",
    "Fat Free Pizza Treats. Top toasted English muffins with pizza sauce, then add toppings of your choice (Parmesan cheese, grated onion, sliced mushrooms, chopped green pepper) ending with 1 Tbsp. (skim) Mozzarella cheese, and toast in a toaster oven until cheese melts.",
    "Sesame Chicken Wings. Whisk together 3 1/2 Tbsp. light soy sauce, 1/2 garlic clove (minced), 1/2 tsp. minced fresh ginger, 1 Tbsp. honey, and freshly ground black pepper to taste, add 2 lb. chicken wings with wing tips cut off and toss to coat, marinate covered and chilled for 2 hours, line a pan with foil and arrange wings drained in one layer reserving the marinade, brush with 1 Tbsp. Oriental sesame oil, sprinkle with 2 Tbsp. sesame seed toasted lightly, and bake at 400° for 30 to 35 minutes basting occasionally with the reserved marinade until browned (serves 4 as an hors d'oeuvre).",
    "Imitation Crabmeat Salad. Shred 2 lb. imitation crabmeat into thin strips about 1/4 to 1/2-inch x 3 to 4-inches long, cook 1 pkg. spiral pasta and drain, combine pasta with crabmeat and add 8 oz. sour cream, 10 oz. mayonnaise, 3 to 4 sticks celery, 3 to 4 green onions (about 1/4 c.), 6 oz. water chestnuts, 6 oz. frozen pea pods, 1/4 tsp. dill weed, 1/4 tsp. onion powder, 1/4 tsp. Beau Monde, 1/4 tsp. celery salt, and 1/4 tsp. pepper, then chill at least 1/2 hour before serving.",
    "Macaroni And Cottage Cheese. Cook 3 c. uncooked macaroni until tender and drain, mix in 1 small carton cottage cheese, brown 1/2 stick butter in a skillet and pour over, then add salt and pepper to taste and serve.",
    "String Pie. Brown 1 lb. ground beef with 1/2 c. chopped onion and 1/4 c. chopped green onion in a large skillet, stirring to separate meat, then drain fat and stir in 1 jar spaghetti sauce; in a large bowl combine 1/3 c. grated Parmesan cheese, 2 eggs (beaten), and 2 tsp. butter, mix well and spread in the bottom of a 13 x 9-inch pan, spread 1 c. cottage cheese over top, pour the sauce mixture over cottage cheese, sprinkle with 1/2 c. shredded Mozzarella cheese, and bake at 350° until cheese melts, about 20 minutes.",
    "Easters Hamburgers. Brown hamburger patties, slice raw onions on top, mix 1 can tomato sauce with 1/2 tsp. dry mustard, 2 Tbsp. brown sugar, salt and pepper, and chili powder, pour over hamburgers, and bake 1 hour at 350°.",
    "Coconut Macaroons. Combine 1 1/3 c. coconut, 2 Tbsp. all-purpose flour, 1/3 c. sugar, and 1/2 tsp. salt, stir in 2 egg whites (beaten) and 1/2 tsp. almond extract and mix well, lightly grease a cookie sheet, cut maraschino cherries into quarters, drop batter by teaspoonfuls about 1-inch apart and top each with a cherry, then bake at 325° for 20 to 25 minutes, removing when edges are golden.",
    "Cranberry Salad. Put 3 oranges, 2 apples, and 1 lb. cranberries through a food chopper, drain a No. 2 can crushed pineapple and add it, dissolve 2 small pkg. Jell-O * plus 1 1/2 c. sugar in 2 c. water (heat to dissolve thoroughly), mix into the fruit mixture, add 1 c. chopped nuts, and refrigerate about 24 hours before serving.",
    "Lazy Day Salad. Mix by hand 1 large can fruit cocktail (drained), 2 cans mandarin oranges (drained), 1 large can crushed pineapple with juice, 1 box instant vanilla pudding, and 1 box instant lemon pudding, then fold in 1 large container Cool Whip and refrigerate all day or overnight (keeps 7 to 10 days in the fridge).",
    "Pear Relish. Let 16 pears (peeled, cored and ground), 6 medium onions (ground), 8 green peppers (ground), 4 sweet red peppers (ground), and 2 hot peppers (ground) stand in separate containers for 1 hour, pour boiling water over each and drain well; combine everything in a large kettle with 6 c. vinegar (5% acidity), 6 c. sugar, 3 Tbsp. mustard seed, 2 Tbsp. pickling salt, 1 Tbsp. ground turmeric, 1 tsp. ground allspice, 1 tsp. ground cinnamon, and 1 tsp. ground ginger, bring to a boil then reduce heat and simmer uncovered 30 minutes, ladle into hot sterilized jars leaving 1/4 inch headspace, remove air bubbles and wipe rims, apply lids and bands, process in a boiling water bath 20 minutes, store in a cool place 1 month before using, and chill before serving (makes 10 pints).",
    "Strawberry Bread. Preheat oven to 350° and grease and flour two 9 x 5-inch loaf pans; in a medium bowl stir 2 1/2 c. fresh crushed strawberries or 2 (10 oz.) containers frozen strawberries (thawed) with 4 eggs and 1 1/4 c. salad oil, in a large bowl combine 3 c. flour, 2 c. sugar, 1 Tbsp. cinnamon, 1 tsp. soda, 1 tsp. salt, and 1 c. chopped nuts, add the berry mixture to the dry ingredients and stir until blended, pour into pans, and bake 1 hour or until a toothpick comes out clean.",
    "Strawberry Salad. Dissolve 2 pkg. strawberry Jello in 1 c. boiling water, add 1 large can crushed pineapple, 3 mashed bananas, 1 c. chopped pecans, and 1 to 2 c. sour cream, pour half of the mixture into a pan and let set, spread on cream, then add remaining Jello mixture.",
    "Chili Beef Casserole. Cook 1 c. macaroni in boiling salted water until done (10 to 12 minutes) and drain thoroughly, brown 3/4 lb. extra lean ground beef in a skillet, add 1 can tomato soup and 1 can chili beef soup along with the macaroni, heat 5 to 7 minutes stirring occasionally until bubbly, cut 3 slices sharp processed American cheese diagonally and place on top, cover briefly to melt, and serve at once (serves 6).",
    "Bread. Dissolve 1 pkg. yeast (RapidRise) in 2 c. warm water, add 3 Tbsp. melted butter or shortening, 1/2 c. sugar, and 1/2 tsp. salt, then add 1 to 1 1/2 c. whole wheat flour and blend with an electric mixer, gradually add bread flour (3 1/2 to 4 c., total flour about 5 c.), folding in the last flour by hand, knead on a floured surface, place in an oiled bowl and oil the top, cover with wax paper and let rise until doubled, divide into pans and let rise again, then bake at 350° for 30 minutes to 1 hour until golden brown (makes 2 large or 4 small loaves).",
    "Salmon Casserole Baked In Sour Cream. Place reserved liquid from 1 (1 lb.) can salmon in a bowl, add 1 Tbsp. lemon juice, 1/2 tsp. salt, pepper to taste, 1 c. sour cream, and 1/2 tsp. dry dill weed and mix until smooth, break salmon into large chunks and place in a greased 1-quart casserole, pour sauce over salmon, top with 1 medium onion (thinly sliced) and 1 Tbsp. chopped parsley, cover and bake at 350° for 30 minutes (yields 4 servings).",
    "Carrie’s Cabbage Soup. Combine 6 c. shredded green cabbage, 1 c. chopped onion, 2 c. diced celery, 1/2 c. diced green pepper, 1/2 tsp. thyme, 1/4 tsp. marjoram, 1 Tbsp. anisette, 6 peeled ripe red tomatoes, 2 pkg. dry onion soup mix, 2 cans beef bouillon, water to cover, 1/4 tsp. tarragon, 2 Tbsp. granulated sugar, and 8 oz. tomato sauce, bring to a rolling boil then simmer about 3 hours, adding 1 lb. ground beef (browned, optional) or 1 lb. sausage in the last 20 minutes.",
    "Chicken And Pasta Salad. Combine 1 (8 oz.) pkg. egg noodles (cooked and drained) with 3 cooked chicken breasts cut into small pieces, seedless green grapes, and sliced almonds, mix mayonnaise, Ranch dressing, creamy Italian dressing, and a little sugar to taste to form the dressing, then pour over and toss.",
    "Calico Beans. Brown 1 lb. hamburger and 1/2 lb. chopped bacon, add 1 medium chopped onion and 1/2 c. chopped celery, mix together 1 Tbsp. Worcestershire sauce, 1/2 c. brown sugar, 1 c. catsup, 1 Tbsp. liquid smoke, 1 Tbsp. mustard, and 1 Tbsp. vinegar, then combine with 1 can lima beans (drained), 1 can pork and beans, 1 can red beans, and 1 can black-eyed peas, mix with the meat and bake in a casserole dish at 350° for 45 minutes.",
    "Ham Roll-Up. Spread soft cream cheese over 1/2 lb. thin-sliced boiled ham, place a breadstick on each slice, and roll up.",
    "Penne Al’Arrabbiato. Bring water to a boil and cook pasta (DeCecco penne, plain or rigati), meanwhile sauté 3 to 4 cloves garlic in 1 to 2 Tbsp. olive oil briefly, add 2 cans (14 oz.) Italian-style chopped tomatoes, 1 Tbsp. minced parsley, whole tiny red peppers to taste, and salt, simmer in an open skillet until pasta is cooked, then toss sauce with drained pasta and garnish with fresh parsley.",
    "Corn Casserole. Mix 1 can creamed corn, 1 can whole corn (drained), 2 eggs, 1 stick oleo, 1 (8 oz.) sour cream, and 1 small box corn muffin mix, pour into a 2 to 3-quart casserole, and bake at 350° for 45 minutes.",
    "Vegetable Dip. Mix 2 c. mayonnaise, 2 c. sour cream, 1/2 c. chopped green onions, 1/2 c. chopped green pepper, 1/2 c. diced pimentos, 1/4 c. Bac*Os, 1/2 tsp. Tabasco, 1/4 tsp. garlic powder or salt, and salt and pepper to taste, then refrigerate several hours or overnight (keeps well for about 2 weeks).",
    "Spinach Dip. Thaw and drain 1 (10 oz.) pkg. frozen chopped spinach and mix with 2 (8 oz.) pkg. cream cheese, 1/2 c. chopped green parsley, 1/2 c. chopped green onions, 1/2 tsp. salt, 1/2 tsp. Season-All, 1/2 tsp. black pepper, and 1/4 tsp. garlic powder; separately slice California oranges, limes, and lemons paper thin, pour 2 bottles Sangaree Punch over an ice mold, add 5 oz. cognac, 2 oz. Triple Sec, 5 oz. vodka, and 2 oz. Meyers dark rum, add fruit after 1 hour, and serve.",
    "Easy Chicken Pot Pie. Lightly spray a 9 x 13-inch baking dish and line the bottom with approximately 2 c. cooked chicken cut into bite-size pieces, mix 1 can Veg-All vegetables with 1 can cream of chicken soup, 1 can cream of onion soup, and 1 can cream of mushroom soup and spread over chicken, then mix 1 1/2 sticks melted butter or margarine with 1 c. milk and 1 1/2 c. flour and pour over the soup layer, baking at 350° for 30 to 40 minutes until golden brown.",
    "Fruit Pizza. Spread Pillsbury sugar cookie dough onto a baking sheet, bake 12 to 13 minutes at the temperature listed on the package, cool completely, loosen with a knife, then top with 1 (8 oz.) pkg. cream cheese mixed with 1/3 c. sugar and decorate with fruit of your choice.",
    "Grasshopper Pie. Crush 22 Oreo cookies and mix with 1/4 c. margarine, press into a pie plate and bake at 350° for 10 minutes, cool completely, then fill with a mixture of 1/4 c. creme de menthe, 1 (7 oz.) jar Marshmallow Fluff, and 1 pt. whipped cream.",
    "Squash Souffle. Cook and mash 2 c. yellow squash with 2 Tbsp. butter or Land O Lakes sweet cream spread, add 4 egg whites (well beaten), 1 c. milk (1%), salt and pepper, and 1 c. Ritz cracker crumbs, pour into a Pam-sprayed casserole, top with 1 c. grated reduced fat Cheddar cheese, and bake at 350° for 30 to 45 minutes until firm (serves 6 to 8).",
    "Loris’s Egg Salad. Mash 6 hard-boiled eggs with 1 chopped sweet gherkin pickle, 1/4 c. chopped onion, and 1/4 c. chopped green pepper, stir in 1/3 c. Miracle Whip, then add 1 small jar pimento (drained), 1 tsp. salt, and a dash of paprika, chill, and serve.",
    "Apple Kuchen. Divide 1 recipe Basic Sweet Dough into 2 portions and press into two buttered 8 x 8-inch pans, press in 4 peeled and sliced apples rounded side up, sprinkle with 1/2 c. sugar mixed with 1/2 tsp. cinnamon, drizzle with 1/4 c. melted butter, let rise until doubled, then bake at 350° for 40 to 50 minutes until golden (serves 16).",
    "Eggplant-Tomato Casserole. Peel and slice 1 small eggplant into 1/4-inch rounds and peel and slice 4 tomatoes, layer eggplant in a 2-quart casserole and sprinkle with salt and pepper, top with tomatoes, oregano flakes, and some of the 1/4 c. grated Parmesan cheese, repeat layers ending with cheese, cover and bake at 400° for 30 to 40 minutes until eggplant is tender.",
    "Strawberry Bread. Combine 3 c. flour, 2 c. sugar, 1 1/2 tsp. cinnamon, 1 tsp. baking soda, and 1 tsp. salt, mix separately 1 c. cooking oil, 4 beaten eggs, and 2 c. crushed strawberries, stir wet ingredients into dry just until moistened, add 1 1/4 c. chopped walnuts or pecans if desired, pour into 2 greased loaf pans, and bake at 350° for 1 hour.",
    "Salsa. Mix 6 diced tomatoes with 1 (4 oz.) can chopped green chilies, 1 bunch green onions (chopped), 1 1/2 c. finely diced celery, 2 medium yellow onions (chopped), 1 c. chopped cilantro or parsley, 2 (15 oz.) cans tomato sauce, 1 tsp. cumin, garlic (fresh or powdered) to taste, and salt and pepper, chill 1 hour before serving and stir well before serving.",
    "Meatball Stew. Combine 2 c. cornflakes cereal, 1 egg, 1 (10 1/2 oz.) can beef broth, 1/2 tsp. salt, 1/8 tsp. pepper, and 1 lb. ground beef and form into meatballs; brown in 2 Tbsp. vegetable oil, then add 1 (10 3/4 oz.) can tomato soup, 1/2 c. water, 1/4 tsp. thyme, 2 c. frozen sliced carrots, 1 medium onion (sliced into rings), 1 can whole white potatoes (drained), and dried parsley flakes, simmer until vegetables are tender.",
    "Cauliflower-Broccoli Casserole. Simmer 1 box frozen cauliflower and 1 box frozen broccoli until tender, melt 1/2 stick butter in a skillet and add 1 c. cream of mushroom soup, 1/2 c. milk, and 8 oz. Velveeta cheese cooking until melted, combine vegetables with cheese mixture and 1 c. cooked rice, then bake uncovered at 350° for 30 minutes.",
    "Cabbage Casserole. Sauté 1/2 c. chopped celery, 1/2 c. chopped bell pepper, and 1 small chopped onion in margarine, add 3 c. shredded cabbage and 1/4 c. water, cover and steam 5 to 6 minutes, remove from heat and stir in 1 c. cream of celery soup and pimento, transfer to a casserole dish, top with Pepperidge Farm herb stuffing mixed with butter, and bake at 350° for 20 minutes.",
    "Best Ever Salad. Cook 8 oz. fine spaghetti (or 7 oz. Mexican angel hair) and cool, add 1/2 c. chopped onion, 1/2 c. chopped green pepper, 1/2 c. chopped celery, 2 c. chopped tomatoes, salt and pepper, and 1/4 tsp. curry, then stir in 1 can cream of celery soup and a small amount of mayonnaise if needed.",
    "Black Bottom Cupcakes. Mix 8 oz. cream cheese with 1 egg, 1/3 c. sugar, and 1/8 tsp. salt and stir in 6 oz. chocolate chips; separately mix 1 1/2 c. flour, 1 c. sugar, 1/4 c. cocoa, 1 tsp. baking soda, and 1/2 tsp. salt, add 1 c. water, 1/3 c. cooking oil, 1 Tbsp. vinegar, and 1 tsp. vanilla and beat well, fill cupcake liners 1/3 full with chocolate batter, top with cream cheese mixture, and bake at 350° for 30 to 35 minutes (makes 18 to 20 cupcakes).",
    "Tuna Casserole. Prepare 1 large box Kraft macaroni and cheese dinner according to package directions, cook and drain peas if frozen, sauté 3/4 c. celery and 1/2 c. onion and drain, then combine with 1 large can tuna fish, 1 can cream of mushroom soup, 2 c. peas, 1/2 c. water, and salt and pepper to taste, and bake at 350° for 35 minutes.",
    "Oatmeal Pie (Cheap Pecan Pie). Cream 2 sticks margarine with 4 eggs, add 3/4 c. uncooked quick oatmeal, 1 1/4 c. dark Karo syrup, 1 1/4 c. white sugar, 3/4 c. coconut, and 2 tsp. vanilla, pour into 2 unbaked pie shells, and bake at 325° for 1 hour.",
    "Chili Con Carne. Brown 1 lb. hamburger with 1 medium-size garlic (finely cut) and 1/2 c. chopped onion, stir in 2 to 3 tsp. chili powder, 2 tsp. salt, and 1/4 tsp. pepper, add 1 (No. 2 1/2) can tomatoes and 1 (No. 303) can kidney beans (rinsed and drained), bring to a boil, add 1/2 lb. spaghetti, reduce heat and simmer 30 minutes stirring often, adding hot water if too thick (serves 4).",
    "Swiss Chicken Casserole. Arrange 6 chicken breast halves in a lightly greased 12 x 8 x 2-inch baking dish, top each with a 4 x 4-inch slice of Swiss cheese, mix 1 (10 3/4 oz.) can cream chicken soup with 1/4 c. milk and spoon over chicken, sprinkle with 2 c. Pepperidge Farm herb stuffing mix, drizzle with 1/4 c. melted margarine, cover, and bake at 350° for 50 minutes (yields 6 servings).",
    "Happy Face Cookies. Mix 4 1/2 lb. butter with 3 lb. sugar (6 c.), add 3 c. eggs and 2 c. water, then blend in 5 1/2 lb. flour, 1/3 c. baking powder, 3 tsp. salt, 2 Tbsp. vanilla, and 2 1/3 c. powdered milk, roll out dough, cut shapes, and bake at 350°.",
    "No Bake Cookies. Melt 1 pkg. almond bark according to package directions, stir in 2 c. miniature marshmallows, 2 c. salted peanuts, 2 c. Rice Krispies, and 2 c. Cap’n Crunch, then drop and shape on waxed paper.",
    "Slinky (Apple Flavored, Low Calorie Quencher). Pour 2/3 c. sparkling mineral water and 2/3 c. unsweetened apple juice into a glass, stir in 1 Tbsp. freshly squeezed lemon juice, and garnish with a few apple slices and a sprig of mint.",
    "Peach Cobbler. Melt 1 stick butter in a baking dish, sift together 1 c. flour and 1 c. sugar, stir in 2 c. milk to form a batter, pour into the melted butter, add 2 c. fresh peaches on top without stirring, and bake at 350° for 40 to 45 minutes.",
    "Green Bean Specialty. Heat and drain 2 cans green beans, mix 4 Tbsp. bacon grease, 4 Tbsp. brown sugar, 4 Tbsp. vinegar, and 2 Tbsp. military, and heat, pour over beans, top with 4 pieces cooked and crumbled bacon, and serve hot.",
    "Raspberry Nut Perfection (Serves 8). Spread 1 (1 lb. 6 oz.) can red raspberry pie filling in an 8-inch square pan, mix 1/2 pkg. white cake mix with 1/3 c. water and 1 egg and pour over filling, sprinkle with 1/2 c. flaked coconut and 1/2 c. chopped pecans, drizzle with 1/2 c. melted butter, and bake at 350° for 40 minutes, serving warm.",
    "Mexican Chicken. Boil 1 whole chicken in water, cool and cut into pieces, place in a 9 x 13-inch dish, then layer with 1 chopped onion, 1/2 lb. cheese, 1/2 bag taco-flavored Doritos (crushed), and top with 1 can cream of chicken soup, 1 can cream of mushroom soup, and 1 can Ro-Tel.",
    "10 Minute Peanut Brittle. In a 2-quart casserole mix 1 c. sugar, 1/2 c. light corn syrup, and 1/4 tsp. salt, microwave on High 5 minutes, stir in 1 to 1 1/2 c. raw peanuts, microwave 2 to 6 minutes stirring every 2 minutes until lightly browned, then stir in 1 Tbsp. butter or margarine, 1 tsp. vanilla, and 1 tsp. soda until foamy, spread 1/4-inch thick on a well-buttered cookie sheet, cool, and break into pieces.",
    "Onion Dressing. Shake together in a covered jar 1 (14 oz.) can evaporated skim milk, 1/2 c. vinegar, 1 envelope onion soup mix, 1/3 c. catsup, 1/2 tsp. Worcestershire sauce, and 1/4 tsp. red pepper sauce (makes about 2 1/2 c.).",
    "Chocolate Oatmeal Cookies. Melt 1 stick margarine in a saucepan, add 1/2 c. milk, 1/2 c. cocoa, and 2 c. sugar and boil 1 minute, remove from heat and stir in 3/4 c. peanut butter until smooth, then add 1 tsp. vanilla, 3 c. oats, and 1/4 c. nuts if desired, and drop quickly by teaspoonfuls onto waxed paper.",
    "Pastry Crunch. Melt 1 (24 oz.) vanilla flavored candy coating in the microwave according to package directions, stir in 3 c. corn and rice cereal, 2 c. chow mein noodles, 2 c. small pretzel twists, 1 c. salted peanuts, and 1 c. pecan pieces, spread on wax paper to cool, then break into pieces and store in an airtight container.",
    "Chippewa Soup. Soak 1 lb. green split peas overnight, heat a small amount of oil in a very large pot and add 1 qt. diced carrots, 2 qt. chopped onions, and 1 qt. diced celery and sweat the vegetables, add 6 cloves garlic and cook a few minutes more, stir in 4 oz. curry powder, then add 2 qt. tomato puree, 1 gal. water, 10 oz. smoked ham, 2 diced potatoes, 1 oz. marjoram, 1 oz. basil, 1 oz. oregano, 1 oz. thyme, 4 bay leaves, the soaked peas, and salt and pepper, bring to a boil, reduce heat and simmer about 3 hours, remove bay leaves, puree the soup, chill thoroughly, and note this is a base that should be slightly salty.",
    "Coconut Cream Pie (Low Calorie). Mix 2 boxes sugar-free instant vanilla pudding with 3 c. low-fat milk, stir in 1 c. coconut, pour into 1 baked pie crust, top with 4 oz. Cool Whip (lite), sprinkle with toasted coconut, and refrigerate.",
    "Voodoo Chicken. Mix 2 c. white vinegar, 2 Tbsp. soy sauce, 1 large sliced onion, 4 cloves garlic, and 1/2 tsp. black pepper, add 2 1/2 lb. chicken pieces and marinate overnight, remove chicken, coat with a mixture of 4 Tbsp. Dijon mustard and 6 Tbsp. ketchup, bake skin-side down at 350° for 30 minutes, turn chicken, mix remaining mustard and ketchup with 2 Tbsp. milk and remaining marinade, pour over chicken, and bake another 45 minutes (total baking time 1 hour 15 minutes), serving over rice.",
    "Chicken Divan. Butter a 13 x 9-inch dish, boil and debone 2 whole chicken breasts and cut into bite-size pieces, steam 20 oz. frozen chopped broccoli and place in dish, top with chicken and 6 to 8 slices American cheese, mix 2 cans cream of celery soup with 1 tsp. curry powder and 2 tsp. lemon juice and pour over, top with 8 oz. sliced sharp Cheddar cheese, mix 1 c. plain bread crumbs with 1/2 stick melted margarine and sprinkle over casserole, then bake at 350° for 35 to 40 minutes.",
    "Old Fashioned Tea Cakes. Cream 1 c. butter with 2 c. sugar, add 4 eggs one at a time, mix in 1 Tbsp. milk, 1 tsp. nutmeg, and 1 tsp. vanilla, sift together 4 c. flour and 3 Tbsp. baking powder and combine with the mixture, add enough extra flour to roll and cut, and bake on ungreased cookie sheets at 325° for 15 minutes (yields about 4 dozen).",
    "French Toast And Sauce. Beat together 4 eggs, 1/2 c. milk, 1/2 c. thawed frozen lemonade, and 1/4 tsp. salt, melt 1 Tbsp. of the 8 Tbsp. butter in a skillet for each slice, dip 8 slices day-old bread in the egg mixture and cook until browned on both sides, serving with sauce as directed.",
    "Spaghetti Pie. Cook 6 oz. spaghetti (about 3 cups), stir 2 Tbsp. olive oil into hot spaghetti, add 1/3 c. grated Parmesan cheese and 2 well-beaten eggs, press into a 10-inch pie plate sprayed with Pam to form a crust, spread 1 c. cottage cheese or Ricotta over the bottom, brown 1 lb. low-fat ground beef with 1/2 c. chopped onion and 1/4 chopped green pepper, add about 2 c. spaghetti sauce, pour over crust, top with 1/2 c. shredded Mozzarella (2 oz.), and bake at 350°.",
    "Baked Corn. Mix 1 can whole kernel corn (drained), 1 can cream-style corn, 8 oz. sour cream, 1 stick melted oleo, and 1 box Jiffy corn bread mix, then bake at 350° for 1 hour.",
    "Chicken Casserole. Mix 2 c. diced cooked chicken, 1 c. chopped celery, 1 c. cooked rice, 1 can cream of chicken soup, 3/4 c. mayonnaise, 1 c. chopped pecans, 1/2 c. melted butter, and 1 diced onion in a buttered casserole, sprinkle grated cheese on top, and bake at 350° until bubbly, at least 1 hour (serves 4 to 6).",
    "Favorite Chocolate Cake. Sift together 1 3/4 c. flour, 1 tsp. baking powder, 1/2 tsp. salt, 1/2 tsp. soda, and 7 Tbsp. cocoa, add 1 1/2 c. sugar, 2 eggs, 1 tsp. red food coloring, 1/2 c. shortening, and 1 c. buttermilk, beat with an electric mixer 2 minutes, pour into greased and floured pans, and bake at 350° for 30 minutes (tube pan about 40 minutes).",
    "Blue Muffins. Mix 1 egg, 1 c. milk, 2 Tbsp. melted shortening, 2 Tbsp. sugar, 2 c. flour, and 2 c. blueberries thoroughly, pour into muffin pans, and bake at 350°.",
    "Devil’s Chicken (Pollo Alla Diavola). Cut open 1 (3 to 4 lb.) chicken by removing the backbone and flattening, mix 1/2 c. olive oil, 2 Tbsp. hot pepper sauce, juice of 2 lemons, ground pepper to taste, and 1 tsp. salt, marinate the chicken for 2 hours, then cook as directed.",
    "Friendship Tea. Mix 1/2 c. instant tea, 2 c. Tang, 1 pkg. Wyler’s lemonade mix (3 oz.), 1 tsp. cinnamon, and 2 c. sugar, then use 3 heaping tablespoonfuls per cup of hot water.",
    "Heath Bar Pie. Mix 3 finely chopped Heath bars with 1 medium container whipping topping, pour into 1 graham cracker pie crust, top with chocolate curls or chopped nuts, and refrigerate.",
    "Victorian Baked French Toast. Cook 1 c. brown sugar, 1/3 c. butter, and 2 Tbsp. light corn syrup until melted and pour into a 9 x 13-inch pan, arrange 8 slices French bread on top, mix 5 eggs with 1 1/2 c. milk and 1 tsp. cinnamon and pour over bread, refrigerate overnight if desired, then bake at 350° for 30 minutes until lightly browned.",
    "Quick Swedish Meatballs. Mix 1 lb. ground beef, 1 c. soft bread crumbs, 1 (3 oz.) pkg. cream cheese, 1/4 c. dry onion soup mix, 1/4 tsp. salt, 1/4 tsp. nutmeg, and 1/4 c. of the 1 1/2 c. milk, form into balls and brown lightly, place in a baking dish, blend remaining milk with 1 pkg. stroganoff sauce mix, cook until thick, pour over meatballs, and bake at 350° for 30 to 45 minutes.",
    "Irish Stew (Microwave). Combine 2 lb. lamb cubes with 1 1/4 c. water, 1 envelope onion soup mix, and 1 bay leaf in a 4-quart casserole, microwave covered on High 7 to 8 minutes, then cook on power level 3 for 35 to 40 minutes stirring occasionally, add 2 thinly sliced carrots and cook 15 minutes more, add 2 baked potatoes cubed, blend 1/4 c. flour with remaining water, stir in, and microwave 5 minutes until thickened (serves 6).",
    "Peach Salad. Dissolve 2 to 3 oz. boxes Peach Jello in 2/3 c. hot water with 1/2 c. sugar, bring to a boil, remove from heat and add 1 large jar Peach baby food and 1 large can crushed pineapple (not drained), then mix in 1 can condensed milk and 1 (8 oz.) pkg. softened cream cheese.",
    "Hidden Valley Ranch Oyster Crackers. Mix 1 pkg. Hidden Valley Ranch salad dressing mix with 3/4 to 1 c. salad oil, add 1/4 tsp. lemon pepper, 1/2 to 1 tsp. dill weed, and 1/4 tsp. garlic powder, pour over 12 to 16 oz. plain oyster crackers and stir to coat, then warm in a very low oven for 15 to 20 minutes."
]


# =======================
# GENERATION
# =======================
def generate_once(recipe_text: str, max_new_tokens: int, system_text: str):
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": "Recipe text:\n" + recipe_text},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}


    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                 # deterministico
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,

        )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def run_extraction(recipes):
    results = []
    failures = []

    for i, recipe_text in enumerate(recipes):
        print(f"\n=== Processing recipe {i+1}/{len(recipes)} ===")

        # 1) primo tentativo
        decoded = generate_once(recipe_text, max_new_tokens=1200, system_text=SYSTEM)
        obj, err, json_str = try_parse(decoded)

        # 2) retry se serve
        if obj is None:
            decoded_retry = generate_once(recipe_text, max_new_tokens=2000, system_text=RETRY_SYSTEM)
            decoded = decoded_retry
            obj, err, json_str = try_parse(decoded)

        if obj is None:
            print("❌ Failed:", err)
            failures.append({
                "index": i + 1,
                "reason": err,
                "recipe_text": recipe_text,
                "raw": decoded,
                "json_str": json_str,
            })
            continue

        results.append(obj)
        print("✅ OK:", obj.get("title", ""))

        # salva anche jsonl (comodo per debug)
        with open(OUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps({"index": i + 1, "input": recipe_text, "output": obj}, ensure_ascii=False) + "\n")

    return results, failures


if __name__ == "__main__":
    if not RECIPES:
        print("RECIPES è vuoto. Incolla la lista RECIPES nel file.")
        raise SystemExit(1)

    # azzera il jsonl ad ogni run
    if os.path.exists(OUT_JSONL):
        os.remove(OUT_JSONL)

    results, failures = run_extraction(RECIPES)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(OUT_FAIL, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print("Extracted:", len(results), "/", len(RECIPES))
    print("Failures:", len(failures))
    print(f"Saved: {OUT_JSON} and {OUT_FAIL}")
    print(f"Also saved JSONL: {OUT_JSONL}")
