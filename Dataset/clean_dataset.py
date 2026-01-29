import pandas as pd

input_path = r"C:\Users\tomas\Desktop\universita\Magistrale\Secondo Anno\Primo Semestre\Deep Learning\Progetto\Dataset\RecipeNLG_dataset.csv"
output_path = r"C:\Users\tomas\Desktop\universita\Magistrale\Secondo Anno\Primo Semestre\Deep Learning\Progetto\Dataset\RecipeNLG_clean.csv"

ds = pd.read_csv(
    input_path,
    engine="python",
    sep=",",
    quotechar='"',
    doublequote=True,
    on_bad_lines="skip",
    dtype=str,
    keep_default_na=False
)

# rinomina la colonna id (se nel file è "Unnamed: 0")
if "Unnamed: 0" in ds.columns:
    ds.rename(columns={"Unnamed: 0": "id"}, inplace=True)

# tieni solo le colonne che ti servono (aggiungi/togli a piacere)
cols = ["id", "ingredients", "directions"]
cols = [c for c in cols if c in ds.columns]  # evita errori se una colonna manca
ds_clean = ds[cols].copy()

# salva
ds_clean.to_csv(output_path, index=False)

print("Salvato:", output_path)
print("Righe, Colonne:", ds_clean.shape)
print("Colonne:", ds_clean.columns.tolist())
