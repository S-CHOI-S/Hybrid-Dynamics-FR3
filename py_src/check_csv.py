import pandas as pd

df = pd.read_csv("data.csv")

rows_with_1_in_A = df[df["done"] == 1]

print(rows_with_1_in_A)

