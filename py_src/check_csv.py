import pandas as pd

df = pd.read_csv("data_test.csv", header=None)

# rows_with_1_in_A = df[df["A"] == 1]
rows_with_1_in_A = df[df[0] == 1]

print(rows_with_1_in_A)

