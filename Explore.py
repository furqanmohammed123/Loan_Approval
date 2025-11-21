# # src/01_explore.py
import pandas as pd

df = pd.read_csv('loan_approval_dataset.csv')

# print shape of dataset
print("Shape:", df.shape)

# column names in data
print("\nColumns (exact):")
for c in df.columns:
    print(repr(c))
print("\nSample rows:")

# first 6 rows of dataset
print(df.head(6))

# Show unique values for categorical columns
for c in df.select_dtypes(include='object').columns:
    print("\nUnique values in", repr(c), "->", df[c].unique())

# print(df.columns)


