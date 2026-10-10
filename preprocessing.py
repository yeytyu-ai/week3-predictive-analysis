import pandas as pd

Load the original dataset

df = pd.read_csv("beauty_products.csv")

Check missing values

print("Missing values before cleaning:")
print(df.isnull().sum())

Remove duplicate rows

df = df.drop_duplicates()

Fill missing numerical values

for column in ["price", "rating", "reviews"]:
df[column] = df[column].fillna(df[column].median())

Fill missing category values

df["category"] = df["category"].fillna(df["category"].mode()[0])

Save the cleaned dataset

df.to_csv("cleaned_beauty_products.csv", index=False)

print("Data preprocessing completed!")
print("Cleaned dataset shape:", df.shape)
