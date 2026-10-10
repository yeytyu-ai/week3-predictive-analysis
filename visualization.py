import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Load the cleaned dataset

df = pd.read_csv("cleaned_beauty_products.csv")

Create rating categories

df["rating_category"] = df["rating"].apply(
lambda x: "High" if x >= 4.0 else "Low"
)

Plot 1: Rating category distribution

df["rating_category"].value_counts().plot(kind="bar")
plt.title("Distribution of Rating Categories")
plt.xlabel("Rating Category")
plt.ylabel("Number of Products")
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()

Plot 2: Price vs Reviews

sns.scatterplot(data=df, x="price", y="reviews", hue="rating_category")
plt.title("Product Price vs Number of Reviews")
plt.tight_layout()
plt.savefig("price_vs_reviews.png")
plt.show()

print("Visualizations created successfully!")
