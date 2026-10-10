import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("cleaned_beauty_products.csv")

df["rating_category"] = df["rating"].apply(
    lambda x: "High" if x >= 4.0 else "Low"
)

X = df[["price", "reviews"]]
y = df["rating_category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("5-Fold CV Scores:", scores)
print("Mean CV Accuracy:", scores.mean())
