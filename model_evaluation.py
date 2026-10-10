import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("cleaned_beauty_products.csv")

# Create target
df["rating_category"] = df["rating"].apply(
    lambda x: "High" if x >= 4.0 else "Low"
)

# Features and target
X = df[["price", "reviews"]]
y = df["rating_category"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["High", "Low"])

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["High", "Low"],
    yticklabels=["High", "Low"]
)

plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.show()

print("Graph saved as confusion_matrix.png")
