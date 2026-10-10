import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load cleaned dataset
df = pd.read_csv("cleaned_beauty_products.csv")

# Create target variable
df["rating_category"] = df["rating"].apply(
    lambda x: "High" if x >= 4.0 else "Low"
)

# Select input features and target
X = df[["price", "reviews"]]
y = df["rating_category"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create Decision Tree model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed!")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nActual Values:", list(y_test))
print("Predicted Values:", list(y_pred))
