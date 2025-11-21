import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# -----------------------------------------
# 1. Load processed dataset (without loan_id)
# -----------------------------------------
df = pd.read_csv('loan_approval_dataset2.csv')
print("Processed Data Loaded:")
print(df.head())

# -----------------------------------------
# 2. Split features and target
# -----------------------------------------
target = 'loan_status'
X = df.drop(columns=[target])
y = df[target]

# -----------------------------------------
# 3. Train-test split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -----------------------------------------
# 4. Scale numeric columns
# -----------------------------------------
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

scaler = StandardScaler()
scaler.fit(X_train[num_cols])

X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# -----------------------------------------
# 5. Models to train
# -----------------------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(probability=True, random_state=42)
}

results = {}

best_model = None
best_accuracy = 0

# -----------------------------------------
# 6. Train models + evaluate
# -----------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    results[name] = (train_acc, test_acc)

    print(f"{name}: Train={train_acc:.4f}  Test={test_acc:.4f}")

    # choose best model based on test accuracy
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model = model
        best_model_name = name

# -----------------------------------------
# 7. Save best model + scaler + feature names
# -----------------------------------------
print(f"\nBest Model = {best_model_name}, Test Accuracy = {best_accuracy:.4f}")

joblib.dump(best_model, "best_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(list(X.columns), "model_features.joblib")

print("\nModel, scaler, and feature list saved successfully!")