# ========================================
# evaluate.py
# ========================================
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset
df_model = pd.read_csv('loan_approval_dataset2.csv')

# Clean up column names (remove leading/trailing spaces)
df_model.columns = df_model.columns.str.strip()

# Define target and feature columns
target = 'loan_status'
features = [c for c in df_model.columns if c != target]

X = df_model[features]
y = df_model[target]

# load scaler and a model (eg 'rf')
scaler = joblib.load('loan_approval_dataset2.csvscaler.joblib')
model = joblib.load('loan_approval_dataset2.csv.joblib')

# Identify numeric columns and apply scaling safely
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
X_scaled = X.copy()
# X_scaled[num_cols] = scaler.transform(X[num_cols])

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------------

scaler.fit(X_train[num_cols])

X_train[num_cols]=scaler.transform(X_train[num_cols])
X_test[num_cols]=scaler.transform(X_test[num_cols])

model.fit(X_train, y_train)

# -----------------------------------------------------------------------

# Predict
y_pred = model.predict(X_test)

# Evaluation results
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred, digits=4))
