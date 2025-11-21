# ===========================================
# 🤖 train.py — Model Training 
# ===========================================

# importing all the libraries 

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


df_model=pd.read_csv('loan_approval_dataset.csv')
# print(df.head())

# Strip any accidental spaces in column names
df_model.columns = df_model.columns.str.strip()

# Initialize label encoder
le = LabelEncoder()

# Encode the categorical columns (binary)
categorical_cols = ['education', 'self_employed', 'loan_status']
for col in categorical_cols:
    df_model[col] = le.fit_transform(df_model[col])

# print(df['education'])

# dependant Variable 
    target = 'loan_status'
    features = [c for c in df_model.columns if c != target]

# print(df.columns)

    y=df_model['loan_status']

    X = df_model[features]
    y = df_model[target]

    print(X)
    print(y)


# Train-test spliting

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

# Scale numeric columns in one scale
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    scaler = StandardScaler().fit(X_train[num_cols])
    X_train[num_cols] = scaler.transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    print(X_train)
    print("*"*50)
    
    print(y_train)
 
    
    # scaler.fit(X_train, y_train)
    
    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "DT": DecisionTreeClassifier(random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC(random_state=42)
    }
    

    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        results[name] = {
        "train_acc": accuracy_score(y_train, y_pred_train),
        "test_acc": accuracy_score(y_test, y_pred_test)}
        
        # Save model
        joblib.dump(model, f'loan_approval_dataset2.csv.joblib')

        # Save scaler
        joblib.dump(scaler, 'loan_approval_dataset2.csvscaler.joblib')
        
        joblib.dump(features, "loan_approval_dataset2.features.joblib")


        # Print results
        for name, r in results.items():
            print(name, "train_acc={:.4f}, test_acc={:.4f}".format(r['train_acc'], r['test_acc']))
            
            
# train and test accuracy 
from sklearn.metrics import accuracy_score

# y_train, y_test, y_pred_train, y_pred_test are from training code

print("This is Training accuracy")
train_acc = accuracy_score(y_train, y_pred_train)


print("This is Testing accuracy")
test_acc  = accuracy_score(y_test, y_pred_test)


print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)
