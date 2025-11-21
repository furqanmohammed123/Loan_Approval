# ================================================
# 🖼️ Visualization — four visualizations (matplotlib only)
# ================================================

# src/05_visualizations.py
import pandas as pd
import matplotlib.pyplot as plt

df_model = pd.read_csv('loan_approval_dataset.csv')

# 1) Count of loan_status
status_counts = df_model[' loan_status'].value_counts()
plt.figure(figsize=(6,4))
plt.bar(status_counts.index.astype(str), status_counts.values)
plt.title("Loan Status Counts")
plt.xlabel("loan_status")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('vis_loan_status_counts.png')
plt.show()

# 2) Correlation heatmap (use numeric columns)
num_df = df_model.select_dtypes(include=['int64','float64'])
corr = num_df.corr()
plt.figure(figsize=(8,6))
plt.imshow(corr, interpolation='nearest', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation matrix (numeric features)")
plt.tight_layout()
plt.savefig('vis_correlation.png')
plt.show()

# 3) Boxplot of loan_amount by loan_status
plt.figure(figsize=(6,4))
# create two lists manually for plotting
approved = df_model[df_model[' loan_status']==' Approved'][' loan_amount']
rejected = df_model[df_model[' loan_status']==' Rejected'][' loan_amount']
plt.boxplot([approved, rejected], labels=['Approved','Rejected'])
plt.title('Loan Amount by Loan Status')
plt.ylabel('loan_amount')
plt.tight_layout()
plt.savefig('vis_loan_amount_boxplot.png')
plt.show()

# 4) Scatter: income_annum vs loan_amount colored by status
plt.figure(figsize=(7,5))
colors = {' Approved':'blue', ' Rejected':'red'}
for status in df_model[' loan_status'].unique():
    sub = df_model[df_model[' loan_status'] == status]
    plt.scatter(sub[' income_annum'], sub[' loan_amount'], label=status.strip(), alpha=0.6)
plt.xlabel('income_annum')
plt.ylabel('loan_amount')
plt.title('Income vs Loan Amount by Status')
plt.legend()
plt.tight_layout()
plt.savefig('vis_income_vs_loan.png')
plt.show()
