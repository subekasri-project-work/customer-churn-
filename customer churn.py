# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('customer_churn_data.csv')  # Use Telco or similar dataset

# Explore data
print("Dataset shape:", df.shape)
print(df.head())
print(df['Exited'].value_counts())

# Encode categorical variables
le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if df[column].nunique() == 2:
        df[column] = le.fit_transform(df[column])
    else:
        df = pd.get_dummies(df, columns=[column])

# Handle missing values
df = df.dropna()
df.columns = df.columns.str.strip()
# Split features and target
X = df.drop(['Exited', 'RowNumber', 'CustomerId'], axis=1)  # Drop unnecessary columns
y = df['Exited']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance (revealing hidden patterns)
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
plt.title('Top 15 Important Features for Churn Prediction')
plt.tight_layout()
plt.show()

# KDE plot for selected numeric features
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="Age", hue="Exited", fill=True)
plt.title('KDE Plot of Age by Churn')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

# Identify missing columns *before* dropping NaN values
missing_cols = df.columns[df.isnull().any()]

# Plot heatmap only if missing columns exist
if len(missing_cols) > 0:
    print(f"⚠️ Missing values detected in: {list(missing_cols)}")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[missing_cols].isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Values Heatmap (Only Columns with Missing Data)')
    plt.tight_layout()
    plt.show()
else:
    print("✅ No missing values found in the dataset.")


# Group by age and compute mean churn rate
age_churn = df.groupby('Age')['Exited'].mean().reset_index()

# Line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=age_churn, x='Age', y='Exited', marker='o')
plt.title('Churn Rate by Age')
plt.xlabel('Age')
plt.ylabel('Average Churn Rate')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Exited', y='Age')
plt.title('Age Distribution by Churn Status')
plt.xlabel('Churned (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Not Churned (0)', 'Churned (1)'], 
            yticklabels=['Not Churned (0)', 'Churned (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# Select relevant numeric columns for pair plot (excluding non-numeric or unnecessary columns)
numeric_columns = ['Age', 'CreditScore', 'Balance', 'Tenure', 'EstimatedSalary']

# Pair plot for selected numeric features, color-coded by churn status
sns.pairplot(df[numeric_columns + ['Exited']], hue='Exited', diag_kind='kde', markers=['o', 's'])
plt.suptitle('Pair Plot of Numeric Features by Churn Status', y=1.02)
plt.tight_layout()
plt.show()


