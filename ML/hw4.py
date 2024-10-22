import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# Load the dataset
df = pd.read_csv('D:/MLZoomcamp/bank+marketing/bank/bank.csv', sep=';')

# Select only the required columns
columns = [
    'age', 'job', 'marital', 'education', 'balance', 'housing', 'contact',
    'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'
]
df = df[columns]

# Split the data into train/validation/test with a 60%/20%/20% distribution
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)  # 0.25 of 80% is 20%

# Check the sizes of each dataset
print(f"Training set size: {len(df_train)}")
print(f"Validation set size: {len(df_val)}")
print(f"Test set size: {len(df_test)}")

# Convert the target variable 'y' into binary (1 for 'yes', 0 for 'no')
df_train['y'] = (df_train['y'] == 'yes').astype(int)

# List of numerical variables to evaluate
numerical_vars = ['balance', 'day', 'duration', 'previous']

# Calculate the AUC for each numerical variable
for col in numerical_vars:
    auc = roc_auc_score(df_train['y'], df_train[col])
    if auc < 0.5:
        # If AUC is less than 0.5, invert the variable
        auc = roc_auc_score(df_train['y'], -df_train[col])
    print(f'{col}: AUC = {auc}')

# Split the dataset into train, validation, and test sets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# Prepare the data for training (One-hot encoding using DictVectorizer)
train_dicts = df_train.drop(columns=['y']).to_dict(orient='records')
val_dicts = df_val.drop(columns=['y']).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

y_train = df_train['y'].values
y_val = df_val['y'].values

# Train the Logistic Regression model
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# Predict probabilities for the validation set
y_pred = model.predict_proba(X_val)[:, 1]

# Calculate the AUC score
val_auc = roc_auc_score(y_val, y_pred)
val_auc_rounded = round(val_auc, 3)

val_auc_rounded

# Define thresholds from 0.0 to 1.0 with step 0.01
thresholds = np.arange(0.0, 1.01, 0.01)

# Lists to store precision and recall for each threshold
precisions = []
recalls = []

# Compute precision and recall for each threshold
for t in thresholds:
    y_pred_binary = (y_pred >= t).astype(int)
    precisions.append(precision_score(y_val, y_pred_binary))
    recalls.append(recall_score(y_val, y_pred_binary))

# Plot Precision and Recall curves
plt.plot(thresholds, precisions, label='Precision', color='b')
plt.plot(thresholds, recalls, label='Recall', color='r')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.title('Precision and Recall Curves')
plt.show()

# Find the threshold where precision and recall intersect
for t, p, r in zip(thresholds, precisions, recalls):
    if abs(p - r) < 0.01:  # Consider them equal if the difference is less than 0.01
        print(f'Precision and Recall intersect at threshold: {t}')
        break

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# List to store AUC scores for each fold
auc_scores = []

# Perform KFold cross-validation
for train_idx, val_idx in kf.split(df_full_train):
    df_fold_train = df_full_train.iloc[train_idx]
    df_fold_val = df_full_train.iloc[val_idx]

    # Prepare the data for training (One-hot encoding using DictVectorizer)
    train_dicts = df_fold_train.drop(columns=['y']).to_dict(orient='records')
    val_dicts = df_fold_val.drop(columns=['y']).to_dict(orient='records')

    X_fold_train = dv.fit_transform(train_dicts)
    X_fold_val = dv.transform(val_dicts)

    y_fold_train = df_fold_train['y'].values
    y_fold_val = df_fold_val['y'].values

    # Train the Logistic Regression model
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_fold_train, y_fold_train)

    # Predict probabilities for the validation set
    y_fold_pred = model.predict_proba(X_fold_val)[:, 1]

    # Calculate the AUC score for this fold
    auc = roc_auc_score(y_fold_val, y_fold_pred)
    auc_scores.append(auc)

# Calculate the standard deviation of the AUC scores
auc_std_dev = np.std(auc_scores)
auc_std_dev

C_values = [0.000001, 0.001, 1]
results = {}

# Iterate over the C values
for C in C_values:
    auc_scores = []

    # Perform KFold cross-validation
    for train_idx, val_idx in kf.split(df_full_train):
        df_fold_train = df_full_train.iloc[train_idx]
        df_fold_val = df_full_train.iloc[val_idx]

        # Prepare the data for training (One-hot encoding using DictVectorizer)
        train_dicts = df_fold_train.drop(columns=['y']).to_dict(orient='records')
        val_dicts = df_fold_val.drop(columns=['y']).to_dict(orient='records')

        X_fold_train = dv.fit_transform(train_dicts)
        X_fold_val = dv.transform(val_dicts)

        y_fold_train = df_fold_train['y'].values
        y_fold_val = df_fold_val['y'].values

        # Train the Logistic Regression model with current C value
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_fold_train, y_fold_train)

        # Predict probabilities for the validation set
        y_fold_pred = model.predict_proba(X_fold_val)[:, 1]

        # Calculate the AUC score for this fold
        auc = roc_auc_score(y_fold_val, y_fold_pred)
        auc_scores.append(auc)

    # Store the mean and standard deviation of AUC scores for this C value
    results[C] = (round(np.mean(auc_scores), 3), round(np.std(auc_scores), 3))

# Print the results
for C, (mean_score, std_dev) in results.items():
    print(f'C={C}: mean={mean_score}, std={std_dev}')

# Identify the best C value
best_C = max(results, key=lambda x: (results[x][0], -results[x][1]))
print(f'The best C value is: {best_C}')
