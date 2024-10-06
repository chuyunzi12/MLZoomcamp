import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('D:/MLZoomcamp/laptops.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
df_filtered = df[['ram', 'storage', 'screen', 'final_price']]

#EDA
plt.hist(df_filtered['final_price'], bins=50)
plt.show()

# #q1
# df_filtered.isnull().sum()
#
# #q2
# df_filtered['ram'].median()
#
# #q3
# df_shuffled = shuffle(df_filtered, random_state=42)
#
# n = len(df_shuffled)
# n_train = int(n * 0.6)
# n_val = int(n * 0.2)
# n_test = n - n_train - n_val
#
# df_train = df_shuffled[:n_train]
# df_val = df_shuffled[n_train:n_train+n_val]
# df_test = df_shuffled[n_train+n_val:]
# ## with 0
# df_train_0 = df_train.fillna(0)
# df_val_0 = df_val.fillna(0)
# ## with mean
# mean_val = df_train.mean()
# df_train_mean = df_train.fillna(mean_val)
# df_val_mean = df_val.fillna(mean_val)
#
# def train_model(df_train, df_val):
#     X_train = df_train[['ram', 'storage', 'screen']]
#     y_train = df_train['final_price']
#
#     X_val = df_val[['ram', 'storage', 'screen']]
#     y_val = df_val['final_price']
#
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, predictions))
#     return rmse
#
# rmse_0 = train_model(df_train_0, df_val_0)
# rmse_mean = train_model(df_train_mean, df_val_mean)
#
# print('RMSE with 0:', round(rmse_0, 2))
# print('RMSE with mean:', round(rmse_mean, 2))
#
# #q4
# def train_ridge_model(df_train, df_val, r):
#     X_train = df_train[['ram', 'storage', 'screen']]
#     y_train = df_train['final_price']
#
#     X_val = df_val[['ram', 'storage', 'screen']]
#     y_val = df_val['final_price']
#
#     model = Ridge(alpha=r)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, predictions))
#     return rmse
#
# r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
# for r in r_values:
#     rmse = train_ridge_model(df_train_0, df_val_0, r)
#     print(f'RMSE for r={r}:', round(rmse, 2))
# #q5
# seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# rmses = []
#
# for seed in seeds:
#     df_shuffled = shuffle(df_filtered, random_state=seed)
#
#     # Train/Val/Test split
#     df_train = df_shuffled[:n_train]
#     df_val = df_shuffled[n_train:n_train+n_val]
#
#     # Train model
#     rmse = train_model(df_train.fillna(0), df_val.fillna(0))
#     rmses.append(rmse)
#
# std = np.std(rmses)
# print('Standard Deviation:', round(std, 3))

#q6

# Split the data into train (60%), validation (20%), and test (20%)
train_data, temp_data = train_test_split(df_filtered, test_size=0.4, random_state=9)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=9)

# Combine train and validation datasets
train_val_data = pd.concat([train_data, val_data])

train_val_data_0 = train_val_data.fillna(0)
test_data_0 = test_data.fillna(0)

# Train a Ridge regression model with r=0.001 and evaluate on the test set
ridge_model = Ridge(alpha=0.001)
X_train_val = train_val_data_0[['ram', 'storage', 'screen']]
y_train_val = train_val_data_0['final_price']

X_test = test_data_0[['ram', 'storage', 'screen']]
y_test = test_data_0['final_price']

# Fit the model
ridge_model.fit(X_train_val, y_train_val)

# Make predictions on the test set
test_predictions = ridge_model.predict(X_test)

# Calculate RMSE on the test set
rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))

# Output the RMSE result
print('Test RMSE:', round(rmse_test, 2))
