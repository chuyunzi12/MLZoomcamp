import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sqlalchemy import column

df_bank = pd.read_csv('D:/MLZoomcamp/bank+marketing/bank/bank.csv')
df_bank_full = pd.read_csv('D:/MLZoomcamp/bank+marketing/bank/bank-full.csv', sep=';')

df = df_bank_full.drop(columns=['default', 'loan'])
df.isnull().sum()

#q1
most_frequent_education = df['education'].mode()[0]
print(most_frequent_education)

#q2
numerical_features = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_features.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# 找出相关性最大的两对特征
biggest_correlation = correlation_matrix.unstack().sort_values(ascending=False)
biggest_correlation = biggest_correlation[biggest_correlation < 1].head(1)
print("Highest correlation between features:", biggest_correlation)

# 对目标变量 'y' 进行编码，'yes' 替换为 1，'no' 替换为 0
df['y'] = df['y'].map({'yes': 1, 'no': 0})

X = df.drop(columns=['y'])  # 特征
y = df['y']  # 目标变量

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 打印数据集的大小
# print("Training set size:", X_train.shape)
# print("Validation set size:", X_val.shape)
# print("Test set size:", X_test.shape)

#q3
# 选择分类变量
categorical_features = ['contact', 'education', 'housing', 'poutcome']

# 将分类变量转换为类别编码
X_train_categorical = X_train[categorical_features].copy()

# 对每个分类变量进行类别编码
for col in categorical_features:
    X_train_categorical[col] = X_train_categorical[col].astype('category').cat.codes

# 计算互信息分数
mutual_info = mutual_info_classif(X_train_categorical, y_train, discrete_features=True)

# 将互信息分数四舍五入到两位小数
mutual_info_rounded = [round(score, 2) for score in mutual_info]

# 输出互信息分数
mi_scores = dict(zip(categorical_features, mutual_info_rounded))
print("Mutual Information Scores:", mi_scores)

# 找出互信息分数最大的分类变量
best_feature = max(mi_scores, key=mi_scores.get)
print(best_feature)

#q4
# Select categorical and numerical features
categorical_features = ['job', 'marital', 'education', 'contact', 'housing', 'poutcome']
numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Define preprocessing: OneHotEncoder for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Define logistic regression model with the specified parameters
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42))
])

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the validation data
y_val_pred = model.predict(X_val)

# Calculate accuracy and round to 2 decimal places
accuracy = round(accuracy_score(y_val, y_val_pred), 2)
print(accuracy)

#5
# 选择数值型和分类型特征
numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'contact', 'housing', 'poutcome']

# 用于存储消除每个特征后对准确率的影响
accuracy_differences = {}

# 遍历每个特征
for feature in ['age', 'balance', 'marital', 'previous']:
    # 修改数值型或分类型特征集，排除当前特征
    modified_numerical_features = [f for f in numerical_features if f != feature]
    modified_categorical_features = [f for f in categorical_features if f != feature]

    # 定义处理器，进行 OneHot 编码和数值型特征的传递
    modified_preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', modified_numerical_features),
            ('cat', OneHotEncoder(), modified_categorical_features)
        ]
    )

    # 定义逻辑回归模型
    modified_model = Pipeline(steps=[
        ('preprocessor', modified_preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42))
    ])

    # 训练模型
    modified_model.fit(X_train, y_train)

    # 在验证集上进行预测
    y_val_pred_modified = modified_model.predict(X_val)

    # 计算准确率
    modified_accuracy = accuracy_score(y_val, y_val_pred_modified)

    # 计算准确率差异
    accuracy_differences[feature] = accuracy - modified_accuracy

# 打印消除各特征后的准确率差异
for feature, diff in accuracy_differences.items():
    print(f"Feature: {feature}, Accuracy Difference: {diff}")

#6
# 选择数值型和分类型特征
numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'contact', 'housing', 'poutcome']

# 定义处理器，将数值型特征保留，分类型特征进行 OneHot 编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# 定义不同 C 值的集合
C_values = [0.01, 0.1, 1, 10, 100]

# 初始化一个字典来存储每个 C 值对应的准确率
accuracy_for_C = {}

# 遍历每个 C 值，训练模型并计算验证集的准确率
for C in C_values:
    # 定义逻辑回归模型，带上不同的 C 值
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42))
    ])

    # 训练模型
    model.fit(X_train, y_train)

    # 预测验证集
    y_val_pred = model.predict(X_val)

    # 计算并存储验证集准确率（保留 3 位小数）
    accuracy = round(accuracy_score(y_val, y_val_pred), 3)
    accuracy_for_C[C] = accuracy

# 找到具有最高准确率的 C 值
best_C = max(accuracy_for_C, key=accuracy_for_C.get)

# 打印每个 C 值的准确率以及最佳 C 值
print("Accuracy for each C value:", accuracy_for_C)
print("Best C value:", best_C)

