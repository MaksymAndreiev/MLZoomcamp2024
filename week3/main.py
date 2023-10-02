import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mutual_info_score, accuracy_score, mean_squared_error
import numpy as np

df = pd.read_csv('raw.githubusercontent.com_alexeygrigorev_mlbookcamp-code_master_chapter-02-car-price_data.csv')
print(df.head())
df = df[[
    'Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders', 'Transmission Type', 'Vehicle Style', 'highway MPG',
    'city mpg', 'MSRP']]

df.columns = df.columns.str.lower().str.replace(' ', '_')
df.fillna(0, inplace=True)
df = df.rename(columns={'msrp': 'price'})

mode_transmission_type = df['transmission_type'].mode()[0]
print("1. The most frequent observation (mode) for the column 'transmission_type' is:", mode_transmission_type)

mean_price = df['price'].mean()

df['above_average'] = df['price'].apply(lambda x: 1 if x > mean_price else 0)

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_columns]

correlation_matrix = df_numeric.corr()

max_correlation = correlation_matrix.unstack().sort_values(ascending=False)
max_correlation_features = max_correlation[max_correlation < 1].index[0]

print("2. The two features with the biggest correlation are:", max_correlation_features)

from sklearn.model_selection import train_test_split

# Split the data into train, val, and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

y_train = train_df.above_average.values
y_val = val_df.above_average.values
y_test = test_df.above_average.values

del train_df['above_average']
del val_df['above_average']
del test_df['above_average']


def mutual_info_avg_score(series):
    return mutual_info_score(series, y_train)


categorical_vars = train_df.select_dtypes(include=['object']).columns

mi_score_dict = {}

for var in categorical_vars:
    mi_score = mutual_info_avg_score(train_df[var])
    mi_score_dict[var] = np.round(mi_score, 2)

for var, score in mi_score_dict.items():
    print(f"Column: {var}, MI Score: {score}")

print('3. Feature with the lowest mutual information score: ', min(mi_score_dict, key=mi_score_dict.get))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

dv = DictVectorizer(sparse=False)
categorical = categorical_vars.tolist()
numerical = numeric_columns.tolist()
numerical.remove('above_average')

train_dict = train_df[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = val_df[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_val)[:, 1]
avg_decision = (y_pred >= 1)

accuracy = accuracy_score(y_val, avg_decision)

rounded_accuracy = round(accuracy, 2)

# Print the rounded accuracy
print("4. Validation Accuracy:", rounded_accuracy)

alphas = [0, 0.01, 0.1, 1, 10]
best_alpha = None
best_mse = float('inf')

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)

    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha

print("6. Best Alpha:", best_alpha)


categories = ['year', 'engine_hp', 'transmission_type', 'city_mpg']

# cat_acc_dict = dict()
# for cat in categories:
#     small = categorical + numerical
#     small.remove(cat)
#
#     dicts_train_small = train_df[small].to_dict(orient='records')
#     dicts_val_small = train_df[small].to_dict(orient='records')
#
#     dv_small = DictVectorizer(sparse=False)
#     dv_small.fit(dicts_train_small)
#
#     X_train_small = dv_small.transform(dicts_train_small)
#     model_small = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
#     model_small.fit(X_train_small, y_train)
#
#     y_pred = model.predict_proba(dicts_val_small)[:, 1]
#     avg_decision = (y_pred >= 1)
#     accuracy = accuracy_score(y_val, avg_decision)
#     cat_acc_dict[cat] = accuracy
#
#
# print(cat_acc_dict)



