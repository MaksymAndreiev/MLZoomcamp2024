import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

data = pd.read_csv('housing.csv')

plt.hist(data['median_house_value'], bins=50)
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.title('Distribution of Median House Value')
plt.show()

filtered_data = data[data['ocean_proximity'].isin(['<1H OCEAN', 'INLAND'])]

selected_columns = ['latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                    'households', 'median_income', 'median_house_value']
subset_data = filtered_data[selected_columns]

missing_feature = subset_data.columns[subset_data.isnull().any()]
print("The feature with missing values is:", missing_feature)

population_median = subset_data['population'].median()
print("The median for the variable 'population' is:", population_median)

np.random.seed(42)

n = len(subset_data)

n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)

idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = subset_data.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train + n_val].copy()
df_test = df_shuffled.iloc[n_train + n_val:].copy()

df_train['median_house_value'] = np.log1p(df_train['median_house_value'])
df_val['median_house_value'] = np.log1p(df_val['median_house_value'])
df_test['median_house_value'] = np.log1p(df_test['median_house_value'])

# Option 1: Fill missing values with 0
train_data_fill_0 = df_train.fillna(0)
val_data_fill_0 = df_val.fillna(0)

# Option 2: Fill missing values with the mean of the variable (computed using the training data only)
mean_value = df_train['total_bedrooms'].mean()
train_data_fill_mean = df_train.fillna(mean_value)
val_data_fill_mean = df_val.fillna(mean_value)

X_train_fill_0 = train_data_fill_0.drop('median_house_value', axis=1)
y_train_fill_0 = train_data_fill_0['median_house_value']
X_val_fill_0 = val_data_fill_0.drop('median_house_value', axis=1)
y_val_fill_0 = val_data_fill_0['median_house_value']

X_train_fill_mean = train_data_fill_mean.drop('median_house_value', axis=1)
y_train_fill_mean = train_data_fill_mean['median_house_value']
X_val_fill_mean = val_data_fill_mean.drop('median_house_value', axis=1)
y_val_fill_mean = val_data_fill_mean['median_house_value']


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)


def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


w0_fill_0, w_fill_0 = train_linear_regression(X_train_fill_0, y_train_fill_0)
y_pred_fill_0 = w0_fill_0 + X_val_fill_0.dot(w_fill_0)
rmse_fill_0 = rmse(y_val_fill_0, y_pred_fill_0)

w0_fill_mean, w_fill_mean = train_linear_regression(X_train_fill_mean, y_train_fill_mean)
y_pred_fill_mean = w0_fill_mean + X_val_fill_mean.dot(w_fill_mean)
rmse_fill_mean = rmse(y_val_fill_mean, y_pred_fill_mean)

print("RMSE with missing values filled with 0:", round(rmse_fill_0, 2))
print("RMSE with missing values filled with mean:", round(rmse_fill_mean, 2))

X_train_fill_0 = X_train_fill_0.fillna(0)
X_val_fill_0 = X_val_fill_0.fillna(0)

r_values = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]

best_rmse = float('inf')
best_r = None

for r in r_values:
    w0_fill_0, w_fill_0 = train_linear_regression_reg(X_train_fill_0, y_train_fill_0, r=r)

    y_pred = w0_fill_0 + X_val_fill_0.dot(w_fill_0)

    rmse = np.sqrt(mean_squared_error(y_val_fill_0, y_pred))

    if rmse < best_rmse:
        best_rmse = rmse
        best_r = r

best_rmse = round(best_rmse, 2)

print("Best r:", best_r)
print("Best RMSE:", best_rmse)

seed_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_scores = []

for seed in seed_values:
    np.random.seed(seed)

    n = len(subset_data)

    n_val = int(0.2 * n)
    n_test = int(0.2 * n)
    n_train = n - (n_val + n_test)

    idx = np.arange(n)
    np.random.shuffle(idx)

    df_shuffled = subset_data.iloc[idx]

    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train:n_train + n_val].copy()
    df_test = df_shuffled.iloc[n_train + n_val:].copy()

    df_train['median_house_value'] = np.log1p(df_train['median_house_value'])
    df_val['median_house_value'] = np.log1p(df_val['median_house_value'])
    df_test['median_house_value'] = np.log1p(df_test['median_house_value'])

    train_data_fill_0 = df_train.fillna(0)
    val_data_fill_0 = df_val.fillna(0)

    X_train_fill_0 = train_data_fill_0.drop('median_house_value', axis=1)
    y_train_fill_0 = train_data_fill_0['median_house_value']
    X_val_fill_0 = val_data_fill_0.drop('median_house_value', axis=1)
    y_val_fill_0 = val_data_fill_0['median_house_value']


    def train_linear_regression(X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)

        return w[0], w[1:]


    def rmse(y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)


    w0_fill_0, w_fill_0 = train_linear_regression(X_train_fill_0, y_train_fill_0)
    y_pred_fill_0 = w0_fill_0 + X_val_fill_0.dot(w_fill_0)
    rmse_fill_0 = rmse(y_val_fill_0, y_pred_fill_0)
    rmse_scores.append(rmse_fill_0)

std = np.std(rmse_scores)
std = round(std, 3)
print("Standard deviation of RMSE scores:", std)

np.random.seed(9)

n = len(subset_data)

n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)

idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = subset_data.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train + n_val].copy()
df_test = df_shuffled.iloc[n_train + n_val:].copy()

df_train['median_house_value'] = np.log1p(df_train['median_house_value'])
df_val['median_house_value'] = np.log1p(df_val['median_house_value'])
df_test['median_house_value'] = np.log1p(df_test['median_house_value'])

train_val_data = pd.concat([df_train, df_val])
train_val_data_fill_0 = df_train.fillna(0)
test_data_fill_0 = df_test.fillna(0)

X_train_val_fill_0 = train_val_data_fill_0.drop('median_house_value', axis=1)
y_train_val_fill_0 = train_val_data_fill_0['median_house_value']
X_test_fill_0 = test_data_fill_0.drop('median_house_value', axis=1)
y_test_fill_0 = test_data_fill_0['median_house_value']

w0_fill_0, w_fill_0 = train_linear_regression_reg(X_train_val_fill_0, y_train_val_fill_0, r=0.001)

y_pred_fill_0 = w0_fill_0 + X_test_fill_0.dot(w_fill_0)

rmse_fill_0 = np.sqrt(mean_squared_error(y_test_fill_0, y_pred_fill_0))

print("RMSE on the test dataset:", round(rmse_fill_0, 2))
