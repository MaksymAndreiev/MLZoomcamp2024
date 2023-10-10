import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold
from tqdm.auto import tqdm


def train(df_train, y_train, solver='liblinear', C=1.0, max_iter=1000, random_state=1):
    dv = DictVectorizer(sparse=False)
    train_dict = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    model = LogisticRegression(solver=solver, C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return dv, model


def predict(df, dv, model):
    df_dict = df.to_dict(orient='records')
    X = dv.transform(df_dict)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred


df = pd.read_csv('raw.githubusercontent.com_alexeygrigorev_mlbookcamp-code_master_chapter-02-car-price_data.csv')
df = df[[
    'Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders', 'Transmission Type', 'Vehicle Style', 'highway MPG',
    'city mpg', 'MSRP']]

df.columns = df.columns.str.lower().str.replace(' ', '_')
df.fillna(0, inplace=True)
df = df.rename(columns={'msrp': 'price'})

mean_price = df['price'].mean()
df['above_average'] = df['price'].apply(lambda x: 1 if x > mean_price else 0)
df = df.drop(['price'], axis=1)

# Split the data into train, val, and test sets
df_full_train, test_df = train_test_split(df, test_size=0.2, random_state=1)
train_df, val_df = train_test_split(df_full_train, test_size=0.2, random_state=1)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

y_train = train_df.above_average.values
y_val = val_df.above_average.values
y_test = test_df.above_average.values

auc_scores = {}
numerical_variables = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg']
numerical_column = ['year', 'engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg']

for variable in numerical_variables:
    score = roc_auc_score(train_df['above_average'], train_df[variable])
    auc_scores[variable] = score

for variable, score in auc_scores.items():
    if score < 0.5:
        score = roc_auc_score(y_train, -train_df[variable])
        auc_scores[variable] = score

highest_auc_variable = max(auc_scores, key=auc_scores.get)

print("1. The numerical variable with the highest AUC is:", highest_auc_variable)

del train_df['above_average']
del val_df['above_average']
del test_df['above_average']

dv = DictVectorizer(sparse=False)

train_dict = train_df.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

val_dict = val_df.to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)

print("2. The AUC of the logistic regression model on the validation dataset is:", round(auc, 3))

thresholds = np.arange(0.0, 1.01, 0.01)
scores = []

for threshold in thresholds:
    tp = ((y_pred >= threshold) & (y_val == 1)).sum()
    fp = ((y_pred >= threshold) & (y_val == 0)).sum()
    fn = ((y_pred < threshold) & (y_val == 1)).sum()
    tn = ((y_pred < threshold) & (y_val == 0)).sum()
    scores.append((threshold, tp, fp, fn, tn))

df_scores = pd.DataFrame(scores)
df_scores.columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
# print(df_scores.head())


df_scores['precision'] = df_scores['tp'] / (df_scores['tp'] + df_scores['fp'])
df_scores['recall'] = df_scores['tp'] / (df_scores['tp'] + df_scores['fn'])

plt.plot(df_scores['threshold'], df_scores['precision'], label='Precision')
plt.plot(df_scores['threshold'], df_scores['recall'], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.show()

absolute_difference = np.abs(df_scores['precision'] - df_scores['recall'])
intersection_threshold_index = np.argmin(absolute_difference)
intersection_threshold = thresholds[intersection_threshold_index]
print(f"3. Threshold at intersection: ~{intersection_threshold}")

df_scores['f1'] = 2 * ((df_scores['precision'] * df_scores['recall']) / (df_scores['precision'] + df_scores['recall']))
print('4. {}'.format(df_scores[df_scores['f1'] == df_scores['f1'].max()]['threshold']))

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []
for train_idx, val_idx in tqdm(kfold.split(df_full_train)):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    y_train = df_train['above_average'].values
    y_val = df_val['above_average'].values
    df_train = df_train.drop('above_average', axis=1)
    df_val = df_val.drop('above_average', axis=1)
    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
print(f'5. Standard devidation of the scores across different folds is {np.array(scores).std().round(3)}')

best_C = None
best_mean_auc = -1

for C in tqdm([0.01, 0.1, 0.5, 10]):
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        y_train = df_train['above_average'].values
        y_val = df_val['above_average'].values
        df_train = df_train.drop('above_average', axis=1)
        df_val = df_val.drop('above_average', axis=1)
        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)
    mean_auc = np.mean(scores)
    std_auc = np.std(scores)

    if mean_auc > best_mean_auc:
        best_mean_auc = mean_auc
        best_C = C
print(f'6. Best C: {best_C} (Mean AUC: {best_mean_auc:.3f})')
