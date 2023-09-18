import pandas as pd

df = pd.read_csv('housing.csv')
print(df.info())
print(len(df['ocean_proximity'].unique()))
print(df['ocean_proximity'].unique())
print(df[df['ocean_proximity'] == 'NEAR BAY']['median_house_value'].mean())
print(df[df['ocean_proximity'] == 'NEAR BAY']['median_house_value'].fillna(1000000).mean())
