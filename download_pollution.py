import pandas as pd

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv'
df = pd.read_csv(url)
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Head:')
print(df.head(3))
print('Dtypes:')
print(df.dtypes)
print('Missing per column:')
print(df.isnull().sum())
