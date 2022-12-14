import pandas as pd

df = pd.read_csv('../teststats249.csv')
# print(df.columns)
# print(df['action'])
nonAttack = df.loc[df['action'] != 3]
print(nonAttack)