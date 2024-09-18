import pandas as pd
import seaborn as sea
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('dane/train.csv')
print(df.head())

#Impresión de las columnas y sus tipos de Datos
print(df.info())

#Los Datos mas importantes serian SalePrice, YrSold, OverallQual, OverallCond, LotArea, Neighborhood debido a que nuestro
#enfoque seria saber cuantas casas se han vendido, los años mas populares, el precio mas comun, Condicion de la casa,
#Area del lote y Vecindario que se encuentra.

#Cantidad de variables nulas y las columnas
percent = df.isnull().sum() * 100 / len(df)
missingtable = pd.DataFrame({'percent': percent})
print(missingtable, "\n")
print(df.columns[df.isnull().any()].tolist())

print("\n")

#Imputa de valores
#Utilizare la mediana debido a que es una forma de imputación mas consistente que la media y la moda cuando los datos
#estan sesgados o tiene valores extremos como en este Dataframe, al mismo tiempo cuando los datos faltantes son aleatorios.
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].median())
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())
print(df.head(8))

# Detección de outliers
sea.boxplot(data=df, x="YrSold", y="SalePrice")
plt.show()

sea.boxplot(data=df, x="OverallQual", y="LotArea")
plt.show()

sea.boxplot(data=df, x="OverallCond", y="LotArea")
plt.show()

#Tratamiento de outliers
df.loc[(df['YrSold'] == 2008) & (df['SalePrice'] > 325000), 'SalePrice'] = np.nan
df.loc[(df['YrSold'] != 2008) & (df['SalePrice'] > 350000), 'SalePrice'] = np.nan
sea.boxplot(x=df['YrSold'], y=df['SalePrice'])
plt.show()

df.loc[(df['OverallQual'] == 10) & (df['LotArea'] > 20000), 'LotArea'] = np.nan
df.loc[(df['OverallQual'] == 9) & (df['LotArea'] > 18000), 'LotArea'] = np.nan
df.loc[(df['OverallQual'] < 9) & (df['LotArea'] > 18000), 'LotArea'] = np.nan
sea.boxplot(x=df['OverallQual'], y=df['LotArea'])
plt.show()

df.loc[(df['OverallCond'] < 9) & (df['LotArea'] > 18000), 'LotArea'] = np.nan
sea.boxplot(x=df['OverallCond'], y=df['LotArea'])
plt.show()

#Normalización, Estandarización y  Transformación logarítmica

scaler = StandardScaler()
MinMax_scaler = MinMaxScaler()

df['SalePrice_Scaled'] = scaler.fit_transform(df[['SalePrice']])
df['SalePrice_MinMax'] = MinMax_scaler.fit_transform(df[['SalePrice']])
df['SalePrice_Log'] = np.log1p(df['SalePrice'])

df['OverallQual_Scaled'] = scaler.fit_transform(df[['OverallQual']])
df['OverallQual_MinMax'] = MinMax_scaler.fit_transform(df[['OverallQual']])
df['OverallQual_Log'] = np.log1p(df['OverallQual'])

df['LotArea_Scaled'] = scaler.fit_transform(df[['LotArea']])
df['LotArea_MinMax'] = MinMax_scaler.fit_transform(df[['LotArea']])
df['LotArea_Log'] = np.log1p(df['LotArea'])

df['OverallCond_Scaled'] = scaler.fit_transform(df[['OverallCond']])
df['OverallCond_MinMax'] = MinMax_scaler.fit_transform(df[['OverallCond']])
df['OverallCond_Log'] = np.log1p(df['OverallCond'])

df['YrSold_Scaled'] = scaler.fit_transform(df[['YrSold']])
df['YrSold_MinMax'] = MinMax_scaler.fit_transform(df[['YrSold']])

sea.histplot(df['SalePrice_Log'], kde=True)
plt.show()

#Creación de nuevas variables y Label Encoding

df['LotAreaIndividualValue'] = df['SalePrice'] / df['LotArea']
YrSold_cut = pd.qcut(df['YrSold'], q=3, labels=['Old Sale', 'Recent Sale', 'Brand New Sale'])
df['YrSold_cat'] = YrSold_cut

category_mapping = {
    'Old Sale': 0,
    'Recent Sale': 1,
    'Brand New Sale': 2
}

print(df[['YrSold', 'YrSold_cat']].sample(8))
