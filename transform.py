import pandas as pd
import seaborn as sea
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import winsorize
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.preprocessing import MinMaxScaler as MinMax_scaler

df = pd.read_csv('dane/train.csv')
#print(df.head())

#Impresión de las columnas y sus tipos de Datos
#print(df.info())

#Los Datos mas importantes serian SalePrice, YrSold, OverallQual, OverallCond, LotArea, Neighborhood debido a que nuestro
#enfoque seria saber cuantas casas se han vendido, los años mas populares, el precio mas comun, Condicion de la casa,
#Area del lote y Vecindario que se encuentra.

#Cantidad de variables nulas y las columnas
#percent = df.isnull().sum() * 100 / len(df)
#missingtable = pd.DataFrame({'percent': percent})
#print(missingtable, "\n")
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
#sea.boxplot(data=df, x="YrSold", y="SalePrice")
#plt.show()

#sea.boxplot(data=df, x="OverallQual", y="LotArea")
#plt.show()

#sea.boxplot(data=df, x="Neighborhood", y="OverallCond")
#plt.show()

#Tratamiento de outliers

#Normalización y estandarización
df['SalePrice_Scaled'] = scaler.fit_transform(df[['SalePrice']])
df['SalePrice_MinMax'] = MinMax_scaler.fit_transform(df[['SalePrice']])
df['SalePrice_Log'] = np.log(df['SalePrice'])

df['SalePrice_Scaled'] = scaler.fit_transform(df[['OverallQual']])
df['SalePrice_MinMax'] = MinMax_scaler.fit_transform(df[['OverallQual']])
df['SalePrice_Log'] = np.log(df['OverallQual'])

df['SalePrice_Scaled'] = scaler.fit_transform(df[['LotArea']])
df['SalePrice_MinMax'] = MinMax_scaler.fit_transform(df[['LotArea']])
df['SalePrice_Log'] = np.log(df['LotArea'])

df['SalePrice_Scaled'] = scaler.fit_transform(df[['OverallCond']])
df['SalePrice_MinMax'] = MinMax_scaler.fit_transform(df[['OverallCond']])
df['SalePrice_Log'] = np.log(df['OverallCond'])

df['SalePrice_Scaled'] = scaler.fit_transform(df[['YrSold']])
df['SalePrice_MinMax'] = MinMax_scaler.fit_transform(df[['YrSold']])

