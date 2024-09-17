import pandas as pd
import seaborn as sea

df = pd.read_csv('dane/train.csv')
#print(df.head())

#Impresión de las columnas y sus tipos de Datos
#print(df.info())

#Los Datos mas importantes serian SalePrice, YrSold, OverallQual, OverallCond, LotArea, Neighbborhood debido a que nuestro
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

sea.boxplot(data=df, x="SalePrice", y="YrSold")

