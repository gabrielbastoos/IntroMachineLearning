import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv('iris.data',delimiter=",",decimal=".")
#print(dados.groupby('species').describe())
# resultado = dados.groupby('species').agg(
#     {
#         x:['median','mean','std'] for x in dados.columns if x != "species"
#     }
# )

classes = list(dados['species'].unique())
print(classes)

rotulos = dados.iloc[:,-1]
atributos = dados.iloc[:,:-1]
cores = ['red','green','magenta']

cores_amostras = [cores[classes.index(r)] for r in rotulos] 
print(cores_amostras)

pd.plotting.scatter_matrix(
    atributos,
    c=cores_amostras,
    figsize=(11,11),
    marker="o",
    s=30,
    alpha=0.5,
    diagonal='kde',
    hist_kwds={"bins":20}
)
# grafico = dados.plot.scatter('petal_width','petal_length')
# grafico.set(
#     title="Gaussiana",
#     xlabel="Comprimento da pétala em CM",
#     ylabel="Número de amostras"
# )
plt.show()