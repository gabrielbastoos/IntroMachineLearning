import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

dados = pd.read_csv('iris.data')
#print(dados)
#################67% a 80% ---- tipico 75%
dados_embaralhados = dados.sample(frac=1)

dados_treinamento = dados_embaralhados.iloc[:100,:-1].values
dados_treinamento_rotulo = dados_embaralhados.iloc[:100,-1].values

dados_teste = dados_embaralhados.iloc[100:,:-1].values
dados_teste_rotulo = dados_embaralhados.iloc[100:,-1].values

vetor=[]

for i in range(1,20): 
    classificador = KNeighborsClassifier(n_neighbors=i)
    classificador = classificador.fit(dados_treinamento,dados_treinamento_rotulo)

    dadoEstimado = classificador.predict(dados_teste)
    #print(dadoEstimado,dados_teste_rotulo)

    print("Acurácia:",100*sum(dadoEstimado==dados_teste_rotulo)/len(dadoEstimado),"%")
    print("N_neighbors:",i)
    vetor.append(100*sum(dadoEstimado==dados_teste_rotulo)/len(dadoEstimado))

plt.plot(vetor)
plt.show()