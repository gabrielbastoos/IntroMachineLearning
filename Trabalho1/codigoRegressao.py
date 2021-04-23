# -*- coding: utf-8 -*-
"""
@author: Gabriel Brito Bastos
"""

#--------------------IMPORTACAO DAS BIBLIOTECAS------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error,r2_score
# import math


#--------------------IMPORTACAO DOS DADOS DE ENTRADA ----------------------------
dadosTreinamento = pd.read_csv("conjunto_de_treinamento.csv", index_col = 0)
dadosTeste = pd.read_csv("conjunto_de_teste.csv", index_col = 0)
dadosExemplo = pd.read_csv("exemplo_arquivo_respostas.csv", index_col = 0)
print("Arquivos importados com sucesso\n")
#--------------------PREENCHIMENTO DOS DADOS NULOS/NAN0---------------------------
dadosTreinamento = dadosTreinamento.fillna(0)
dadosTeste = dadosTeste.fillna(0)

atributos_para_retirar = ['grau_instrucao', 'estado_onde_nasceu', 'estado_onde_reside',
                              'possui_telefone_celular', 'qtde_contas_bancarias_especiais', 
                              'estado_onde_trabalha', 'codigo_area_telefone_trabalho', 
                              'codigo_area_telefone_residencial', 'local_onde_reside', 
                              'local_onde_trabalha']

#--------------------RETIRANDO ATRIBUTOS NULOS DOS DADOS----------------------------
dadosTreinamento = dadosTreinamento.drop(atributos_para_retirar, axis=1)
dadosTeste = dadosTeste.drop(atributos_para_retirar, axis=1)

#--------------------MAPEAMENTO DE RESPOSTAS PARA VALORES NUMERICOS ----------------
dadosTreinamento['possui_telefone_residencial'] = np.where(dadosTreinamento['possui_telefone_residencial'] == 'Y' , 1, dadosTreinamento['possui_telefone_residencial'])
dadosTreinamento['possui_telefone_residencial'] = np.where(dadosTreinamento['possui_telefone_residencial'] == 'N' , 0, dadosTreinamento['possui_telefone_residencial'])
dadosTeste['possui_telefone_residencial'] = np.where(dadosTeste['possui_telefone_residencial'] == 'Y' , 1, dadosTeste['possui_telefone_residencial'])
dadosTeste['possui_telefone_residencial'] = np.where(dadosTeste['possui_telefone_residencial'] == 'N' , 0, dadosTeste['possui_telefone_residencial'])

dadosTreinamento['vinculo_formal_com_empresa'] = np.where(dadosTreinamento['vinculo_formal_com_empresa'] == 'Y' , 1, dadosTreinamento['vinculo_formal_com_empresa'])
dadosTreinamento['vinculo_formal_com_empresa'] = np.where(dadosTreinamento['vinculo_formal_com_empresa'] == 'N' , 0, dadosTreinamento['vinculo_formal_com_empresa'])
dadosTeste['vinculo_formal_com_empresa'] = np.where(dadosTeste['vinculo_formal_com_empresa'] == 'Y' , 1, dadosTeste['vinculo_formal_com_empresa'])
dadosTeste['vinculo_formal_com_empresa'] = np.where(dadosTeste['vinculo_formal_com_empresa'] == 'N' , 0, dadosTeste['vinculo_formal_com_empresa'])

dadosTreinamento['possui_telefone_trabalho'] = np.where(dadosTreinamento['possui_telefone_trabalho'] == 'Y' , 1, dadosTreinamento['possui_telefone_trabalho'])
dadosTreinamento['possui_telefone_trabalho'] = np.where(dadosTreinamento['possui_telefone_trabalho'] == 'N' , 0, dadosTreinamento['possui_telefone_trabalho'])
dadosTeste['possui_telefone_trabalho'] = np.where(dadosTeste['possui_telefone_trabalho'] == 'Y' , 1, dadosTeste['possui_telefone_trabalho'])
dadosTeste['possui_telefone_trabalho'] = np.where(dadosTeste['possui_telefone_trabalho'] == 'N' , 0, dadosTeste['possui_telefone_trabalho'])

dadosTreinamento['sexo'] = np.where(dadosTreinamento['sexo'] == 'M' , 1, dadosTreinamento['sexo'])
dadosTreinamento['sexo'] = np.where(dadosTreinamento['sexo'] == 'F' , 2, dadosTreinamento['sexo'])
dadosTreinamento['sexo'] = np.where(dadosTreinamento['sexo'] == ' ' , 3, dadosTreinamento['sexo'])
dadosTreinamento['sexo'] = np.where(dadosTreinamento['sexo'] == 'N' , -1, dadosTreinamento['sexo'])
dadosTeste['sexo'] = np.where(dadosTeste['sexo'] == 'M' , 1, dadosTeste['sexo'])
dadosTeste['sexo'] = np.where(dadosTeste['sexo'] == 'F' , 2, dadosTeste['sexo'])
dadosTeste['sexo'] = np.where(dadosTeste['sexo'] == ' ' , 3, dadosTeste['sexo'])
dadosTeste['sexo'] = np.where(dadosTeste['sexo'] == 'N' , -1, dadosTeste['sexo']) 


dadosTreinamento['forma_envio_solicitacao'] = np.where(dadosTreinamento['forma_envio_solicitacao'] == 'internet' , 0, dadosTreinamento['forma_envio_solicitacao'])
dadosTreinamento['forma_envio_solicitacao'] = np.where(dadosTreinamento['forma_envio_solicitacao'] == 'presencial' , 1, dadosTreinamento['forma_envio_solicitacao'])
dadosTreinamento['forma_envio_solicitacao'] = np.where(dadosTreinamento['forma_envio_solicitacao'] == 'correio' , 2, dadosTreinamento['forma_envio_solicitacao'])
dadosTeste['forma_envio_solicitacao'] = np.where(dadosTeste['forma_envio_solicitacao'] == 'internet' , 0, dadosTeste['forma_envio_solicitacao'])
dadosTeste['forma_envio_solicitacao'] = np.where(dadosTeste['forma_envio_solicitacao'] == 'presencial' , 1, dadosTeste['forma_envio_solicitacao'])
dadosTeste['forma_envio_solicitacao'] = np.where(dadosTeste['forma_envio_solicitacao'] == 'correio' , 2, dadosTeste['forma_envio_solicitacao'])
print("Mapeamento de variáveis string para número feito com sucesso\n")

#--------------------RANDOMIZACAO DOS DADOS--------------------------------
dadosTreinamento = dadosTreinamento.sample(frac=1).reset_index(drop=True)

#--------------------DIVISAO DOS DADOS ENTRE VARIAVEIS E ROTULO-------------
y_treinamento = dadosTreinamento['inadimplente']
x_treinamento = dadosTreinamento.drop(['inadimplente'], axis=1)

#--------------------NORMALIZACAO DO DADO----------------------------------
escala = StandardScaler()
escala.fit(x_treinamento)

x_treinamento = escala.transform(x_treinamento)
x_teste_predicao = escala.transform(dadosTeste)
print("Normalizacao dos dados feita com sucesso\n")
print("Tratamento de dados feito com sucesso\n")
#--------------------PLOTTING DO HISTOGRAMA DOS DADOS----------------------
distribuicao = dadosTeste.hist(bins=10)
#plt.show()
print("Classificando utilizando regressor linear....\n")
#--------------------REGRESSAO LINEAR --------------------------------------
regressor_linear= LinearRegression()
regressor_linear.fit(x_treinamento,y_treinamento)
y_prob_linear = regressor_linear.predict(x_teste_predicao)
y_predicao_linear = np.where(y_prob_linear > 0.482, 1, 0) 
print('\tR2 da Regressão Linear:',regressor_linear.score(x_teste_predicao, y_predicao_linear),'\n\n')


print("Classificando utilizando regressor logístico....\n")
#--------------------REGRESSAO LOGISTICA -----------------------------------
regressor_logistico = LogisticRegression()
regressor_logistico.fit(x_treinamento,y_treinamento)
y_prob_logistic = regressor_logistico.predict_proba(x_teste_predicao)[:,1]
y_predicao_logistic = np.where(y_prob_logistic > 0.484, 1, 0) 
print('\tR2 da Regressão Logística:',regressor_logistico.score(x_teste_predicao, y_predicao_logistic),'\n\n')


print("Classificando utilizando regressor KNN com n=20....\n")
#--------------------REGRESSAO KNN COM N=20 ----------------------------------
regressor_knn= KNeighborsRegressor(n_neighbors=20,weights='distance')
regressor_knn.fit(x_treinamento,y_treinamento)
y_prob_knn = regressor_knn.predict(x_teste_predicao)
y_predicao_knn = np.where(y_prob_knn > 0.484, 1, 0) 
print('\tAcurácia da Regressão KNN:',regressor_knn.score(x_teste_predicao, y_predicao_knn),'\n\n')


print("Exportando os dados para arquivo csv....\n\n")
#--------------------EXPORTACAO DA PREDICAO PARA CSV---------------------------
resposta = pd.DataFrame(index=dadosExemplo.index)
resposta['inadimplente'] = y_predicao_linear
resposta.to_csv('arquivo_resposta.csv')
print("Arquivo 'arquivo_resposta.csv' gerado com sucesso")