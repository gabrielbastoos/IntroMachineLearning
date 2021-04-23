#Discplina EEL 891 - Aprendizado de Maquinas
#Trabalho 1 - Classificacao: Sistema de apoio a decisao
#             para aprovacao de crédito
#
#Aluno: Arthur de Andrade Barcellos
#DRE: 115089858

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer, RobustScaler
from sklearn.ensemble import AdaBoostClassifier

#Leitura dos dados de treinamento
data_train = pd.read_csv('conjunto_de_treinamento.csv')

#Dando replace em vazios
data_train['estado_onde_nasceu'] = data_train['estado_onde_nasceu'].replace('XX','')


print ( '\nImprimir o conjunto de dados:\n')
print(data_train)

print ( '\nImprimir o conjunto de dados transposto')
print ('para visualizar os nomes de todas as colunas:\n')
print(data_train.T)

#Subistituindo os Nans 
data_train['profissao'] = data_train['profissao'].fillna(19)
data_train['ocupacao'] = data_train['ocupacao'].fillna(5)
data_train['profissao_companheiro'] = data_train['profissao_companheiro'].fillna(19)
data_train['grau_instrucao_companheiro'] = data_train['grau_instrucao_companheiro'].fillna(6)
data_train['meses_na_residencia'] = data_train['meses_na_residencia'].fillna(0)
data_train = data_train.fillna(0)

print ( '\nImprimir o conjunto de dados:\n')
print(data_train)

print ( '\nImprimir o conjunto de dados transposto')
print ('para visualizar os nomes de todas as colunas:\n')
print(data_train.T)

columns_to_drop = ['sexo','local_onde_reside','forma_envio_solicitacao','meses_no_trabalho',
                    'valor_patrimonio_pessoal','renda_mensal_regular','renda_extra','dia_vencimento',
                    'local_onde_trabalha','id_solicitante','grau_instrucao','codigo_area_telefone_residencial',
                    'possui_telefone_celular','codigo_area_telefone_trabalho','estado_onde_nasceu',
                    'estado_onde_reside','estado_onde_trabalha','possui_email','possui_cartao_visa',
                    'possui_cartao_mastercard','possui_cartao_diners','possui_cartao_amex',
                    'possui_outros_cartoes','possui_carro','profissao','ocupacao','profissao_companheiro',
                    'qtde_contas_bancarias','qtde_contas_bancarias_especiais','tipo_endereco'
                ]

#Retirando alguns dados que não acho relevante
data_train = data_train.drop(columns_to_drop, axis=1)

#Categiras para o pre processamento dos dados
categorias_one_hot = ['estado_civil','tipo_residencia']

categorias_bin = ['possui_telefone_residencial','vinculo_formal_com_empresa','possui_telefone_trabalho']

#Aplicar binarizacao e one hot
bin = LabelBinarizer()
for i in categorias_bin:
    data_train[i] = bin.fit_transform(data_train[i])

data_train = pd.get_dummies(data_train,columns=categorias_one_hot)

#Embaralhar o conjunto de dados
dados_embaralhados = data_train.sample(frac=1,random_state=100)

#Criar os arrays X e Y separando atributos e alvo
x = dados_embaralhados.loc[:,dados_embaralhados.columns!='inadimplente'].values
y = dados_embaralhados.loc[:,dados_embaralhados.columns=='inadimplente'].values

#Ajustar a escala dos atributos
ajustador_de_escala = RobustScaler()
ajustador_de_escala.fit(x)
x = ajustador_de_escala.transform(x)

quantidade_de_amostra = 5000  

x = data_train.loc[:,data_train.columns!='inadimplente'].values
y = data_train.loc[:,data_train.columns=='inadimplente'].values

x_treino = x[:quantidade_de_amostra,:]
y_treino = y[:quantidade_de_amostra].ravel()

ajustador_de_escala.fit(x_treino)
x_treino = ajustador_de_escala.transform(x_treino)

classificador = AdaBoostClassifier()
classificador = classificador.fit(x_treino, y_treino)


#Importando e tratando o conjunto de teste
data_test = pd.read_csv('conjunto_de_teste.csv')  
ids = data_test['id_solicitante']

data_test['estado_onde_nasceu'] = data_test['estado_onde_nasceu'].replace('XX', '')

data_test['profissao'] = data_test['profissao'].fillna(19)
data_test['ocupacao'] = data_test['ocupacao'].fillna(5)
data_test['profissao_companheiro'] = data_test['profissao_companheiro'].fillna(19)
data_test['grau_instrucao_companheiro'] = data_test['grau_instrucao_companheiro'].fillna(6)
data_test['meses_na_residencia'] = data_test['meses_na_residencia'].fillna(0)

data_test = pd.get_dummies(data_test,columns=categorias_one_hot)

binarizador = LabelBinarizer()
for v in categorias_bin:
    data_test[v] = binarizador.fit_transform(data_test[v])
    
data_test = data_test.drop(columns_to_drop,axis=1)

#Ajustar escala dos atributos
data_test = ajustador_de_escala.transform(data_test)

#Aplicando o classificador ja treinado
resultado = classificador.predict(data_test)

#Criando arquivo de saida
saida = {'id_solicitante': ids,
          'inadimplente': resultado}

saida = pd.DataFrame (saida, columns = ['id_solicitante', 'inadimplente'])

saida.to_csv('resultadoFinal.csv', index=False)