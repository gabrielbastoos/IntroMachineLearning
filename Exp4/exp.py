import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer,MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt

dados = pd.read_csv('Orange_Telecom_Churn_Data.csv')

#print(dados)

variaveis_categoricas =[
    x for x in dados.columns if dados[x].dtype =='object' or x =='area_code'
    ]   

# print(variaveis_categoricas)

# for v in variaveis_categoricas:
#     print('%15s:' %v, '%4d categorias' %len(dados[v].unique()))
#     print(dados[v].unique())

dados = dados.drop(['state','phone_number'],axis=1)
dados=pd.get_dummies(dados,columns=['area_code'])
binarizador = LabelBinarizer()

for v in ['intl_plan','voice_mail_plan']:
    dados[v] = binarizador.fit_transform(dados[v])

#print(dados.groupby(['churned']).mean().T)
   

atrib1='total_day_minutes'
atrib2='number_customer_service_calls'

cores = ['red' if x else 'blue' for x in dados['churned']]

grafico = dados.plot.scatter(
    atrib1,
    atrib2,
    c=cores,
    s=10,
    marker='o',
    alpha=0.5,
    figsize=(10,10)
)
#plt.show()

print(dados.columns)

atribs = ['account_length', 'intl_plan', 'voice_mail_plan',
       'number_vmail_messages', 'total_day_minutes', 
       'total_day_charge', 'total_eve_minutes', 
       'total_eve_charge', 'total_night_minutes',
       'total_night_charge', 'total_intl_minutes','total_intl_charge', 'number_customer_service_calls','churned'
       ]

dados=dados[atribs]
alvo='churned'

dados_embaralhados = dados.sample(frac=1)

x=dados_embaralhados.loc[:,dados_embaralhados.columns!='churned'].values
y=dados_embaralhados.loc[:,dados_embaralhados.columns=='churned'].values

dados_treinamento, dados_teste, dados_treinamento_rotulo, dados_teste_rotulo = train_test_split(x,y.ravel(),test_size=0.2,random_state=42)

# dados_treinamento = x[:4000,:]
# dados_treinamento_rotulo = y[:4000].ravel()

# dados_teste = x[4000:,:]
# dados_teste_rotulo = y[4000:].ravel()

adjust = MinMaxScaler()
adjust.fit(dados_treinamento)
dados_treinamento = adjust.transform(dados_treinamento)
dados_teste = adjust.transform(dados_teste)

 
vetor=[]

for i in range(1,20): 
    classificador = KNeighborsClassifier(n_neighbors=i,weights='uniform',p=1)
    classificador = classificador.fit(dados_treinamento,dados_treinamento_rotulo)

    dadoEstimado = classificador.predict(dados_teste)
    #print(dadoEstimado,dados_teste_rotulo)

    print("Acurácia:",100*sum(dadoEstimado==dados_teste_rotulo)/len(dadoEstimado),"%")
    print("N_neighbors:",i)
    vetor.append(100*sum(dadoEstimado==dados_teste_rotulo)/len(dadoEstimado))

for i in range(1,20): 
    classificador = KNeighborsClassifier(n_neighbors=i,weights='uniform',p=1)
    #classificador = classificador.fit(dados_treinamento,dados_treinamento_rotulo)

    #dadoEstimado = classificador.predict(dados_teste)
    score = cross_val_score(classificador,x,y.ravel(),cv=5)
    #print(dadoEstimado,dados_teste_rotulo)
    print('i = %2d'%i,'scores = ',score,'acuracia media = %6.1f' % (100*sum(score)/5))
    #print("Acurácia com validação cruzada:",100*sum(dadoEstimado==dados_teste_rotulo)/len(dadoEstimado),"%")
    #print("N_neighbors:",i)
    #vetor.append(100*sum(dadoEstimado==dados_teste_rotulo)/len(dadoEstimado))