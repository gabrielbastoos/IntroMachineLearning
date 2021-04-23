import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler,PolynomialFeatures


dados = pd.read_excel("D02_Boston.xlsx")
x = dados.iloc[:,1:-1].to_numpy()
y = dados.iloc[:,-1].to_numpy()

x,xteste,y,yteste = train_test_split(x,y,test_size=0.7,random_state=0)

escala = StandardScaler()
escala.fit(x)

xteste = escala.transform(xteste)
x = escala.transform(x)

regressor = LinearRegression()
regressor = regressor.fit(x,y)

for knn in range(1,30):
    regressorKNN = KNeighborsRegressor(n_neighbors=knn,weights='distance')

    regressorKNN = regressorKNN.fit(x,y)
    yrespostaKNN = regressorKNN.predict(xteste)
    
    mseKNN = mean_squared_error(yteste,yrespostaKNN)
    rmseKNN = math.sqrt(mseKNN)
    #r2KNN = r2_score(yteste,yrespostaKNN)
    #print(knn)
    #print(r2KNN)
    print(knn,'RMSE DO KNN Regression:',rmseKNN)

    
yresposta = regressor.predict(xteste)
#print(yresposta)
#print(yteste)

mse = mean_squared_error(yteste,yresposta)
rmse = math.sqrt(mse)
#r2 = r2_score(yteste,yresposta)
print('RMSE DO Linear Regression:',rmse)
#print(r2)

for grau in range(1,6):
    regressorPolinomial = PolynomialFeatures(degree=grau)

    regressorPolinomial = regressorPolinomial.fit(x)
    xPoli = regressorPolinomial.transform(x)
    xtestePoli = regressorPolinomial.transform(xteste)
    
    regressorNonLinear = LinearRegression()

    regressorNonLinear = regressorNonLinear.fit(xPoli,y)

    yrespostaPoli = regressorNonLinear.predict(xPoli)
    ytestePoli = regressorNonLinear.predict(xtestePoli)
    

    msePoli = mean_squared_error(yteste,ytestePoli)
    rmsePoli = math.sqrt(msePoli)
    #r2KNN = r2_score(yteste,yrespostaKNN)
    #print(knn)
    #print(r2KNN)
    print('Grau:',grau,'\tRMSE DO Regression Linear:',rmsePoli)

plt.scatter(x=yteste,y=yrespostaKNN)
plt.scatter(x=yteste,y=ytestePoli)
plt.scatter(x=yteste,y=yresposta)
plt.show()

# print(dados.columns)

# atributo_selec = "CRIM"
# for a in dados.columns:

#     dados.plot.scatter(x=a,y='target')

#     print('pearson %.3f' % pearsonr(dados[a],dados['target'])[0])
    
# plt.show()
