import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dados = pd.read_excel("D02_Boston.xlsx")
x = dados.iloc[:,1:-1].to_numpy()
y = dados.iloc[:,-1].to_numpy()

x,xteste,y,yteste = train_test_split(x,y,test_size=0.7,random_state=0)

regressor = LinearRegression()
regressor = regressor.fit(xteste,yteste)

yresposta = regressor.predict(xteste)
#print(yresposta)
#print(yteste)

mse = mean_squared_error(yteste,yresposta)
rmse = math.sqrt(mse)
r2 = r2_score(yteste,yresposta)
print(rmse)
print(r2)

plt.scatter(x=yteste,y=yresposta)
plt.show()

# print(dados.columns)

# atributo_selec = "CRIM"
for a in dados.columns:

    dados.plot.scatter(x=a,y='target')

    print('pearson %.3f' % pearsonr(dados[a],dados['target'])[0])
    
plt.show()
