import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
dados = pd.read_excel("D02_Boston.xlsx")
dados = dados.iloc[:,1:]
print(dados.columns)

atributo_selec = "CRIM"
for a in dados.columns:

    dados.plot.scatter(x=a,y='target')

    print('pearson %.3f' % pearsonr(dados[a],dados['target'])[0])
    
plt.show()
