import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.stats as stats
import pandas as pd
mat=sio.loadmat('BancoSeñales.mat')
print(sio.whosmat('BancoSeñales.mat'))
lsen=mat['lSenR']
X =[]
Y =[]
vmax=213.6912883660619 #max
vmin=-214.2656460876913 #min
prom=-0.4407322641999223
sig=0
C=[]
count=0
i=0
for canal in range(0,8):
    C.append([])
    for movimiento in range(0,20,2):
        #Defnimos que solo usaremos los datos de la mano izquierda de cada sujeto
        for sujeto in range(0,48):
            #Vector de entrada
            x=lsen[0][sujeto][0][movimiento]
            #Vector esperado
            y=int(movimiento/2)
            Y.append(y)
            C[canal].append(x[canal])

C=np.array(C)
C=np.reshape(C,(8,int(480*3200)))

print(C.shape)

print(C[0][15],C[4][15])


data=[]
for i in range(0,8):
    data.append(C[i])

Canales=['1','2','3','4','5','6','7','8']
plt.boxplot(data,     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=Canales)  # will be used to label x-ticks
plt.rcParams['font.size'] = '25'
plt.title('Media por canales',fontsize=25)
plt.grid(True)
plt.xlabel('Canal',fontsize=20)
plt.ylabel('Amplitud',fontsize=20)
plt.show()
