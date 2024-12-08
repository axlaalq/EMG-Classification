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
for movimiento in range(0,20,2):
    #Defnimos que solo usaremos los datos de la mano izquierda de cada sujeto
    for sujeto in range(0,48):
        #Vector de entrada
        x=lsen[0][sujeto][0][movimiento]
        '''for i in range(0,8):
            for j in range(0,3200):
                x[i][j]=(x[i][j]-vmin)/(vmax-vmin)'''
        for i in range(0,8):
            for j in range(0,3200):
                sig=sig+((x[i][j]-prom)**2)
        X.append(np.array(x))
        #Vector esperado
        y=int(movimiento/2)
        Y.append(y)
print(sig)
sig=np.sqrt((sig)/(20*48*8*3200))
print (sig)
Y=np.array(Y)
X=np.array(X)
np.delete(X, (48,97,241), axis=0)
sig=np.arange(0,3200,1)
'''vec=stats.zscore()'''
'''plt.plot(sig,vec, color='b')'''

'''plt.subplot(8, 1, 1)
plt.plot(sig, X[5][5][:3200], 'b')
plt.title('Señal electromiográfica para el movimiento de expansión para diferentes canales')
plt.xlabel('4a')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.minorticks_on()


plt.subplot(8, 1, 2)
plt.plot(sig, X[90][5][:3200], 'k')
plt.xlabel('Tiempo (ms)')
plt.minorticks_on()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('4a')


plt.subplot(8, 2, 1)
plt.plot(sig, X[120][][:3200], 'r')
plt.xlabel('Tiempo (ms)')
plt.minorticks_on()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('4c')

plt.subplot(8, 2, 2)
plt.plot(sig, X[250][3][:3200], 'c')
plt.xlabel('Tiempo (ms)')
plt.minorticks_on()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('4d')'''
'''
plt.subplot(8, 1, 5)
plt.plot(sig, X[250][4][:3200], 'g')
plt.xlabel('Tiempo (ms)')
plt.minorticks_on()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylabel('Canal 5')

plt.subplot(8, 1, 6)
plt.plot(sig, X[250][5][:3200], 'y')
plt.xlabel('Tiempo (ms)')
plt.minorticks_on()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylabel('Canal 6')

plt.subplot(8, 1, 7)
plt.plot(sig, X[250][6][:3200], 'darkorange')
plt.xlabel('Tiempo (ms)')
plt.minorticks_on()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylabel('Canal 7')

plt.subplot(8, 1, 8)
plt.plot(sig, X[250][7][:3200], 'm')
plt.xlabel('Tiempo (ms)')
plt.minorticks_on()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.ylabel('Canal 8')'''

'''plt.show()'''

'''import statistics

list = X[218][3][:3200]
print("List : " + str(list))

st_dev = statistics.pstdev(list)
print("Standard deviation of the given list: " + str(st_dev))'''
