import scipy.io as sio
import numpy as np
mat=sio.loadmat('datasetD.mat')
lsen=mat['lSen']
X =[]
Y =[]
for sujeto in range(0,25):
    #Defnimos que solo usaremos los datos de la mano izquierda de cada sujeto
    for movimiento in range(0,20,2):
        #Vector de entrada
        x=lsen[0][sujeto][0][movimiento][::][:260]
        X.append(np.array(x))
        #Vector esperado
        y=int(movimiento/2)
        Y.append(y)
X_train=np.array(X)
Y_train=np.array(Y)
sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)
