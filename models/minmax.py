import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
mat=sio.loadmat('BancoSeñales.mat')
print(sio.whosmat('BancoSeñales.mat'))
lsen=mat['lSenR']
X =[]
Y =[]
vmax=0
vmin=0
for movimiento in range(0,20,2):
    #Defnimos que solo usaremos los datos de la mano izquierda de cada sujeto
    for sujeto in range(0,48):
        #Vector de entrada
        x=lsen[0][sujeto][0][movimiento]
        for i in range(0,8):
            for j in range(0,3200):
                if vmax<x[i][j]:
                    vmax=x[i][j]
                if vmin>x[i][j]:
                    vmin=x[i][j]
print(vmax)
print(vmin)
