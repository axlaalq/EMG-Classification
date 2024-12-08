import numpy as np
#Función sigmoide
def sigmoide(a):
    a
    f=1/(1+np.exp(-a))
    return f
#Derivada de la función sigmoide
def dsigmoide(a):
    df=sigmoide(a)*(1-sigmoide(a))
    return df
#Función tanh
def tanh(a):
    return (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
#derivada de tanh:
def dtanh(a):
    return 1-((tanh(a))**2)
#Función de error medio cuadrado
def J(Y,T):
    J=0
    for i in range(0,len(Y)):
        Ji=(Y[i]-T[i])**2
        J=J+Ji
    J=J/len(T)
    return J
#Derivada del error medio cuadrado
def dJ(Y,T):
    dJ=0
    for i in range(0,len(Y)):
        Ji=(Y[i]-T[i])
        dJ=dJ+Ji
    dJ=2*(J/len(T))
    return dJ
#Función softmax
def softmax(X):
    S=[]
    s=0
    for i in range(0,len(X)):
        softmax=np.exp(X[i])
        s=s+softmax
        S.append(softmax)
    S=np.array(S)/s
    return S
#Derivada de la función softmax
def dsoftmax(X):
    dS=[]
    s=0
    for i in range(0,len(X)):
        softmax=X[i]*np.exp(X[i])
        s=s+softmax
        dS.append(softmax)
    dS=np.array(S)/s
    return dS
#Cross-entropy loss function
def cross_entropy(Y,T):
    H=0
    for i in range(0,len(T)):
        H=H-T[i]*np.log(Y[i])
    return H
#Derivada de cross-entropy loss function
def dcross_entropy(Y,T):
    dH=[]
    for i in range(0,len(T)):
        dH.append(Y[i]-T[i])
    return dH
#Función RELU
def relu(x):
    if x<=0:
        f=0
    if x>0:
        f=x
    return f
#Derivada de la función RELU
def drelu(x):
    if x<=0:
        f=0
    if x>0:
        f=1
    return f
