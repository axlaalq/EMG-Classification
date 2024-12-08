import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.utils import np_utils
from keras.utils import to_categorical
import scipy.stats as stats

mat=sio.loadmat('datasetD.mat')
lsen=mat['lSen']
X =[]
Y =[]
vmax=213.6912883660619 #max
vmin=-214.2656460876913 #min
for sujeto in range(0,48):
    #Defnimos que solo usaremos los datos de la mano izquierda de cada sujeto
    for movimiento in range(0,20,2):
        #Vector de entrada
        x=lsen[0][sujeto][0][movimiento][::][::]
        for i in range(0,8):
            for j in range(0,640):
                x[i][j]=(x[i][j]-vmin)/(vmax-vmin)
        sp=np.hsplit(x,5)
        for k in range(0,5):
            X.append(np.array(sp[k]))
            #Vector esperado
            y=int(movimiento/2)
            Y.append(y)
X=np.array(X)
Y=np.array(Y)
print (X.shape)
permutation = np.random.permutation(X.shape[0])
X = X[permutation]
Y = Y[permutation]
X_train=np.array(X[:int(4*len(X)/5)])
print(X_train.shape)

Y_train=np.array(Y[:int(4*len(X)/5)])
Y_train=to_categorical(Y_train,num_classes=10)
print (Y_train.shape)


X_test=np.array(X[int(4*len(X)/5):])
Y_test=np.array(Y[int(4*len(X)/5):])
Y_test=to_categorical(Y_test,num_classes=10)



model = Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(8,640)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
sgd = optimizers.SGD(learning_rate=8e-5)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# fit network
modelo1=model.fit(X_train, Y_train, epochs=100, batch_size=64,validation_split=0.333)
loss = modelo1.history['loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Cross entropy loss')
plt.title('Error por iteración')
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.legend()
plt.show()
plt.clf()
acc = modelo1.history['accuracy']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.title('Precisión por iteración')
plt.xlabel('Iteración')
plt.ylabel('Presición')
plt.legend()
plt.show()
'''predictions = model.predict(X_test)
mc=[[0]*10]*10
mc=np.array(mc)
Posiciones=['Initial', 'Pronotion', 'Supination', 'Extension', 'Flexion', 'Cubital', 'Radial', 'Picking', 'Closed', 'Open']
for i in range(0,100):
    mc[np.argmax(Y_test[i])][np.argmax(predictions[i])]+=1
for i in range(0,10):
    print(mc[i])
df_cm = pd.DataFrame(mc, index = [i for i in Posiciones], columns = [i for i in Posiciones])
plt.figure(figsize = (10,10))
sn.heatmap(df_cm, annot=True)
plt.title('Set de prueba')
plt.show()'''
