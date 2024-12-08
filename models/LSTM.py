from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import models, layers, optimizers
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
import scipy.io as sio
from keras.layers import Dense, LSTM

#Generamos las entradas y salidas esperadas de la base de datos
mat=sio.loadmat('datasetD.mat')
lsen=mat['lSen']
X =[]
Y =[]
for sujeto in range(0,48):
    #Defnimos que solo usaremos los datos de la mano izquierda de cada sujeto
    for movimiento in range(0,20,2):
        #Vector de entrada
        x =[]
        #Valor promedio de los datos para el movimiento de un sujeto
        for i in range(0,8):
            prom=0
            for j in range(0,len(lsen[0][sujeto][0][movimiento])):
                prom=prom+lsen[0][sujeto][0][movimiento][i][j]
            prom=prom/len(lsen[0][sujeto][0][movimiento])
            x .append(prom)
        X.append(x )
        #Vector esperado
        y=int(movimiento/2)
        Y.append(y)
#Este es el set de entrenamiento, usamos los primeros 30 sujetos para hacer el entrenamiento de la red
X ,Y =np.array(X ),np.array(Y )
x_train = np.reshape(X[:300], (X[:300].shape[0], X[:300].shape[1], 1))

y_train = to_categorical(Y[:300],num_classes=10)
#Este es el test de pruebas, usamos los últimos 18 sujetos para probar la red ya entrenada
x_test =np.reshape(X[300:], (X[300:].shape[0], X[300:].shape[1], 1))

y_test =to_categorical(Y[300:],num_classes=10)

# obtenemos el conjunto para hacer las pruebas
model = models.Sequential()
model.add(layers.LSTM(256, activation='relu', input_shape=(8,1)))
model.add(layers.Dense(128, activation='relu'))
'''model.add(layers.Dense(126, activation='relu'))'''
model.add(layers.Dense(10, activation='softmax'))
opt = optimizers.Adam(learning_rate=1e-6,clipnorm=100)
model.compile(optimizer = opt,
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])
train_log = model.fit(x_train, y_train,
                      epochs=100, batch_size=1)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(test_accuracy)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(test_accuracy)
loss = train_log.history['loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Cross entropy loss')
plt.title('Error por iteración')
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.legend()
plt.show()
plt.clf()
prediction = model.predict(x_train)
'''np.argmax(predictions[0]) # clase más probable'''
print(prediction[0])
