import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import scipy.stats as stats
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
mat=sio.loadmat('BancoSeñales.mat')
lsen=mat['lSenR']
sc = MinMaxScaler(feature_range=(-1,1))
#Genera las matrices de entrada y las salidas esperadas
X =[]
Y =[]
vmax=213.6912883660619 #max
vmin=-214.2656460876913 #min
for movimiento in range(0,20):
    #Defnimos que solo usaremos los datos de la mano izquierda de cada sujeto
    for sujeto in range(0,48):
        #Vector de entrada
        x=lsen[0][sujeto][0][movimiento][::][::]
        for i in range(0,8):
            for j in range(0,3200):
                x[i][j]=(x[i][j]-vmin)/(vmax-vmin)
        X.append(np.array(x))
        #Vector esperado
        y=int(movimiento/2)
        Y.append(y)
Y=np.array(Y)
X=np.array(X)


X2=[]
Y2=[]
for movimiento in range(0,20):
    for canal in range(0,8):
        for sig in range(0,3200,640):
            X_t=[]
            for sujeto in range(movimiento*48,(movimiento*48)+48):
                X_t.append(X[sujeto][canal][sig:sig+640])
            X2.append(X_t)
            Y2.append(int(movimiento/2))
X2=np.array(X2)
Y2=np.array(Y2)
'''print (X2.shape)
print(Y2)'''
#set de entrenamiento
Y2=to_categorical(Y2,num_classes=10)
np.random.seed(0)
permutation = np.random.permutation(X2.shape[0])
X2 = X2[permutation]
Y2 = Y2[permutation]

X_train=X2[:int(len(X2)*0.7)]
Y_train=Y2[:int(len(X2)*0.7)]

X_test=X2[int(len(X2)*0.7):]
Y_test=Y2[int(len(X2)*0.7):]

X_train=np.array(X_train)
'''print (X_train.shape)'''
Y_train=np.array(Y_train)

X_test=np.array(X_test)
'''print(X_test.shape)'''
Y_test=np.array(Y_test)

#guardamos los valores esperados
np.savetxt('Valores_esperados.txt', Y_test)


#Moelo de la CNN
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(48,640)))
'''model.add(Dropout(0.1))'''
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.2)) #[0.1,0.5]
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit network
modelo1=model.fit(X_train, Y_train, epochs=100, batch_size=64,validation_split=0.3)
loss = modelo1.history['loss']
epochs = range(1, len(loss) + 1)
val_loss = modelo1.history['val_loss']
val_acc = modelo1.history['val_accuracy']
#Gráfica del error y la presición
plt.plot(epochs, loss, 'b', label='Cross entropy')
plt.plot(epochs, val_loss, 'r', label='Error de validación')
plt.title('Error del modelo')
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.legend()
plt.show()
plt.clf()
acc = modelo1.history['accuracy']
plt.plot(epochs, acc, 'b', label='Exactitud de entrenamiento')
plt.plot(epochs, val_acc, 'r', label='Exactitud de validación')
plt.title('Exactitud del modelo')
plt.xlabel('Iteración')
plt.ylabel('Exactitud')
plt.legend()
plt.show()
plt.clf()


#Predicciones del modelo
predictions = model.predict(X_test)
np.savetxt('Predicciones.txt', predictions)
'''for i in range(0,len(predictions)):
    print ('Prueba ', i+1, ':')
    print (predictions[i])
    print ('Valor esperado:')
    print (Y_test[i])
    print ('--------------------------------------------')'''
mc=np.zeros((10,10))
'''mc=np.array(mc)'''
Posiciones=['Inicial', 'Pronación', 'Supinación', 'Extensión', 'Flexión', 'Cubital', 'Radial', 'P, fina', 'P. gruesa', 'Extensión']
TP=np.array([0]*10)
TN=np.array([0]*10)
FP=np.array([0]*10)
FN=np.array([0]*10)
for i in range(0,len(X_test)):
    real=np.argmax(Y_test[i])
    pred=np.argmax(predictions[i])
    '''print (real,pred)'''
    mc[real][pred]=mc[real][pred]+1.
    if real==pred:
        TP[real]+=1
        TN[:real]+=1
        TN[real+1:]+=1
    else:
        FP[pred]+=1
        FN[real]+=1
        TN[:pred]+=1
        TN[pred+1:]+=1
        TN[real]-=1
avsens=0
avspec=0
avf1=0
avfb=0
for i in range(0,10):
    total=0
    print ('Movimiento ',i+1, ': ')
    print ('TP = ',TP[i])
    print ('TN = ',TN[i])
    print ('FP = ',FP[i])
    print ('FN = ',FN[i])
    sens=TP[i]/(TP[i]+FN[i])
    print ('Sensitivity = ', sens)
    spec=TN[i]/(TN[i]+FP[i])
    print ('Specificity = ', spec)
    acc=(TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i])
    print ('Accuracy = ', acc)
    avsens=avsens+sens
    avspec=avspec+spec

    f1=TP[i]/(TP[i]+0.5*(FP[i]+FN[i]))
    avf1=avf1+f1
    print('f1 = ',f1)
    fb=5*((spec*sens)/((4*sens) + spec))
    avfb=avfb+fb
print (avsens,avspec, avf1,fb)

for i in range(0,10):
    prom=sum(mc[i])
    mc[i]=mc[i]/prom
    '''print (mc[i])'''
df_cm = pd.DataFrame(mc, index = [i for i in Posiciones], columns = [j for j in Posiciones])
plt.figure(figsize = (10,10))
sn.heatmap(df_cm, annot=True)
plt.title('Test set')
plt.show
plt.clf()

boxplots=[0]*10
for i in range(0,10):
    boxplots[i]=[]
for j in range(0,len(predictions)):
    boxplots[np.argmax(Y_test[j])].append(predictions[j][np.argmax(Y_test[j])])

# rectangular box plot
plt.boxplot(boxplots,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=Posiciones)  # will be used to label x-ticks
plt.rcParams['font.size'] = '25'
plt.title('Varianza',fontsize=25)
plt.grid(True)
plt.xlabel('Movimientos',fontsize=20)
plt.ylabel('Probabilidades predichas',fontsize=20)
plt.show
