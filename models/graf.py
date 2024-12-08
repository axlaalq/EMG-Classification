import matplotlib.pyplot as plt
#!/usr/bin/env python
# -*- coding: utf-8 -*-
def clasificacion(Y,sujeto,movimiento):
    for i in range(0,len(Y)):
        Y[i]=Y[i]*100
    plt.bar(range(len(Y)), Y, edgecolor='black',color='g')
    '''plt.bar(Y.index(max(Y)), max(Y), edgecolor='black',color='r')'''
    Posiciones=['Initial', 'Pronotion', 'Supination', 'Extension', 'Flexion', 'Cubital', 'Radial', 'Picking', 'Closed', 'Open']
    title='Clasificación del sujeto '+str(sujeto)+' en posicion '+Posiciones[movimiento]
    plt.xticks(range(len(Posiciones)), Posiciones)
    plt.title(title)
    plt.xlabel('Posición',fontsize = 20)
    plt.ylabel('Probabilidad de que sea esa Posición (%)',fontsize = 20)
    plt.show()
