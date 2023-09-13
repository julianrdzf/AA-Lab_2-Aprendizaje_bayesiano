import pandas as pd
from sklearn.model_selection import train_test_split
from bayes import BayesPredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_wpp_data

FILENAME = 'Datos/chat_big.txt'

data=load_wpp_data(FILENAME)

data, drop = train_test_split(data, test_size=0.90, random_state=45)

train, test = train_test_split(data, test_size=0.15, random_state=45)
#print(data.head)

SPANISH_DICT_FILENAME='Datos/es.txt'
palabras_validas=set()   
with open(SPANISH_DICT_FILENAME, 'r', encoding='utf-8') as archivo:
    palabras_validas=set()
    for linea in archivo:
        palabra = linea.strip()
        palabras_validas.add(palabra) 


m=2
Ns=[1,2,3,4]
predicciones = []
resultados= []
comparaciones = []
for N in Ns:
    HORIZONTE=N
    
    predictor=BayesPredictor(train["palabras"],HORIZONTE, palabras_validas=palabras_validas) 
    
    test_real=[]
    prediccion=[]
    for frase in test['palabras']:
        # print()
        # print('Frase nueva:')
        # print(frase)
        for i in range(1,len(frase)):
            
            #print()
            
            frase_pred=frase[max(0,i-HORIZONTE):i]
            test_real.append(frase[i])
            pred=predictor.predict(frase_pred,verbose=False)
            prediccion.append(pred)
            
            # print('Frase a predecir:')
            # print(frase_pred)
            # print('Real:')
            # print(frase[i])
            # print('Predicción:')
            # print(pred)
    
          
    comparacion=[]        
    for i in range(len(test_real)):
        if test_real[i]==prediccion[i]:
            comparacion.append(True)
        else:
            comparacion.append(False)
    
    predicciones.append(prediccion)
    resultados.append(sum(comparacion))
    comparaciones.append(comparacion)    
    
#%%    
plt.plot(np.array(resultados)/len(test_real)*100)
plt.xticks(range(N), range(1,N+1))
plt.xlabel('N')
plt.ylabel('Porcentaje de aciertos [%]')

#%%
frecuencias_N=[]
frecuencia=np.zeros(len(predicciones[0]))
for pred_N in predicciones:
    for idx, palabra in enumerate(pred_N):
        frecuencia[idx]=predictor.priori[palabra]
    frecuencias_N.append(frecuencia.copy())

res_1=np.array(frecuencias_N[0])-np.array(frecuencias_N[1])

df = pd.DataFrame(data={'predicciones':predicciones[0],'comparaciones': comparaciones[0],'frecuencias': frecuencias_N[0]})
aciertos=pd.DataFrame()
aciertos['mean'] = df.groupby('comparaciones')['frecuencias'].mean()
aciertos['std'] = df.groupby('comparaciones')['frecuencias'].std()
aciertos.plot(kind='bar', y='mean', yerr='std', title = "Promedio y desviación estándar de las frecuencias de las palabras",color='grey', legend=False)
plt.xticks(ticks=[False, True], labels=['error', 'acierto'], rotation=0 )
plt.xlabel('Comparaciones')
plt.ylabel('Frecuencia')