import pandas as pd
from sklearn.model_selection import train_test_split
from bayes import BayesPredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from preprocess import load_wpp_data

FILENAME = 'Datos/chat_big.txt'

data=load_wpp_data(FILENAME)

data = data[:10000]

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
tiempos=[]
for N in Ns:
    HORIZONTE=N
    inicio = time.time()
    predictor=BayesPredictor(train["palabras"],HORIZONTE, palabras_validas=palabras_validas) 
    fin = time.time()
    tiempos.append(fin-inicio)
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
plt.figure()    
plt.plot(np.array(resultados)/len(test_real)*100)
plt.xticks(range(N), range(1,N+1))
plt.xlabel('N')
plt.ylabel('Porcentaje de aciertos [%]')
plt.title('Aciertos en función del hiperparámtero N')
plt.show()

plt.figure()    
plt.plot(np.array(tiempos))
plt.xticks(range(N), range(1,N+1))
plt.xlabel('N')
plt.ylabel('Tiempo [s]')
plt.title('Tiempo de entrenamiento en función del hiperparámtero N')
plt.show()

#%%
frecuencias_N=[]
frecuencia=np.zeros(len(predicciones[0]))
for pred_N in predicciones:
    for idx, palabra in enumerate(pred_N):
        frecuencia[idx]=predictor.priori[palabra]
    frecuencias_N.append(frecuencia.copy())
N_eval=2
df = pd.DataFrame(data={'predicciones':predicciones[N_eval],'comparaciones': comparaciones[N_eval],'frecuencias': frecuencias_N[N_eval]})
aciertos=pd.DataFrame()
aciertos['mean'] = df.groupby('comparaciones')['frecuencias'].mean()
aciertos['std'] = df.groupby('comparaciones')['frecuencias'].std()
aciertos.plot(kind='bar', y='mean', yerr='std', title = "Promedio y desviación estándar de las frecuencias de las palabras",color='grey', legend=False)
plt.xticks(ticks=[False, True], labels=['error', 'acierto'], rotation=0 )
plt.xlabel('Comparaciones')
plt.ylabel('Frecuencia')

#%%

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
axs[0].violinplot(np.array(df[df['comparaciones']==True]['frecuencias']),widths=0.1, showmeans=True, showextrema=True, showmedians=False)
axs[1].violinplot(np.array(df[df['comparaciones']==False]['frecuencias']),widths=0.8, showmeans=True, showextrema=True, showmedians=False)
axs[0].set_title('Aciertos')
axs[1].set_title('Errores')
axs[0].set_xticks([])
axs[1].set_xticks([])
axs[0].set_ylim([df['frecuencias'].min()-0.1*df['frecuencias'].max(), df['frecuencias'].max()*1.1])
axs[1].set_ylim([df['frecuencias'].min()-0.1*df['frecuencias'].max(), df['frecuencias'].max()*1.1])
axs[0].set_ylabel('Frecuencia')

