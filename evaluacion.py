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

data = data[:1000]

train, test = train_test_split(data, test_size=0.15, random_state=45)
#print(data.head)

SPANISH_DICT_FILENAME='Datos/es.txt'
palabras_validas=set()   
with open(SPANISH_DICT_FILENAME, 'r', encoding='utf-8') as archivo:
    palabras_validas=set()
    for linea in archivo:
        palabra = linea.strip()
        palabras_validas.add(palabra) 

#Variar N
m_s=[2]
Ns=[1,2,3,4]
predicciones = []
resultados= []
comparaciones = []
tiempos=[]
for m in m_s:
    for N in Ns:
        HORIZONTE=N
        inicio = time.time()
        predictor=BayesPredictor(train["palabras"],HORIZONTE, m, palabras_validas=palabras_validas) 
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
plt.plot(Ns, np.array(resultados)/len(test_real)*100, '.-', ms=10, lw=1)
plt.xticks( Ns )
plt.xlabel('N')
plt.ylabel('Porcentaje de aciertos [%]')
plt.title('Aciertos en función del hiperparámetro N')
plt.show()

plt.figure()    
plt.plot(Ns, np.array(tiempos), '.-', ms=10, lw=1)
plt.xticks(Ns)
plt.xlabel('N')
plt.ylabel('Tiempo [s]')
plt.title('Tiempo de entrenamiento en función del hiperparámetro N')
plt.show()



#%%
frecuencias_N=[]
frecuencias_media = []
frecuencia=np.zeros(len(predicciones[0]))
for pred_N in predicciones:
    for idx, palabra in enumerate(pred_N):
        frecuencia[idx]=predictor.priori[palabra]
    frecuencias_N.append(frecuencia.copy())
    frecuencias_media.append(np.array(frecuencia).mean())
N_eval=1
df = pd.DataFrame(data={'predicciones':predicciones[N_eval-1],'comparaciones': comparaciones[N_eval-1],'frecuencias': frecuencias_N[N_eval-1]})
aciertos=pd.DataFrame()
aciertos['mean'] = df.groupby('comparaciones')['frecuencias'].mean()
aciertos['std'] = df.groupby('comparaciones')['frecuencias'].std()
aciertos.plot(kind='bar', y='mean', yerr='std', title = "Promedio y desviación estándar de las frecuencias de las palabras",color='grey', legend=False)
plt.xticks(ticks=[False, True], labels=['error', 'acierto'], rotation=0 )
plt.xlabel(f'Comparaciones N={N_eval}')
plt.ylabel('Frecuencia')

#%%
N_evals=[1,4]
for N_evals_1 in N_evals:
    df = pd.DataFrame(data={'predicciones':predicciones[N_evals_1-1],'comparaciones': comparaciones[N_evals_1-1],'frecuencias': frecuencias_N[N_evals_1-1]})
    
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
    fig.suptitle(f'Distribución de las frecuencias de las predicciones para N={N_evals_1}', fontsize=16)

#%%
plt.figure()    
plt.plot(Ns, np.array(frecuencias_media), '.-', ms=10, lw=1)
plt.xticks( Ns )
plt.xlabel('N')
plt.ylabel('Frecuencia media')

plt.title('Frecuencia media de predicciones en función del hiperparámetero N')
plt.show()

#%%
#Variar m
FILENAME = 'Datos/chat_big.txt'

data=load_wpp_data(FILENAME)

data = data[:1000]

train, test = train_test_split(data, test_size=0.15, random_state=45)
#print(data.head)

SPANISH_DICT_FILENAME='Datos/es.txt'
palabras_validas=set()   
with open(SPANISH_DICT_FILENAME, 'r', encoding='utf-8') as archivo:
    palabras_validas=set()
    for linea in archivo:
        palabra = linea.strip()
        palabras_validas.add(palabra) 


m_s=[1,40,64,200]
#Ns=[1,2,3,4]
Ns=[4]
predicciones = []
resultados= []
comparaciones = []
tiempos=[]
for m in m_s:
    for N in Ns:
        HORIZONTE=N
        inicio = time.time()
        predictor=BayesPredictor(train["palabras"],HORIZONTE, m, palabras_validas=palabras_validas) 
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
plt.plot(m_s, np.array(resultados)/len(test_real)*100, '.-', ms=10, lw=1)
plt.xticks( m_s )
plt.xlabel(f'N={Ns[0]}')
plt.ylabel('Porcentaje de aciertos [%]')
plt.title('Aciertos en función del hiperparámetro m')
plt.show()

plt.figure()    
plt.plot(m_s, np.array(tiempos), '.-', ms=10, lw=1)
plt.xticks(Ns)
plt.xlabel(f'N={Ns[0]}')
plt.ylabel('Tiempo [s]')
plt.title('Tiempo de entrenamiento en función del hiperparámetro m')
plt.show()
