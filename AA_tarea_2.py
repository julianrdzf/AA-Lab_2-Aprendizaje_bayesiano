import pandas as pd
archivo = 'Datos/chat.txt'

#Se cargan el archivo en un data frame
df_data=pd.read_csv(archivo,sep='Sin separador', header=None)
df_data.rename(columns={0:'crudo'},inplace=True)

#Se separan las columnas fecha, autor y texto
df_data['fecha'] = df_data.crudo.str.split(r'[\[\]]', expand=True,regex=True)[1]
df_data['autor'] = df_data.crudo.str.split(r'\[*\] |: ', expand=True,regex=True)[1]
df_data['texto'] = df_data.crudo.str.split(r'\][^:]+: ', expand=True,regex=True)[1]
df_data.drop('crudo', inplace=True, axis=1)

#Se quitan las filas con adjuntos omitidos
df_data=df_data.drop(df_data[df_data['texto']=='‎imagen omitida'].index)
df_data=df_data.drop(df_data[df_data['texto']=='‎audio omitido'].index)
df_data=df_data.drop(df_data[df_data['texto']=='‎sticker omitido'].index)

#se pasa todo a minúsculas
df_data['texto limpio']=df_data.texto.str.lower()

#Se cambia cualquier cosa que no sea una palabra por un espacio
df_data['texto limpio']=df_data['texto limpio'].str.replace(r'[\W]+',' ', regex=True)

#Se eliminan los espacios al final de linea
df_data['texto limpio']=df_data['texto limpio'].str.replace(r'\s$','', regex=True)

#Se eliminan los espacios al comienzo de linea
df_data['texto limpio']=df_data['texto limpio'].str.replace(r'^\s','', regex=True)

#se genera una columna con la lista de palabras de cada fila
df_data['palabras']=df_data['texto limpio'].str.lower().str.split(r'[\W]+')


#%%
#Funciones para generar los diccionarios y agregar listas de palabras

#Suma una aparición a un diccionario PD
def sumar_aparicion(dic, pal_objetivo):
    if dic.get(pal_objetivo) is not None:        
        if dic[pal_objetivo].get('_apariciones') is not None:
            dic[pal_objetivo]['_apariciones']+=1
        else:
            dic[pal_objetivo]['_apariciones']=1
    else:
        dic[pal_objetivo]={}
        dic[pal_objetivo]['_apariciones']=1

#Agrega una palabra a un diccionario PD
def agregar_palabra(dic, pal_objetivo, pal_agregar):
    if dic.get(pal_objetivo) is not None:
        if dic[pal_objetivo].get(pal_agregar) is not None:
            dic[pal_objetivo][pal_agregar]+=1            
        else:
            dic[pal_objetivo][pal_agregar]=1
            
        if dic[pal_objetivo].get('_total') is not None:    
            dic[pal_objetivo]['_total']+=1
        else:
            dic[pal_objetivo]['_total']=1
    else:
        dic[pal_objetivo]={}
        dic[pal_objetivo][pal_agregar]=1
        dic[pal_objetivo]['_total']=1

#Agrega una lista de palabras a un diccionario PD
def agregar_palabras_PD(dic, lista, N):
        
    for i in range(len(lista)):
        agregadas = []
        for n in range(1,N+1):            
            if (i-n>=0):
                #Verifico que no sean dos palabras consecutivas iguales
                if not(lista[i-n] in agregadas):
                    agregar_palabra(dic,lista[i],lista[i-n])
                    agregadas.append(lista[i-n])
        sumar_aparicion(dic, lista[i])

#Agrega un lista de palabras a un diccionario P    
def agregar_palabras_P(dic,lista):
    for i in range(len(lista)):
        if dic.get(lista[i]) is not None:        
            dic[lista[i]]+=1
        else:
            dic[lista[i]]=1
        dic['_total']+=1

#Genera un diccionario P a partir de un Data Frame que contiene un columna con listar de palabras     
def entrenar_P(df):
    P={}
    P['_total']=0
    for lista in df:
        agregar_palabras_P(P, lista)
    return P
  
#Genera un diccionario PD teniendo en cuenta N a partir de un Data Frame que contiene un columna con listar de palabras     
def entrenar_PD(df, N):    
    PD={}    
    for lista in df:
        agregar_palabras_PD(PD, lista, N)
    return PD

#Convierte una lista a minúsculas
def minusculas(lista):
    convertir = []
    for i in range(len(lista)):
        convertir.append(lista[i].lower())
    return convertir

#Genera un diccionario PD entrenado hacia adelante (-no se usa, solo para pruebas-)
def entrenar_PD_adelante(df, N):           
    PD={}
    
    for lista in df:
        for i in range(len(lista)):
            for n in range(1,N+1):
                if (i+n<len(lista)):
                    agregar_palabra(PD,lista[i],lista[i+n])
            sumar_aparicion(PD, lista[i])    
    return PD

#%%
#Generación de diccionarios
N=3
P = entrenar_P(df_data['palabras'])
PD = entrenar_PD(df_data['palabras'], N)
P_nada = 0.001

entrenar_online = True

#%%
####Cliente#####
def recomendacion_bayesiana(frase):
    
    D=minusculas(frase)    
    
    Horizonte = N
    h_MAP = ""
    p_MAP = 0
    
    for h in P:
        
        if h != '_total':
            prob = P[h]/P['_total']
            for d in D[-Horizonte:]:
                prob = prob * PD[h].get(d, P_nada)/PD[h]['_apariciones']
                
            if prob > p_MAP:
                h_MAP , p_MAP = h , prob
    #print(h_MAP)

    return h_MAP

##### LOOP PRINCIPAL #####

print("Ingrese la frase dando ENTER luego de \x1b[3mcada palabra\x1b[0m.")
print("Ingrese sólo ENTER para aceptar la recomendación sugerida, o escriba la siguiente palabra y de ENTER")
print("Ingrese '.' para comenzar con una frase nueva.")
print("Ingrese '..' para terminar el proceso.")

frase = []
palabra_sugerida = ""
while 1:
    palabra = input(">> ")

    if palabra == "..":
        break

    elif palabra == ".":
        if entrenar_online:
            frase = minusculas(frase)
            agregar_palabras_P(P, frase)
            agregar_palabras_PD(PD, frase, N)
            
        print("----- Comenzando frase nueva -----")
        frase = []

    elif palabra == "": # acepta última palabra sugerida
        frase.append(palabra_sugerida)

    else: # escribió una palabra
        frase.append(palabra)

    if frase:
        palabra_sugerida = recomendacion_bayesiana(frase)
    
        frase_propuesta = frase.copy()
        frase_propuesta.append("\x1b[3m"+ palabra_sugerida +"\x1b[0m")
    
        print(" ".join(frase_propuesta))

     