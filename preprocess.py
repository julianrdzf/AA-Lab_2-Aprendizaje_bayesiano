

import pandas as pd
archivo = 'Datos/chat.txt'

#Se cargan el archivo en un data frame

import pandas as pd


def load_wpp_data(filename:str)->pd.DataFrame:
    #Se carga el archivo en un dataframe
    with open(filename, encoding="utf8") as f:
        #saltear las primeras 3 lineas
        for i in range(3):
            f.readline()

        lines = f.readlines()
    df_data = pd.DataFrame()    
    df_data['crudo']=pd.DataFrame(lines)
    is_ios=df_data.crudo.iloc[0][0]=='['
    print("Es IOS?: ",is_ios)
    if is_ios:
        df_data['fecha'] = df_data.crudo.str.split(r'[\[\]]', expand=True,regex=True)[1]
        df_data['autor'] = df_data.crudo.str.split(r'\[*\] |: ', expand=True,regex=True)[1]
        df_data['texto'] = df_data.crudo.str.split(r'\][^:]+: ', expand=True,regex=True)[1]
    else:
    
        df_data['fecha'] = df_data['crudo'].str.extract(r'(\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}) -')[0]
        df_data['autor'] = df_data['crudo'].str.extract(r' - ([^:]+):')[0]
        df_data['texto'] = df_data['crudo'].str.extract(r': (.+)')[0]
    df_clean=clean(df_data)
    return df_clean
    
def clean(df_data:pd.DataFrame)->pd.DataFrame:
    #df_data.drop('crudo', inplace=True, axis=1)
    
    #Se quitan las filas con adjuntos omitidos
    df_data=df_data.drop(df_data[df_data['texto']=='‎imagen omitida\n'].index)
    df_data=df_data.drop(df_data[df_data['texto']=='‎audio omitido\n'].index)
    df_data=df_data.drop(df_data[df_data['texto']=='‎sticker omitido\n'].index)
    df_data=df_data.drop(df_data[df_data['texto']=='<Multimedia omitido>'].index)
    df_data=df_data.dropna()
    #se pasa todo a minúsculas
    df_data['texto limpio']=df_data.texto.str.lower()
    
    #Se cambia cualquier cosa que no sea una palabra por un espacio
    df_data['texto limpio']=df_data['texto limpio'].str.replace(r'[\W]+',' ', regex=True)
    #Se cambia cambian numeros por un espacio
    df_data['texto limpio']=df_data['texto limpio'].str.replace(r'[\d+]+',' ', regex=True)
    
    #Se eliminan los espacios al final de linea
    df_data['texto limpio']=df_data['texto limpio'].str.replace(r'\s$','', regex=True)
    
    #Se eliminan los espacios al comienzo de linea
    df_data['texto limpio']=df_data['texto limpio'].str.replace(r'^\s','', regex=True)
    
    #se genera una columna con la lista de palabras de cada fila
    df_data['palabras']=df_data['texto limpio'].str.lower().str.split(r'[\W]+')
    return df_data


#Convierte una lista a minúsculas
def minusculas(lista):
    convertir = []
    for i in range(len(lista)):
        convertir.append(lista[i].lower())
    return convertir