import pandas as pd
from bayes import BayesPredictor

from preprocess import load_data

FILENAME = 'Datos/chat_big.txt'



if __name__=="__main__":
    data=load_data(FILENAME)
    print(data.head)
    predictor=BayesPredictor(data["palabras"],4)
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
            predictor.update(frase)
                
            print("----- Comenzando frase nueva -----")
            frase = []
    
        elif palabra == "": # acepta última palabra sugerida
            frase.append(palabra_sugerida)
    
        else: # escribió una palabra
            frase.append(palabra)
    
        if frase:
            palabra_sugerida = predictor.predict(frase,verbose=True)[0]
        
            frase_propuesta = frase.copy()
            frase_propuesta.append("\x1b[3m"+ palabra_sugerida +"\x1b[0m")
        
            print(" ".join(frase_propuesta))

     