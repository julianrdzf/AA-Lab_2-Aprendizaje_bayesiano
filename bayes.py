from collections import defaultdict
import time
from preprocess import minusculas,load_wpp_data
import pandas as pd
import numpy as np



class BayesPredictor():
    def __init__(self, ejemplos,horizonte,m=2, palabras_validas=set()):
        self.ejemplos=ejemplos
        
        self.horizonte=horizonte
        self.m=m
        self.posteriori=defaultdict(dict)
        self.priori={}
        self.estimador=defaultdict(dict)
        self.dic_candidatos=defaultdict(dict)
        self.palabras_validas= palabras_validas
        self.__train(ejemplos)
        
   


    def vocab(self):
        vocab= list(self.priori.keys())
        #remove _total
        vocab.remove('_total')
        return vocab
    
    def predict(self,phrase,verbose=False):
        subphrase=phrase[-self.horizonte:]
        #print(subphrase)
        prob_max=-np.inf
        palabra=""


        for new_word in self.vocab():
            prob_word=np.log(self.priori[new_word]/self.priori['_total'])
            for word in subphrase:
 
                    
                prob=np.log(self.estimador[new_word].get(word,self.estimador[new_word]['_default']))
                prob_word+=prob
            if prob_word>prob_max:
                prob_max=prob_word
                palabra=new_word

        if verbose:
            print ({palabra:prob_max})
        
        return palabra     
    
    
     


    
#region ENTRENAMIENTO
    def update(self,frase,solo_cambios=False):
        frase=minusculas(frase)
        self.__train([frase],solo_cambios)
    def __train(self,ejemplos,solo_cambios=False):
        palabras_agregadas=self.__entrenar_prori(ejemplos)
        self.__entrenar_posteriori(ejemplos)
        if solo_cambios: 
            print("entrenando solo cambios, agregando palabras: ",palabras_agregadas)
            self.__entrenar_estimador(palabras_agregadas)  
        else:
            print("entrenando todo")
            self.__entrenar_estimador() 


    #Genera un diccionario P a partir de un Data Frame que contiene un columna con listar de palabras     
    def __entrenar_prori(self,lista_frases):
    
        self.priori['_total']=self.priori.get('_total',0)
        todas_palabras_agregadas=set()
        for frase in lista_frases:
            palabras_agregadas=self.__agregar_palabras_priori(frase)
            todas_palabras_agregadas=todas_palabras_agregadas.union(palabras_agregadas)
        return todas_palabras_agregadas
    


    #Agrega un lista de palabras a un diccionario P    
    def __agregar_palabras_priori(self,frase):
        palabras_agregadas=set()
        try:
            len(frase)
        except Exception as e:
            print("Error: ",frase)
        for i in range(len(frase)):
            if (frase[i] in self.palabras_validas) or (len(self.palabras_validas)==0):
                if self.priori.get(frase[i]) is not None:        
                    self.priori[frase[i]]+=1
                else:
                    self.priori[frase[i]]=1
                self.priori['_total']+=1
                palabras_agregadas.add(frase[i])
        return palabras_agregadas
            
    #Genera un diccionario PD teniendo en cuenta N a partir de un Data Frame que contiene una columna con lista de palabras     
    def __entrenar_posteriori(self,lista_frases):    
  
        for frase in lista_frases:
            self.__agregar_palabras_posteriori(frase)





    #Agrega una lista de palabras a un diccionario PD
    def __agregar_palabras_posteriori(self,lista):
            
        for i in range(0,len(lista)):
            agregadas = []
            for n in range(1,self.horizonte+1):            
                if (i-n>0):
                    #Verifico que no sean dos palabras consecutivas iguales
                    # if not(lista[i-n] in agregadas):
                    pal_horizonte=lista[i-n]
                    pal_fija=lista[i]
                    if ((pal_horizonte in self.palabras_validas) and (pal_fija in self.palabras_validas)) or (len(self.palabras_validas)==0):  
                        self.posteriori[pal_fija][pal_horizonte]=self.posteriori[pal_fija].get(pal_horizonte,0)+1     
                        self.posteriori[pal_fija]['_total']=self.posteriori[pal_fija].get('_total',0)+1
                        self.dic_candidatos[pal_horizonte][pal_fija]=1
                     
                    
                    

           

    def __entrenar_estimador(self,palabras_a_entrenar=None):
        if palabras_a_entrenar is None:
            palabras_a_entrenar=self.vocab()
        for word in palabras_a_entrenar:
            for horizonte_word in self.posteriori[word].keys():
                if horizonte_word in ['_total']:
                    continue
                estimador=self.__compute_estimator(word,horizonte_word)
                self.estimador[word][horizonte_word]=estimador
            self.estimador[word]['_default']=self.__compute_estimator(word,"-1")
    def __compute_estimator(self,word,horizonte_word):
        #se busca computar el m-estimador de la probabilidad de la palabra word dada prev_word
        #eso es, dada la frecuencia de una palabra dada la prev word, se pondera segun la probabilidad
        #de esa palabra y un parametro m

        n=self.priori[word]
        e=self.posteriori.get(word,{}).get(horizonte_word,0)
        p=1/len(self.vocab())

        m_estimador=(e+self.m*p)/(self.m+n)

        return m_estimador
    

#endregion
   

                


if __name__== "__main__":
    filename = 'Datos/chat.txt'
    data=load_wpp_data(filename)
    print(data.head)
    
    palabras_validas=set()    
    with open('Datos/es.txt', 'r', encoding='utf-8') as archivo:
        for linea in archivo:
            palabra = linea.strip()  
            palabras_validas.add(palabra)
    
    predictor=BayesPredictor(data["palabras"],4, palabras_validas=palabras_validas)
    print(predictor.predict(['vamo']))



    






