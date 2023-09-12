from collections import defaultdict
import time
from preprocess import minusculas,load_data
import pandas as pd














class BayesPredictor():
    def __init__(self, ejemplos,horizonte,m=2):
        self.ejemplos=ejemplos
        
        self.horizonte=horizonte
        self.m=m
        self.posteriori={}
        self.priori={}
        self.estimador=defaultdict(dict)
        self.__train(ejemplos)
   



    def predict(self,phrase,verbose=False):
        subphrase=phrase[-self.horizonte:]
        candidates={}
        for word in self.vocab():
            prob_word=1
            for prev_word in subphrase:
            
                if not prev_word in self.vocab():
                    prob=1
                else:
                    prob=self.estimador[word].get(prev_word,self.estimador[prev_word]['_default'])
                prob_word*=prob
            candidates[word]=prob_word

        #sort words to get the most probable
        candidates=sorted(candidates.items(),key=lambda x:x[1],reverse=True)
        if verbose:
            print(candidates[0:5])
        return candidates[0]     
    
    def vocab(self):
        vocab= list(self.priori.keys())
        #remove _total
        vocab.remove('_total')
        return vocab
     


    
#region ENTRENAMIENTO
    def update(self,frase):
        frase=minusculas(frase)
        self.__train([frase])
    def __train(self,ejemplos):
        self.__entrenar_prori(ejemplos)
        self.__entrenar_posteriori(ejemplos)   
        self.__entrenar_estimador()   


    #Genera un diccionario P a partir de un Data Frame que contiene un columna con listar de palabras     
    def __entrenar_prori(self,lista_frases):
    
        self.priori['_total']=self.priori.get('_total',0)
        for frase in lista_frases:
            self.__agregar_palabras_priori(frase)
    


    #Agrega un lista de palabras a un diccionario P    
    def __agregar_palabras_priori(self,frase):
        try:
            len(frase)
        except Exception as e:
            print("Error: ",frase)
        for i in range(len(frase)):
            if self.priori.get(frase[i]) is not None:        
                self.priori[frase[i]]+=1
            else:
                self.priori[frase[i]]=1
            self.priori['_total']+=1
    #Genera un diccionario PD teniendo en cuenta N a partir de un Data Frame que contiene una columna con lista de palabras     
    def __entrenar_posteriori(self,lista_frases):    
  
        for frase in lista_frases:
            self.__agregar_palabras_posteriori(frase)





    #Agrega una lista de palabras a un diccionario PD
    def __agregar_palabras_posteriori(self,lista):
            
        for i in range(len(lista)):
            agregadas = []
            for n in range(1,self.horizonte+1):            
                if (i-n>=0):
                    #Verifico que no sean dos palabras consecutivas iguales
                    if not(lista[i-n] in agregadas):
                        pal_objetivo=lista[i]
                        pal_agregar=lista[i-n]
                        if self.posteriori.get(pal_objetivo) is not None:
                            if self.posteriori[pal_objetivo].get(pal_agregar) is not None:
                                self.posteriori[pal_objetivo][pal_agregar]+=1            
                            else:
                                self.posteriori[pal_objetivo][pal_agregar]=1
                                
                            if self.posteriori[pal_objetivo].get('_total') is not None:    
                                self.posteriori[pal_objetivo]['_total']+=1
                            else:
                                self.posteriori[pal_objetivo]['_total']=1
                        else:
                            self.posteriori[pal_objetivo]={}
                            self.posteriori[pal_objetivo][pal_agregar]=1
                            self.posteriori[pal_objetivo]['_total']=1
                        agregadas.append(lista[i-n])
            self.__sumar_aparicion_posteriori(lista[i])




    #Suma una aparición a un diccionario PD
    def __sumar_aparicion_posteriori(self, pal_objetivo):
        if self.posteriori.get(pal_objetivo) is not None:        
            if self.posteriori[pal_objetivo].get('_apariciones') is not None:
                self.posteriori[pal_objetivo]['_apariciones']+=1
            else:
                self.posteriori[pal_objetivo]['_apariciones']=1
        else:
            self.posteriori[pal_objetivo]={}
            self.posteriori[pal_objetivo]['_apariciones']=1

    def __entrenar_estimador(self):
        for word in self.vocab():
            for prev_word in self.posteriori[word].keys():
                if prev_word in ['_total','_apariciones']:
                    continue
                estimador=self.__compute_estimator(prev_word,word)
                if estimador==1:
                    continue
                self.estimador[word][prev_word]=estimador
            self.estimador[word]['_default']=self.__compute_estimator(word,"-1")
    def __compute_estimator(self,prev_word,word):
        #se busca computar el m-estimador de la probabilidad de la palabra word dada prev_word
        #eso es, dada la frecuencia de una palabra dada la prev word, se pondera segun la probabilidad
        #de esa palabra y un parametro m

        n=self.priori[prev_word]
        e=self.posteriori.get(word,{}).get(prev_word,0)/n
        p=1/len(self.vocab())
        m_estimador=(e+self.m*p)/(self.m+n)

        return m_estimador
#endregion
   

                


if __name__== "__main__":
    filename = 'Datos/chat.txt'
    data=load_data(filename)
    print(data.head)
    predictor=BayesPredictor(data["palabras"],4)
    print(predictor.predict(['hola']))

    





    






