from collections import defaultdict
import time
from preprocess import minusculas,load_data
import pandas as pd














class BayesPredictor():
    def __init__(self, ejemplos,horizonte,m=2, palabras_validas):
        self.ejemplos=ejemplos
        
        self.horizonte=horizonte
        self.m=m
        self.posteriori=defaultdict(dict)
        self.priori={}
        self.estimador=defaultdict(dict)
        self.__train(ejemplos, palabras_validas)
   



    def predict(self,phrase,verbose=False):
        subphrase=phrase[-self.horizonte:]
        print(subphrase)
        candidates=set()
        candidates_scores={}
        for word in subphrase:

            if not word in self.vocab():
                continue
            for new_word in self.estimador[word].keys():
                if new_word in ['_default']:
                    continue
                candidates.add(new_word)
        print(candidates)
        for new_word in candidates:
            prob_word=1
            for word in subphrase:
                if not word in self.vocab():
                    continue
                prob=self.estimador[word].get(new_word,self.estimador[word]['_default'])
                prob_word*=prob
            candidates_scores[new_word]=prob_word

        candidates_scores=sorted(candidates_scores.items(),key=lambda x:x[1],reverse=True)
        
        #todo falta ver que hacer si no hay candidatos, en ese caso todos deberian tener la misma probabilidad
        #asi que capaz lo mejor es elegir la palabra mas probable y punto
        if verbose:
            print(candidates_scores[0:5])
        
        return candidates_scores[0]     
    
    def vocab(self):
        vocab= list(self.priori.keys())
        #remove _total
        vocab.remove('_total')
        return vocab
     


    
#region ENTRENAMIENTO
    def update(self,frase):
        frase=minusculas(frase)
        self.__train([frase])
    def __train(self,ejemplos, palabras_validas):
        self.__entrenar_prori(ejemplos, palabras_validas)
        self.__entrenar_posteriori(ejemplos, palabras_validas)   
        self.__entrenar_estimador()   


    #Genera un diccionario P a partir de un Data Frame que contiene un columna con listar de palabras     
    def __entrenar_prori(self,lista_frases, palabras_validas):
    
        self.priori['_total']=self.priori.get('_total',0)
        for frase in lista_frases:
            self.__agregar_palabras_priori(frase, palabras_validas)
    


    #Agrega un lista de palabras a un diccionario P    
    def __agregar_palabras_priori(self,frase, palabras_validas):
        try:
            len(frase)
        except Exception as e:
            print("Error: ",frase)
        for i in range(len(frase)):
            if (frase[i] in palabras_validas) or (len(palabras_validas)==0):
                if self.priori.get(frase[i]) is not None:        
                    self.priori[frase[i]]+=1
                else:
                    self.priori[frase[i]]=1
                self.priori['_total']+=1
    #Genera un diccionario PD teniendo en cuenta N a partir de un Data Frame que contiene una columna con lista de palabras     
    def __entrenar_posteriori(self,lista_frases, palabras_validas):    
  
        for frase in lista_frases:
            self.__agregar_palabras_posteriori(frase, palabras_validas)





    #Agrega una lista de palabras a un diccionario PD
    def __agregar_palabras_posteriori(self,lista, palabras_validas):
            
        for i in range(0,len(lista)):
            agregadas = []
            for n in range(1,self.horizonte+1):            
                if (i+n<len(lista)):
                    #Verifico que no sean dos palabras consecutivas iguales
                    # if not(lista[i-n] in agregadas):
                        pal_nueva=lista[i+n]
                        pal_horizonte=lista[i]
                        
                        if ((pal_nueva in palabras_validas) and (pal_horizonte in palabras_validas)) or (len(palabras_validas)==0):
                            self.posteriori[pal_horizonte][pal_nueva]=self.posteriori[pal_horizonte].get(pal_nueva,0)+1     
                            self.posteriori[pal_horizonte]['_total']=self.posteriori[pal_horizonte].get('_total',0)+1
                       

            #self.__sumar_aparicion_posteriori(lista[i])




    # # #Suma una aparición a un diccionario PD
    # # def __sumar_aparicion_posteriori(self, pal_objetivo):
    # #     if self.posteriori.get(pal_objetivo) is not None:        
    # #         if self.posteriori[pal_objetivo].get('_apariciones') is not None:
    # #             self.posteriori[pal_objetivo]['_apariciones']+=1
    # #         else:
    # #             self.posteriori[pal_objetivo]['_apariciones']=1
    # #     else:
    # #         self.posteriori[pal_objetivo]={}
    # #         self.posteriori[pal_objetivo]['_apariciones']=1

    def __entrenar_estimador(self):
        for word in self.vocab():
            for post_word in self.posteriori[word].keys():
                if post_word in ['_total']:
                    continue
                estimador=self.__compute_estimator(word,post_word)
                self.estimador[word][post_word]=estimador
            self.estimador[word]['_default']=self.__compute_estimator(word,"-1")
    def __compute_estimator(self,word,post_word):
        #se busca computar el m-estimador de la probabilidad de la palabra word dada prev_word
        #eso es, dada la frecuencia de una palabra dada la prev word, se pondera segun la probabilidad
        #de esa palabra y un parametro m

        n=self.priori[word]
        e=self.posteriori.get(word,{}).get(post_word,0)/n
        p=1/len(self.vocab())
        m_estimador=(e+self.m*p)/(self.m+n)

        return m_estimador
#endregion
   

                


if __name__== "__main__":
    filename = 'Datos/chat.txt'
    data=load_data(filename)
    print(data.head)
    predictor=BayesPredictor(data["palabras"],4)
    print(predictor.posteriori['vamo'],predictor.estimador['vamo'])
    print(predictor.predict(['vamo']))

    





    






