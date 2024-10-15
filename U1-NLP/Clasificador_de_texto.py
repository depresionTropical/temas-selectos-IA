<<<<<<<<<<<<<<  ✨ Codeium Command 🌟  >>>>>>>>>>>>>>>>

import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split
archivos = [
  'Benedetti.txt',
  'Neruda.txt',
]
print (archivos)
textos = []
etiquetas = []

for etiqueta, f in enumerate(archivos):
    print(f"{f} Corresponde a {etiqueta}")    
    with open(f, 'r', encoding='utf-8') as archivo:
        for line in archivo:
            print(line)
            line = line.rstrip().lower()
            print(line)
            if line:
                # eliminar puntuación
                line = line.translate(str.maketrans('', '', string.punctuation))
                textos.append(line)
                etiquetas.append(etiqueta)
            print(line)
train_text, test_text, Ytrain, Ytest = train_test_split(textos, etiquetas, test_size=0.1, random_state=42)
len(Ytrain)
len(Ytest)
"""
Clase para clasificar texto según un modelo de Markov de orden 1.
"<unk>" es una convención que se utiliza a menudo en el procesamiento de lenguaje natural para representar palabras 
desconocidas o fuera del vocabulario. En este caso, se está asignando el índice 0 a esta palabra especial.
"""
indice = 1
indicepalabras = {'<unk>': 0}
for texto in train_text:
    tokens = texto.split()
    for token in tokens:
        if token not in indicepalabras:
            indicepalabras[token] = indice
            indice += 1

train_text_int = []

test_text_int = []

for texto in train_text:
    tokens = texto.split()
    linea_entero = [indicepalabras[token] for token in tokens]
    train_text_int.append(linea_entero)

for texto in test_text:
    tokens = texto.split()
    line_as_int = [indicepalabras.get(token, 0) for token in tokens]
    test_text_int.append(line_as_int)
    
V = len(indicepalabras)
A0 = np.ones((V, V))
pi0 = np.ones(V)

A1 = np.ones((V, V))
pi1 = np.ones(V)

def compute_counts(text_as_int, A, pi):
    """
    Contabiliza las ocurrencias de cada palabra en el texto,
    y las transiciones entre palabras.
    
    Parameters
    ----------
    text_as_int : list of list of int
        Texto como lista de listas de enteros, en el que cada entero
        representa un token.
    A : np.array of int
        Matriz de transición, donde A[i, j] es el número de veces que
        la palabra i es seguida por la palabra j.
    pi : np.array of int
        Vector de probabilidad a priori, donde pi[i] es el número de
        veces que la palabra i es la primera palabra en una secuencia.
    """
    for tokens in text_as_int:
        # Variable para recordar el índice de la palabra anterior
        last_idx = None
        for idx in tokens:
            # estamos en la primera palabra de la secuencia
            if last_idx is None:
                # Aumentar la cuenta de la palabra actual en el vector pi
                pi[idx] += 1
            else:
                # Aumentar la cuenta de la transición de la palabra anterior a
                # la actual en la matriz de transición A
                A[last_idx, idx] += 1
            # Actualizar last_idx para la próxima iteración
            last_idx = idx    
            
compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 0], A0, pi0)
compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 1], A1, pi1)

A0 /= A0.sum(axis=1, keepdims=True)
pi0 /= pi0.sum()

A1 /= A1.sum(axis=1, keepdims=True)
pi1 /= pi1.sum()

logA0 = np.log(A0)
logpi0 = np.log(pi0)

logA1 = np.log(A1)
logpi1 = np.log(pi1)

count0 = sum(y == 0 for y in Ytrain)   
count1 = sum(y == 1 for y in Ytrain)   
total = len(Ytrain)   
p0 = count0 / total   
p1 = count1 / total   
logp0 = np.log(p0)   
logp1 = np.log(p1) 

class Classifier:
    """
    Constructor de la clase Classifier.
    
    Parameters
    ----------
    logAs : list of np.array
        Matrices de transición, donde logAs[i] es la matriz de transición
        para la clase i.
    logpis : list of np.array
        Vectores de probabilidad a priori, donde logpis[i] es el vector de
        probabilidad a priori para la clase i.
    logpriors : np.array
        Vector de probabilidades a priori para cada clase.
    Clase para clasificar texto según un modelo de Markov de orden 1.
    """
    def __init__(self, logAs: list, logpis: list, logpriors: np.ndarray):
    def __init__(self, logAs, logpis, logpriors):
        """
        Constructor de la clase Classifier.
        
        Parameters
        ----------
        logAs : list of np.array
            Matrices de transición, donde logAs[i] es la matriz de transición
            para la clase i.
        logpis : list of np.array
            Vectores de probabilidad a priori, donde logpis[i] es el vector de
            probabilidad a priori para la clase i.
        logpriors : np.array
            Vector de probabilidades a priori para cada clase.
        """
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
        self.K = len(logpriors) # número de clases

    """
    Calcula el logaritmo de la probabilidad de una secuencia de tokens
    según un modelo de Markov de orden 1.
    
    Parameters
    ----------
    input_ : list of int
        Secuencia de tokens.
    class_ : int
        Clase para la que se quiere calcular la probabilidad.
    """
    def _compute_log_likelihood(self, input_: list, class_: int) -> float:
        
    def _compute_log_likelihood(self, input_, class_):
        """
        Calcula el logaritmo de la probabilidad de una secuencia de tokens
        según un modelo de Markov de orden 1.
        
        Parameters
        ----------
        input_ : list of int
            Secuencia de tokens.
        class_ : int
            Clase para la que se quiere calcular la probabilidad.
        """
        logA = self.logAs[class_]
        logpi = self.logpis[class_]

        last_idx = None
        logprob = 0
        for idx in input_:
            if last_idx is None:
                # Es el primer token en la secuencia
                logprob += logpi[idx]
            else:
                # Calcula la probabilidad de transición de la palabra anterior a la actual
                logprob += logA[last_idx, idx]
            
            # Actualiza last_idx para la próxima iteración
            last_idx = idx
        
        return logprob     
    
    """
    Predice la clase para una lista de secuencias de tokens.
    
    Parameters
    ----------
    inputs : list of list of int
        Lista de secuencias de tokens.
    """
    def predict(self, inputs: list) -> np.ndarray:
    def predict(self, inputs):
        """
        Predice la clase para una lista de secuencias de tokens.
        
        Parameters
        ----------
        inputs : list of list of int
            Lista de secuencias de tokens.
        """
        predictions = np.zeros(len(inputs))
        for i, input_ in enumerate(inputs):
            # Calcula los logaritmos de las probabilidades posteriores para cada clase
            posteriors = [self._compute_log_likelihood(input_, c) + self.logpriors[c] \
                          for c in range(self.K)]
            # Elige la clase con la mayor probabilidad posterior como la predicción
            pred = np.argmax(posteriors)
            predictions[i] = pred
        return predictions



clf = Classifier([logA0, logA1], [logpi0, logpi1], [logp0, logp1])

Ptrain = clf.predict(train_text_int)
print(f"Train acc: {np.mean(Ptrain == Ytrain)}")

Ptest = clf.predict(test_text_int)
print(f"Test acc: {np.mean(Ptest == Ytest)}")