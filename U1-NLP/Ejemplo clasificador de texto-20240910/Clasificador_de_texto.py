# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.model_selection import train_test_split

# Lista de archivos de texto a procesar (poemas de Benedetti y Neruda)
archivos = [
  'Benedetti.txt',
  'Neruda.txt',
]
print(archivos)  # Imprimimos la lista de archivos

# Inicializamos listas para almacenar los textos y sus respectivas etiquetas
textos = []
etiquetas = []

# Iteramos sobre los archivos, asignando una etiqueta a cada uno (0 para Benedetti, 1 para Neruda)
for etiqueta, f in enumerate(archivos):
    print(f"{f} Corresponde a {etiqueta}")  # Mostramos el archivo y su etiqueta
    with open(f, 'r', encoding='utf-8') as archivo:
        # Leemos cada línea del archivo
        for line in archivo:
            print(line)  # Imprimimos la línea original
            line = line.rstrip().lower()  # Eliminamos espacios al final y convertimos a minúsculas
            print(line)
            if line:
                # Eliminamos la puntuación de la línea
                line = line.translate(str.maketrans('', '', string.punctuation))
                textos.append(line)  # Agregamos la línea procesada a la lista de textos
                etiquetas.append(etiqueta)  # Asignamos la etiqueta correspondiente
            print(line)

# Dividimos los textos y las etiquetas en conjuntos de entrenamiento (90%) y prueba (10%)
train_text, test_text, Ytrain, Ytest = train_test_split(textos, etiquetas, test_size=0.1, random_state=42)

# Mostramos la cantidad de textos de entrenamiento y prueba
len(Ytrain)
len(Ytest)

"""
"<unk>" es una convención que se utiliza a menudo en el procesamiento de lenguaje natural
para representar palabras desconocidas o fuera del vocabulario. Aquí se asigna el índice 0 a esta palabra especial.
"""
indice = 1
indicepalabras = {'<unk>': 0}  # Diccionario para asignar un índice único a cada palabra

# Asignamos un índice a cada palabra en los textos de entrenamiento
for texto in train_text:
    tokens = texto.split()  # Dividimos cada línea en palabras (tokens)
    for token in tokens:
        if token not in indicepalabras:
            indicepalabras[token] = indice  # Asignamos un nuevo índice a palabras no vistas antes
            indice += 1

# Inicializamos listas para almacenar los textos convertidos a índices (enteros)
train_text_int = []
test_text_int = []

# Convertimos los textos de entrenamiento a sus representaciones de enteros
for texto in train_text:
    tokens = texto.split()
    linea_entero = [indicepalabras[token] for token in tokens]  # Convertimos cada palabra a su índice
    train_text_int.append(linea_entero)

# Convertimos los textos de prueba a sus representaciones de enteros
for texto in test_text:
    tokens = texto.split()
    line_as_int = [indicepalabras.get(token, 0) for token in tokens]  # Si una palabra no existe, se usa el índice 0
    test_text_int.append(line_as_int)

# Inicializamos las matrices de transición (A0, A1) y las distribuciones iniciales (pi0, pi1)
V = len(indicepalabras)  # El tamaño del vocabulario
A0 = np.ones((V, V))  # Matriz de transición para la clase 0 (Benedetti)
pi0 = np.ones(V)      # Distribución inicial para la clase 0

A1 = np.ones((V, V))  # Matriz de transición para la clase 1 (Neruda)
pi1 = np.ones(V)      # Distribución inicial para la clase 1

# Función para actualizar las matrices de transición y distribuciones iniciales
def compute_counts(text_as_int, A, pi):
    for tokens in text_as_int:
        last_idx = None
        for idx in tokens:
            if last_idx is None:
                pi[idx] += 1  # Incrementamos la probabilidad inicial de la palabra
            else:
                A[last_idx, idx] += 1  # Incrementamos la transición de una palabra a otra
            last_idx = idx

# Calculamos las probabilidades para la clase 0 (Benedetti)
compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 0], A0, pi0)

# Calculamos las probabilidades para la clase 1 (Neruda)
compute_counts([t for t, y in zip(train_text_int, Ytrain) if y == 1], A1, pi1)

# Normalizamos las matrices de transición y las distribuciones iniciales
A0 /= A0.sum(axis=1, keepdims=True)
pi0 /= pi0.sum()

A1 /= A1.sum(axis=1, keepdims=True)
pi1 /= pi1.sum()

# Calculamos los logaritmos de las probabilidades (para evitar problemas numéricos)
logA0 = np.log(A0)
logpi0 = np.log(pi0)

logA1 = np.log(A1)
logpi1 = np.log(pi1)

# Calculamos la proporción de textos de cada clase en el conjunto de entrenamiento
count0 = sum(y == 0 for y in Ytrain)   
count1 = sum(y == 1 for y in Ytrain)   
total = len(Ytrain)   
p0 = count0 / total   # Proporción de la clase 0
p1 = count1 / total   # Proporción de la clase 1
logp0 = np.log(p0)    # Logaritmo de la proporción de la clase 0
logp1 = np.log(p1)    # Logaritmo de la proporción de la clase 1

# Definimos una clase para el clasificador basado en el modelo de cadenas de Markov
class Classifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs  # Matrices de transición (en logaritmo)
        self.logpis = logpis  # Distribuciones iniciales (en logaritmo)
        self.logpriors = logpriors  # Prioris de clase (en logaritmo)
        self.K = len(logpriors)  # Número de clases
        
    def _compute_log_likelihood(self, input_, class_):
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
            last_idx = idx
        
        return logprob     
    
    # Método para predecir la clase de nuevas secuencias
    def predict(self, inputs):
        predictions = np.zeros(len(inputs))
        for i, input_ in enumerate(inputs):
            # Calcula las probabilidades posteriores para cada clase
            posteriors = [self._compute_log_likelihood(input_, c) + self.logpriors[c] \
                          for c in range(self.K)]
            # Selecciona la clase con la mayor probabilidad
            pred = np.argmax(posteriors)
            predictions[i] = pred
        return predictions

# Creamos una instancia del clasificador con los parámetros calculados
clf = Classifier([logA0, logA1], [logpi0, logpi1], [logp0, logp1])

# Predecimos y calculamos la precisión en el conjunto de entrenamiento
Ptrain = clf.predict(train_text_int)
print(f"Train acc: {np.mean(Ptrain == Ytrain)}")

# Predecimos y calculamos la precisión en el conjunto de prueba
Ptest = clf.predict(test_text_int)
print(f"Test acc: {np.mean(Ptest == Ytest)}")

# Nuevo texto para predecir la clase (autor)
texto = "Cuerpo de piel, de musgo, de leche ávida y firme."
tokens = texto.split()  # Separamos el texto en palabras
input_ = [indicepalabras.get(token, 0) for token in tokens]  # Convertimos cada palabra a su índice
pred = clf.predict([input_])[0]  # Predecimos la clase del texto
print(f"El texto pertenece a la clase {pred}")  # Imprimimos la clase predicha