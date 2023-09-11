import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.regularizers import l2

# Guardamos los datos en los dataframes.
df_train_pre = pd.read_csv('train.csv')
df_train_pre = df_train_pre.dropna()

test_df = pd.read_csv('test.csv')

df_train_pre = pd.read_csv('train.csv')
df_train_pre = df_train_pre.dropna()

test_df = pd.read_csv('test.csv')

# Iniciamos la clase para nuestro de modelo de NLP
class nlp:
    '''
        En esta clase se crea un modelo para predecir si las reseñas de nuestro dataset son negativas o positivas.
    '''
    def __init__(self, train, test, columns, label_encoder = LabelEncoder()):
        self.text = columns[0]
        self.clase = columns[1]
        
        # Dividimos el dataset de training, en validation y training
        train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)
 
        # Almacenamos en distintos arrys lo datos para training, validation y test, y a sus labels hacemos un encoder.
        self.train_data = train_df[self.text]
        self.train_l = label_encoder.fit_transform(train_df[self.clase])

        self.val_data = val_df[self.text]
        self.val_l = label_encoder.fit_transform(val_df[self.clase])
        
        self.test_data = test[self.text]
        self.test_l = label_encoder.fit_transform(test[self.clase])
    
    def token(self, vocab_size = 10000, embedding_dim = 16, max_length = 100, trunc_type='post', padding_type='post', 
              oov_tok = "<OOV>",training_size = 20000):
        # Iniciamos el objeto Tokenizer con los parámetros por default que tiene el método de nuestra clase.
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        self.tokenizer.fit_on_texts(self.train_data)

        # Almacenamos nuestro índice de palabras en un diccionario.
        self.word_index = self.tokenizer.word_index

        # Pasamos a secuencias todos las oraciones que estan en nuestro training data.
        training_sequences = self.tokenizer.texts_to_sequences(self.train_data)
        # Aplicamos un padding a todas las secuencias que ya se interpretaron con nuestro índice, esto con el fin de que todas secuencias tengan la misma longitud.
        self.training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        
        val_sequences = self.tokenizer.texts_to_sequences(self.val_data)
        self.val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        
        test_sequences = self.tokenizer.texts_to_sequences(self.test_data)
        self.test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    def model(self, vocab_size = 10000, embedding_dim = 16, max_length = 100, trunc_type='post', padding_type='post', 
              oov_tok = "<OOV>",training_size = 20000):
        self.model = tf.keras.Sequential([
            # Hacemos embedding de nuestras secuencias.
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
            GlobalAveragePooling1D(),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # Capa densa con 64 unidades
            Dropout(0.5),  # Dropout para regularización
            Dense(1, activation='sigmoid')  # Capa de salida con activación sigmoide para clasificación binaria
        ])
        
        # Utilizamos la función de perdidoa binary_crossentropy porque solo tenemos dos clases de salida.
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, num_epochs = 10):
        self.history = self.model.fit(self.training_padded, self.train_l, epochs=num_epochs, validation_data=(self.val_padded, self.val_l), verbose=1)

    def plot_graphs(self, string):
        # Graficamos los parametros que estan dentro de nuestro historial de entrenamiento,
        plt.plot(self.history.history[string])
        plt.plot(self.history.history['val_'+string])
        plt.title(string)
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()

    def test(self):
        # Almacenamos en un array todas las salidas de nuestra red neuronal cuando pasamos nuestro dataset para test.
        predicciones = self.model.predict(self.test_padded)

        # Hacemos un umbral para que las predicciones solo esten entre buenas y malas.
        predicciones_binarias = (predicciones > 0.5).astype(int)
        predicciones_binarias = [i[0] for i in predicciones_binarias]
        
        # Obtenemos el acurracy de nuestras predicciones.
        acurracy = (predicciones_binarias == self.test_l).sum()/len(self.test_l)
        print('\nAcurracy: ',acurracy)

        # Analizamos algunas muestras en específico de nuestr dataset de test, esto al azar.
        num = np.random.randint(0, len(self.test_data) + 1)
        print("\nMuestra: ",self.test_data[num])
        print("\nSegún el modelo la reseña es MALA") if predicciones_binarias[num] == 0 else print("\nSegún el modelo la reseña es BUENA")
        print("\nSegún el dataset la reseña es MALA ") if self.test_l[num] == 0 else print("\nSegún el dataset la reseña es BUENA ")

        
columns = df_train_pre.columns

model = nlp(df_train_pre, test_df, columns)
model.token()
model.model()
model.fit()
model.plot_graphs('loss')
model.plot_graphs('accuracy')
model.test()