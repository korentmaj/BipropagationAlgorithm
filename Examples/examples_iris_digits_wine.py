# -*- coding: utf-8 -*-
"""Bipropagation_general.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UeI12kja8JkvnwUIECMF3Vc_NSzB8ZRV
"""

from sklearn.datasets import load_iris, load_digits, load_wine
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

def choose_dataset():
  print("Izberi nabor podatkov za razvrščanje:")
  print("1. Iris Dataset")
  print("2. Digits Dataset")
  print("3. Wine Dataset")
  choice = input("Vnesi številko izbranega nabora podatkov (1, 2 ali 3): ")

  if choice == '1':
      dataset = load_iris()
  elif choice == '2':
      dataset = load_digits()
  elif choice == '3':
      dataset = load_wine()
  else:
      print("Napačna izbira. Poskusi znova.")
      return choose_dataset()

  return dataset

def data_analysis(data_set):
  # Vhodni podatki (značilke)
  X = data_set.data


  # Želeni izhodni podatki (razred)
  y = data_set.target

  # Izpiši informacije o datasetu
  # print(selected_dataset.DESCR)

  #print(X[30:33,:])
  (rows,columns)=X.shape
  classes=max(y)

  print("Lastnosti izbranega nabora podatkov:")
  print("število vrstic je ",rows)
  print("število stolpcev je ",columns)
  print("število razredov je",classes+1)
  ##################
  # Loči podatke po razredih
  y_classes = np.unique(y)

  y_data = {}
  z_data = {}
  y_desired = {}

  for cls in y_classes:
      # Izberi podatke, ki pripadajo trenutnemu razredu
      y_data[cls] = X[y == cls]

      # Izberi podatke, ki ne pripadajo trenutnemu razredu
      z_data[cls] = X[y != cls]

  # Izračunaj povprečne vzorce za vsak razred v y_data
  y_avg_samples = {cls: np.mean(y_data[cls], axis=0) for cls in y_classes}

  # Izračunaj povprečne vzorce za vsak razred v z_data
  z_avg_samples = {cls: np.mean(z_data[cls], axis=0) for cls in y_classes}
  for cls in y_classes:
      #  izpisa povprečnih vzorcev za pripandike razreda
      print("Povprečni vzorec iz razreda ", cls)
      print(y_avg_samples[cls])
      print("Povprečni vzorec, ki ni iz razreda", cls)
      print(z_avg_samples[cls])

  # Inicializacija y_desired matrike
  y_desired = np.zeros_like(X)

  for cls in y_classes:
     # Popravljeni vzorci razreda
     y_desired[y == cls] = (1111 * y_avg_samples[cls] - z_avg_samples[cls]+y_data[cls]) / 1111

  return y_desired




selected_dataset = choose_dataset()
etapa=data_analysis(selected_dataset)
print("*****************")
print(etapa[0:3,:])
print(selected_dataset.data[0:3,:])
print()
print(etapa[50:53,:])
print(selected_dataset.data[50:53,:])
print()
print(etapa[100:103,:])
print(selected_dataset.data[100:103,:])
print()

# Definirajte enoslojni model s 4 nevroni in enotsko začetno matriko uteži

(rows,columns)=etapa.shape
print("vseh stolpcev je+++++++++++++++++++++++++++++++++++++++++", columns)
print("Oblika vhodnih podatkov (X):", etapa.shape)
model = tf.keras.Sequential([



    tf.keras.layers.Dense(columns, activation='relu', input_shape=(columns,), kernel_initializer='identity')
])

# Nastavi learning rate
learning_rate = 0.01630  # Tvoj izbran learning rate

# Uporabi Adam optimizator z določeno learning rate
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
#####

# Nastavi callback za izpisovanje le vsakih 10 epoh
checkpoint = ModelCheckpoint("model_checkpoint.h5", monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', save_freq=10)
X=selected_dataset.data
y=etapa
# Primer uporabe podatkov v modelu z uporabo callback-a
rezultat = model.fit(X, y, epochs=333, validation_data=(X, y), callbacks=[checkpoint])

# Primer uporabe podatkov v modelu
#rezultat = model.fit(X, y, epochs=16, validation_data=(X, y))
np.set_printoptions(precision=5)

print("prvi vhodi\n", X[0:5, :])
print("prvi izhodi želeni\n", y[0:5, :])
print("prvi izhodi dejanski\n", model.predict(X[0:5, :]))
print("sprememba\n", y[0:5, :] - X[0:5, :])
print("napaka\n", model.predict(X[0:5, :]) - y[0:5, :])





print("Konec")