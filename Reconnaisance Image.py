# Importation des librairies Primaires

import tensorflow as tf
from tensorflow import keras
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Importation des librairies Secondaires
import matplotlib.pyplot as plt
from random import randint

# Récupération des information du dataset

mnist = keras.datasets.mnist #Dataset de base composé de chiffre (0-9) écrit à la main

(train_image, train_labels), (test_images, test_labels) = mnist.load_data()

train_image = train_image.astype("float32") / 255
test_images = test_images.astype("float32") / 255
#print(train_image[2020])

print("Label: ", train_labels[0])
plt.imshow(train_image[0], cmap='rainbow')
plt.show()

# Création d'un modèle

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Entrainement du modèle

def model_training(E):
    epochs = E
    if epochs == 0:
        return
    else:
        model.fit(train_image, train_labels, epochs=epochs)
        model.save_weights('/Users/Roman/Desktop/NSI/Projet_Reconnaisance_Image/model.h5')
        print('## Le modèle bien été enregistré ##')
        return "------------------ Fin de l'entrainement ------------------\n"

# Prédiction du réseau sur des images des test connues (on connait la veleur du chiffre)

def prediction(I):
    print('## Récuperation des données du modèle ##')
    model.load_weights('/Users/Roman/Desktop/NSI/Projet_Reconnaisance_Image/model.h5')
    for i in range(I):
        j = randint(0,10000)
        predictions = model.predict(test_images)
        #print(predictions[j])
        print('------------------------------------')
        print("Numéro predit",np.argmax(predictions[j]))
        print("Réalité",test_labels[j])
    return "------------------ Fin de la prédiction ------------------\n"

# Prédiction du réseau sur un dataset totalement inconnue (on veut determiner la valeur du chiffre)

''' A faire '''


# Evaluation du reseau

def evaluation():
    model.load_weights('/Users/Roman/Desktop/NSI/Projet_Reconnaisance_Image/model.h5') # On fait appelle a la memoir du reseau de neuronne
    print("------------------ Evaluation de la préscision du modèle ------------------")
    score = model.evaluate(test_images, test_labels, verbose=0) # Evaluation du modèle
    print("Perte:", score[0])
    print("Précsion du test:", score[1],'\n')
    return ""

# Affichage des graphiques avec matplotlib

''' A faire '''

# Affichage via TKinter



# Lancement du reseau
print(evaluation())
print(model_training(0)) # Pour changer le nombre d'entrainement, changer la valeur de la varible "E" de la fonction
print(prediction(3)) # Pour changer le nombre de prediction, changer la valeur de la variable "I" de la fonction
print(evaluation())



'''
Use this to load the model in any other file :

from tensorflow import keras
model = keras.models.load_model('/Users/Roman/Desktop/NSI/Projet_Reconnaisance_Image/model.h5')

'''