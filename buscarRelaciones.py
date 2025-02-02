import cv2
import numpy as np
import pickle
import funciones
from ultralytics import YOLO
import os


ruta_diccionario = f'diccionario{'0000'}.pkl'


def imprimir_labels(diccionario, idImagen):
    # Verificar si el idImagen existe en el diccionario
    if idImagen not in diccionario:
        print(f"Error: no se encontró el idImagen {idImagen} en el diccionario.")
        return
    
    # Recorrer las clases dentro del idImagen
    for clase_id, datos in diccionario[idImagen].items():
        # Obtener los datos del label
        label = datos['label']
        
        # Imprimir los detalles del label
        print(clase_id)
        print(label)




# Cargar el diccionario desde el archivo .pickle
try:
    with open(ruta_diccionario, 'rb') as file:
        diccionario = pickle.load(file)
    print("Diccionario cargado exitosamente.")
except FileNotFoundError:
    # Si el archivo no existe, se crea un diccionario vacío
    diccionario = {}
    print("No se encontró el archivo, se creará un diccionario vacío.")

#print(diccionario)
#imprimir_labels(diccionario,'000000')
#imprimir_labels(diccionario,'000001')
relaciones = funciones.relacionar_objetos(diccionario)

print(relaciones)
#print(diccionario['000000'][3]['label'])
#print(diccionario['000001'][5]['label'])

#print(diccionario)