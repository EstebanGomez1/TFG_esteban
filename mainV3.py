import cv2
import numpy as np
import pickle
import funciones3
from ultralytics import YOLO
import os
from tqdm import tqdm
import time

# Definicion de rutas

idImagen = "000000"
idSeccion = "0000"
ruta_kitti = 'datos'

ruta_imagen = f'{ruta_kitti}/imagenes/{idSeccion}/{idImagen}.png'
ruta_lidar = f'{ruta_kitti}/velodyne/{idSeccion}/{idImagen}.bin'
ruta_calibracion = f'{ruta_kitti}/data_tracking_calib/training/calib/{idSeccion}.txt'
ruta_label = f'{ruta_kitti}/data_tracking_label_2/training/label_02/{idSeccion}.txt'
#ruta_diccionario = f'diccionario{idSeccion}.pkl'
#ruta_diccionario = f'diccionarioPrueba.pkl'

# Cargar el diccionario desde el archivo .pickle
def load_dictionary(file_pkl, replace = True):
    if(replace):
        if os.path.exists(file_pkl):
            os.remove(file_pkl)
        diccionario = {}
        with open(file_pkl, 'wb') as file:
            pickle.dump(diccionario, file)
    else:
        try:
            with open(file_pkl, 'rb') as file:
                diccionario = pickle.load(file)
            print(f"Diccionario {file_pkl} cargado exitosamente.")
        except FileNotFoundError:
            print(f"Error: El archivo {file_pkl} no fue encontrado.")
            diccionario = {}
            with open(file_pkl, 'wb') as file:
                pickle.dump(diccionario, file)
    return diccionario

def save_dictionary(diccionario, file_pkl):
    with open(file_pkl, 'wb') as file:
        pickle.dump(diccionario, file)
    print(f"Diccionario guardado en {file_pkl}.")

# Crear diccionario
ruta_diccionario = f'dicc4.pkl'
diccionario = load_dictionary(ruta_diccionario)

# Definir el rango de idImagen desde "000000" hasta "000010" (puedes cambiar el valor de x)
start_id = 0  # Representa 000000
end_id = 155  # Representa 000700

diccs = {}
# Bucle para recorrer el rango de idImagen
for i in tqdm(range(start_id, end_id + 1)):
    # Formatear idImagen con ceros a la izquierda
    idImagen = f"{i:06d}"  # Esto genera "000000", "000001", ..., "000010"

    # Definir las rutas para los archivos
    ruta_imagen = f'{ruta_kitti}/imagenes/{idSeccion}/{idImagen}.png'
    ruta_lidar = f'{ruta_kitti}/velodyne/{idSeccion}/{idImagen}.bin'
    ruta_calibracion = f'{ruta_kitti}/data_tracking_calib/training/calib/{idSeccion}.txt'
    ruta_label = f'{ruta_kitti}/data_tracking_label_2/training/label_02/{idSeccion}.txt'

    # Verificar si los archivos existen antes de procesarlos
    if os.path.exists(ruta_imagen) and os.path.exists(ruta_label) and os.path.exists(ruta_lidar) and os.path.exists(ruta_calibracion):
        # Cargar imagen
        imagen = cv2.imread(ruta_imagen)
        
        # Llamar a la función de inferencia
        diccionario = funciones3.inferencia2(imagen, idImagen, ruta_label, ruta_lidar, diccionario, ruta_calibracion, True, 0, 2) #Con 0 no info, 1 toda info, 2 info imagenes ; 1 estrucura normal, 2 estructura tracking
        diccs[idImagen] = diccionario[idImagen]
    else:
        print(f"Archivos no encontrados para idImagen {idImagen}. Verifica las rutas.")

# Guardar el diccionario actualizado en el archivo
with open(ruta_diccionario, 'wb') as file:
    pickle.dump(diccs, file)

print(diccs)

