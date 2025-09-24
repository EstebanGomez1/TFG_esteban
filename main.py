import cv2
import numpy as np
import pickle
import funciones
from ultralytics import YOLO
import os

# Definicion de rutas

idImagen = "000003"
ruta_kitti = '/home/esteban/universidad/curso/datasets/subset_kitti'

ruta_imagen = f'{ruta_kitti}/image_2/{idImagen}.png'
ruta_lidar = f'{ruta_kitti}/velodyne/{idImagen}.bin'
ruta_calibracion = f'{ruta_kitti}/calib/{idImagen}.txt'
ruta_label = f'{ruta_kitti}/label_2/{idImagen}.txt'
ruta_diccionario = 'diccionario.pkl'

# Cargar el diccionario desde el archivo .pickle
try:
    with open(ruta_diccionario, 'rb') as file:
        diccionario = pickle.load(file)
    print("Diccionario cargado exitosamente.")
except FileNotFoundError:
    # Si el archivo no existe, se crea un diccionario vacío
    diccionario = {}
    print("No se encontró el archivo, se creará un diccionario vacío.")

# Cargar imagen
#imagen = cv2.imread(ruta_imagen)

#funciones.inferencia(imagen, idImagen, ruta_label, ruta_lidar, ruta_diccionario, ruta_calibracion, 1)

#############################################

# Definir el rango de idImagen desde "000000" hasta "000010" (puedes cambiar el valor de x)
start_id = 0  # Representa 000000
end_id = 700  # Representa 000700

# Bucle para recorrer el rango de idImagen
for i in range(start_id, end_id + 1):
    # Formatear idImagen con ceros a la izquierda
    idImagen = f"{i:06d}"  # Esto genera "000000", "000001", ..., "000010"
    
    # Definir las rutas para los archivos
    ruta_imagen = f'{ruta_kitti}/image_2/{idImagen}.png'
    ruta_lidar = f'{ruta_kitti}/velodyne/{idImagen}.bin'
    ruta_calibracion = f'{ruta_kitti}/calib/{idImagen}.txt'
    ruta_label = f'{ruta_kitti}/label_2/{idImagen}.txt'
    ruta_diccionario = 'diccionario.pkl'

    # Verificar si los archivos existen antes de procesarlos
    if os.path.exists(ruta_imagen) and os.path.exists(ruta_label) and os.path.exists(ruta_lidar) and os.path.exists(ruta_calibracion):
        # Cargar imagen
        imagen = cv2.imread(ruta_imagen)

        # Llamar a la función de inferencia
        diccionario = funciones.inferencia(imagen, idImagen, ruta_label, ruta_lidar, diccionario, ruta_calibracion, False, 0)
        # Guardar el diccionario actualizado en el archivo
        with open(ruta_diccionario, 'wb') as file:
            pickle.dump(diccionario, file)
    else:
        print(f"Archivos no encontrados para idImagen {idImagen}. Verifica las rutas.")