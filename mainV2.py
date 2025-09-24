import cv2
import numpy as np
import pickle
import funciones
from ultralytics import YOLO
import os

# Definicion de rutas

idImagen = "000000"
idSeccion = "0000"
ruta_kitti = '/media/esteban/DISCO EXT/TFG'

ruta_imagen = f'{ruta_kitti}/imagenes/{idSeccion}/{idImagen}.png'
ruta_lidar = f'{ruta_kitti}/velodyne/{idSeccion}/{idImagen}.bin'
ruta_calibracion = f'{ruta_kitti}/data_tracking_calib/training/calib/{idSeccion}.txt'
ruta_label = f'{ruta_kitti}/data_tracking_label_2/training/label_02/{idSeccion}.txt'
ruta_diccionario = f'diccionario{idSeccion}.pkl'


# Cargar el diccionario desde el archivo .pickle
try:
    with open(ruta_diccionario, 'rb') as file:
        diccionario = pickle.load(file)
    print("Diccionario cargado exitosamente.")
except FileNotFoundError:
    # Si el archivo no existe, se crea un diccionario vacío
    diccionario = {}
    print("No se encontró el archivo, se creará un diccionario vacío.")


# Definir el rango de idImagen desde "000000" hasta "000010" (puedes cambiar el valor de x)
start_id = 2  # Representa 000000
end_id = 2  # Representa 000700

# Bucle para recorrer el rango de idImagen
for i in range(start_id, end_id + 1):
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
        diccionario = funciones.inferencia2(imagen, idImagen, ruta_label, ruta_lidar, diccionario, ruta_calibracion, True, 0, 2) #Con 0 no info, 1 toda info, 2 info imagenes ; 1 estrucura normal, 2 estructura tracking
        # Guardar el diccionario actualizado en el archivo
        with open(ruta_diccionario, 'wb') as file:
            pickle.dump(diccionario, file)
    else:
        print(f"Archivos no encontrados para idImagen {idImagen}. Verifica las rutas.")

#print(diccionario)

