import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pickle
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import os
from tqdm import tqdm
from pprint import pprint

#####################

# Seccion 1: Lectura y guardado de archivos

#####################

 
def leer_puntos_lidar(ruta_archivo):
    puntos = np.fromfile(ruta_archivo, dtype=np.float32).reshape(-1, 4)
    return puntos[:, :4]  # Solo X, Y, Z

def leer_matrices_calibracion(archivo_calibracion):
    P2 = None
    R0_rect = None
    Tr_velo_to_cam = None
    with open(archivo_calibracion, 'r') as f:
        for line in f:
            if line.startswith("P2:"):
                P2 = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)
            elif line.startswith("R0_rect:") or line.startswith("R_rect"):
                R0_rect = np.array(line.split()[1:], dtype=np.float32).reshape(3, 3)
            elif line.startswith("Tr_velo_to_cam:") or line.startswith("Tr_velo_cam"):
                Tr_velo_to_cam = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)
    return P2, R0_rect, Tr_velo_to_cam

def leer_labels_kitti(ruta, idImagen=0, Tr_velo_to_cam=None, R0_rect=None):
    etiquetas = []
    with open(ruta, 'r') as file:
        for linea in file:
            datos = linea.split()
            if int(datos[0]) == idImagen:
                etiqueta = {
                    'idImagen': float(datos[0]),
                    'id': datos[1].lower(),
                    'class': datos[2].lower(),
                    'x': float(datos[13]),
                    'y': float(datos[14]),
                    'z': float(datos[15]),
                    'height': float(datos[10]),
                    'width': float(datos[11]),
                    'length': float(datos[12]),
                    'rot_y': float(datos[16])
                }
                # --- CONVERTIR A LIDAR ---
                if Tr_velo_to_cam is not None and R0_rect is not None:
                    label_xyz_kitti = [etiqueta['x'], etiqueta['y'], etiqueta['z']]
                    label_xyz_lidar = label_kitti_a_lidar(label_xyz_kitti, Tr_velo_to_cam, R0_rect)
                    etiqueta['x'] = label_xyz_lidar[0]
                    etiqueta['y'] = label_xyz_lidar[1]
                    etiqueta['z'] = label_xyz_lidar[2]
                etiquetas.append(etiqueta)
    return etiquetas

#####################

# Seccion 2: Proyeccion y Filtrado

#####################


def proyectar_punto_3d_a_2d(punto_3d, P2, R0_rect, Tr_velo_to_cam):
    punto_hom = np.append(punto_3d[:3], 1) #punto homogeneo
    punto_cam = R0_rect @ (Tr_velo_to_cam @ punto_hom)[:3]
    punto_img_hom = P2 @ np.append(punto_cam, 1)
    u, v, w = punto_img_hom
    u /= w
    v /= w
    return u, v

def filtrar_outliers(puntos, umbral=1):
    # Calcula la media y la desviación estándar
    media = np.mean(puntos, axis=0)
    desviacion_std = np.std(puntos, axis=0)
    
    # Filtrar puntos que están dentro de threshold desviaciones de la media
    puntos_filtrados = [punto for punto in puntos if np.all(np.abs(punto - media) <= umbral * desviacion_std)]
    return np.array(puntos_filtrados)

def filtrar_puntos_por_distancia(puntos, centro, umbral):
    """
    Filtra los puntos que están dentro de una distancia umbral desde un centro dado.
    """
    #distancias = np.linalg.norm(puntos - centro, axis=1)
    distancias = np.linalg.norm(puntos[:, :3] - centro, axis=1)
    puntos_filtrados = puntos[distancias <= umbral]
    return puntos_filtrados

def visualizar_puntos_3d(puntos, titulo="Visualización 3D"):
    """
    Visualiza los puntos en 3D, mostrando solo aquellos que están por encima de un valor límite en el eje Z.

    Parámetros:
        puntos (ndarray): Nube de puntos LiDAR (X, Y, Z).
        titulo (str): Título de la visualización.
        limite (float): Límite inferior para la coordenada Z. Solo se visualizan los puntos con Z > limite.
    """
    # Filtrar los puntos que están por encima del límite en Z
    puntos_filtrados = puntos



    # Crear la visualización
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        puntos_filtrados[:, 0], 
        puntos_filtrados[:, 1], 
        puntos_filtrados[:, 2], 
        s=1, 
        c=puntos_filtrados[:, 2], 
        cmap='viridis'
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{titulo}")
    plt.show()

def mostrar_imagen(imagen):
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def dibujar_caja(diccionario_label, puntos_lidar):
    """
    Dibuja una caja 3D alrededor de un objeto basado en el diccionario de etiquetas de KITTI.
    
    Parámetros:
        diccionario_label (dict): Diccionario con los datos del objeto (clase, centro, dimensiones, rotación).
        puntos_lidar (array): Nube de puntos LiDAR para la visualización.
    """

    
    # Extraer los datos del objeto
    tipo = diccionario_label['class']
    x, y, z = diccionario_label['z'], (-1)*diccionario_label['x'], ((-1/2)*diccionario_label['y'])
    length, height, width = diccionario_label['length'], (-1)*diccionario_label['width'], diccionario_label['height']
    rot_y = diccionario_label['rot_y']
    # 'width' 'length' 'height'
    # Definir los 8 vértices de la caja 3D sin rotación (en el sistema de coordenadas local)
    vertices = np.array([
        [length / 2, width / 2, height / 2],
        [-length / 2, width / 2, height / 2],
        [-length / 2, -width / 2, height / 2],
        [length / 2, -width / 2, height / 2],
        [length / 2, width / 2, -height / 2],
        [-length / 2, width / 2, -height / 2],
        [-length / 2, -width / 2, -height / 2],
        [length / 2, -width / 2, -height / 2]
    ])
    
    # Matriz de rotación alrededor del eje Y
    R = np.array([
        [np.cos(rot_y), 0, np.sin(rot_y)],
        [0, 1, 0],
        [-np.sin(rot_y), 0, np.cos(rot_y)]
    ])
    
    # Aplicar la rotación a los vértices
    vertices_rotados = vertices.dot(R.T)
    
    # Trasladar los vértices a la posición del objeto (x, y, z)
    vertices_rotados[:, 0] += x
    vertices_rotados[:, 1] += y
    vertices_rotados[:, 2] += z
    
    # Definir las caras de la caja (conexiones entre los vértices)
    caras = [
        [vertices_rotados[0], vertices_rotados[1], vertices_rotados[2], vertices_rotados[3]],
        [vertices_rotados[4], vertices_rotados[5], vertices_rotados[6], vertices_rotados[7]],
        [vertices_rotados[0], vertices_rotados[1], vertices_rotados[5], vertices_rotados[4]],
        [vertices_rotados[1], vertices_rotados[2], vertices_rotados[6], vertices_rotados[5]],
        [vertices_rotados[2], vertices_rotados[3], vertices_rotados[7], vertices_rotados[6]],
        [vertices_rotados[3], vertices_rotados[0], vertices_rotados[4], vertices_rotados[7]]
    ]
    
    # Crear la visualización 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujar la nube de puntos LiDAR
    ax.scatter(puntos_lidar[:, 0], puntos_lidar[:, 1], puntos_lidar[:, 2], c='blue', s=0.1)

    # Dibujar la caja 3D
    ax.add_collection3d(Poly3DCollection(caras, facecolors='lightgray', linewidths=1, edgecolors='r', alpha=0.1))
    
    # Establecer límites y etiquetas
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Objeto: {tipo}')
    
    plt.show()

#####################

# Seccion 4: Clustering y Asociaciones de label

#####################

def label_kitti_a_lidar(label_xyz, Tr_velo_to_cam, R0_rect):
    pt_cam = np.array(label_xyz)
    pt_rect = np.linalg.inv(R0_rect) @ pt_cam
    pt_rect_hom = np.append(pt_rect, 1)
    Tr_lidar_to_cam_4x4 = np.eye(4)
    Tr_lidar_to_cam_4x4[:3, :4] = Tr_velo_to_cam
    pt_lidar_hom = np.linalg.inv(Tr_lidar_to_cam_4x4) @ pt_rect_hom
    return pt_lidar_hom[:3]


#####################

# Seccion Final: Extraccion

#####################


def perfect_extraction(idImagen, ruta_label, ruta_lidar, ruta_calibracion):
    diccionario = {}

    # Leer matrices de calibración
    P2, R0_rect, Tr_velo_to_cam = leer_matrices_calibracion(ruta_calibracion)

    # Leer labels de ground truth directamente
    etiquetas = leer_labels_kitti(ruta_label, int(idImagen), Tr_velo_to_cam, R0_rect)
    # Leer puntos LiDAR
    puntos_lidar = leer_puntos_lidar(ruta_lidar)
    diccionario[idImagen] = {}
    # Iterar sobre cada etiqueta para extraer el crop perfecto
    for etiqueta in etiquetas:
        
        clase = etiqueta.get('class', '').lower()
        if clase == 'dontcare':
            continue  # Omitir objetos DontCare

        label_lidar = np.array([etiqueta['x'], etiqueta['y'], etiqueta['z']])

        # Recortar puntos alrededor del centro del objeto
        puntos_crop = filtrar_puntos_por_distancia(puntos_lidar, label_lidar, 3)
        if(len(puntos_crop) <3):
            continue
        
        # Guardar en diccionario
        objeto_id = etiqueta.get('id', '-1')
        diccionario[idImagen][objeto_id] = {
            "points": puntos_crop,
            "label": etiqueta
        }
    return diccionario

def image_processing(ruta_kitti, idSeccion, ruta_diccionario):
    diccs = {}

    ruta = f'{ruta_kitti}/velodyne/{idSeccion}'
    archivos = [f for f in os.listdir(ruta) if os.path.isfile(os.path.join(ruta, f))]
    cantidad = len(archivos)

    for i in tqdm(range(cantidad)):
        try:
            idImagen = f"{i:06d}"

            ruta_lidar = f'{ruta_kitti}/velodyne/{idSeccion}/{idImagen}.bin'
            ruta_calibracion = f'{ruta_kitti}/data_tracking_calib/training/calib/{idSeccion}.txt'
            ruta_label = f'{ruta_kitti}/data_tracking_label_2/training/label_02/{idSeccion}.txt'
            
            if os.path.exists(ruta_label) and os.path.exists(ruta_lidar) and os.path.exists(ruta_calibracion):
                diccionario = perfect_extraction(idImagen, ruta_label, ruta_lidar, ruta_calibracion)
                diccs.update(diccionario)
            else:
                print(f"Archivos no encontrados para idImagen {idImagen}. Verifica las rutas.")
        except Exception as e:
            print(f"Error en la iteración {idImagen}: {e}")

    with open(ruta_diccionario, 'wb') as file:
        pickle.dump(diccs, file)

    print(f"Diccionario completo guardado en {ruta_diccionario}")


### Main ###

ruta_kitti = 'datos'

for num_seccion in range(0,1):
    idSeccion = f"{num_seccion:04d}"
    ruta_diccionario = f'dic2_perf{num_seccion}.pkl'
    print(f"\n-------------- Procesando {idSeccion} -> {ruta_diccionario} --------------")

    image_processing(ruta_kitti, idSeccion, ruta_diccionario)
 