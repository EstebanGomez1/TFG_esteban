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
                #determinar dificultad:
                difficulty = 0
                trunc = float(datos[3])
                occ = float(datos[4])
                h_px = float(datos[9]) - float(datos[7])
                if occ == 3:
                    difficulty = -1 #ignorar
                elif h_px >= 40 and occ == 0 and trunc <= 0.15:
                    difficulty = 1
                elif h_px >= 25 and occ <= 1 and trunc <= 0.30:
                    difficulty = 2
                elif h_px >= 25 and occ <= 2 and trunc <= 0.50:
                    difficulty = 3
                else:
                    difficulty = -1
                etiqueta = {
                    'idImagen': float(datos[0]),
                    'id': datos[1].lower(),
                    'clase': datos[2].lower(),
                    'difficulty' : float(difficulty),
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

def guardar_imagenes_procesadas(imagen, idImagen):
    carpeta = "imagenesProcesadas"  # Nombre de la carpeta donde guardar las imágenes

    # Crear la carpeta si no existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        print(f"Carpeta creada: {carpeta}")
    
    # Nombre del archivo basado en el idImagen
    nombre_archivo = f"{idImagen}.jpg"
    
    # Ruta completa del archivo
    ruta_completa = os.path.join(carpeta, nombre_archivo)
    
    # Guardar la imagen
    cv2.imwrite(ruta_completa, imagen)



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

def encontrar_centro_bounding_box_y_nube(puntos_lidar, bounding_box, P2, R0_rect, Tr_velo_to_cam, threshold=3):
    x_min, y_min, x_max, y_max = bounding_box
    puntos_en_bounding_box = []

    

    # Proyectar cada punto LiDAR y verificar si está dentro del bounding box
    for punto in puntos_lidar:
        u, v = proyectar_punto_3d_a_2d(punto[:3], P2, R0_rect, Tr_velo_to_cam)
        if x_min <= u <= x_max and y_min <= v <= y_max:
            # Filtrar puntos en frente de la cámara usando Z (profundidad)
            punto_hom = np.append(punto[:3], 1)
            punto_cam = R0_rect @ (Tr_velo_to_cam @ punto_hom)[:3]  # Punto en sistema de cámara
            if punto_cam[2] > 0:  # Z > 0 significa "en frente"
                puntos_en_bounding_box.append(punto)
    
    if puntos_en_bounding_box:
        # Filtrar outliers usando la desviación estándar
        puntos_en_bounding_box = filtrar_outliers(puntos_en_bounding_box)
        
        
        if len(puntos_en_bounding_box) > 0:
            # Calcular el centro en 3D usando la media de los puntos filtrados
            #centro_3d = np.mean(puntos_en_bounding_box, axis=0)
            centro_3d = np.mean(puntos_en_bounding_box[:, :3], axis=0)
            puntos_filtrados = filtrar_puntos_por_distancia(puntos_lidar,centro_3d, threshold)
            return centro_3d, puntos_filtrados
    print("No se encontraron puntos LiDAR significativos dentro del bounding box.")
    return None, None


def encontrar_centro_bounding_box_y_nube1(puntos_lidar, bounding_box, P2, R0_rect, Tr_velo_to_cam, threshold=3):
    x_min, y_min, x_max, y_max = bounding_box
    puntos_en_bounding_box = []

    # Proyectar cada punto LiDAR y verificar si está dentro del bounding box
    for punto in puntos_lidar:
        
        u, v = proyectar_punto_3d_a_2d(punto, P2, R0_rect, Tr_velo_to_cam)
        if x_min <= u <= x_max and y_min <= v <= y_max:
            # Filtrar puntos en frente de la cámara usando Z (profundidad)
            punto_hom = np.append(punto, 1)
            punto_cam = R0_rect @ (Tr_velo_to_cam @ punto_hom)[:3]  # Punto en sistema de cámara
            if punto_cam[2] > 0:  # Z > 0 significa "en frente"
                puntos_en_bounding_box.append(punto)
    
    if puntos_en_bounding_box:
        # Filtrar outliers usando la desviación estándar
        puntos_en_bounding_box = filtrar_outliers(puntos_en_bounding_box)
        
        
        if len(puntos_en_bounding_box) > 0:
            # Calcular el centro en 3D usando la media de los puntos filtrados
            centro_3d = np.mean(puntos_en_bounding_box, axis=0)
            #puntos_filtrados = filtrar_puntos_por_distancia(puntos_lidar,centro_3d, threshold)
            return centro_3d, puntos_en_bounding_box
    print("No se encontraron puntos LiDAR significativos dentro del bounding box.")
    return None

def filtrar_puntos_por_distancia(puntos, centro, umbral):
    """
    Filtra los puntos que están dentro de una distancia umbral desde un centro dado.
    """
    #distancias = np.linalg.norm(puntos - centro, axis=1)
    distancias = np.linalg.norm(puntos[:, :3] - centro, axis=1)
    puntos_filtrados = puntos[distancias <= umbral]
    return puntos_filtrados


#####################

# Seccion 3: Visualizacion de Puntos e imagen

#####################


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

def visualizar_nube_puntos_plotly(puntos):
    """
    Visualiza una nube de puntos en 3D usando Plotly.
    
    Args:
        puntos (ndarray): Array de puntos (N, 3) en formato (x, y, z).
    """
    fig = go.Figure(
        data=[go.Scatter3d(
            x=puntos[:, 0], 
            y=puntos[:, 1], 
            z=puntos[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=puntos[:, 2],  # Colorear por coordenada Z
                colorscale='Viridis',  # Escala de color
                opacity=0.8
            )
        )]
    )

    fig.update_layout(scene=dict(aspectmode="data"))
    fig.show()

def visualizar_puntos_filtrados(todos_los_puntos):
    todos_los_puntos = np.vstack(todos_los_puntos) if len(todos_los_puntos) > 0 else np.array([])
    # Extraer las coordenadas X, Y y Z de los puntos
    x = todos_los_puntos[:, 0]
    y = todos_los_puntos[:, 1]
    z = todos_los_puntos[:, 2]

    # Crear la visualización 3D con plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,  # Tamaño de los puntos
            color=z,  # El color varía según la coordenada Z
            colorscale='Viridis',  # Escala de colores
            opacity=0.8
        )
    )])

    # Personalizar el layout
    fig.update_layout(
        title="Visualización 3D de Puntos LiDAR",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Mostrar la gráfica
    fig.show()

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

  
def cluster_mas_cercano(puntos, centro_bb, eps=0.5, min_samples=5):

    """
    Aplica DBSCAN a los puntos LiDAR y selecciona el cluster más cercano al centro del bounding box.
    
    Args:
        puntos (ndarray): Array de puntos LiDAR (N, 3).
        centro_bb (tuple): Coordenadas (x, y) del centro del bounding box.
        eps (float): Máxima distancia entre puntos para considerarlos en el mismo cluster.
        min_samples (int): Número mínimo de puntos para formar un cluster.
    
    Returns:
        ndarray: Puntos pertenecientes al cluster más cercano al centro del bounding box.
    """
    if len(puntos) == 0:
        return np.array([])

    # Aplicar DBSCAN para detectar clusters
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(puntos[:, :3])
    etiquetas = clustering.labels_

    # Inicializar variables
    cluster_mas_cercano = None
    distancia_minima = float('inf')

    # Iterar sobre los clusters detectados
    for etiqueta in np.unique(etiquetas):
        if etiqueta == -1:  # Ignorar ruido
            continue
        
        # Obtener los puntos del cluster actual
        puntos_cluster = puntos[etiquetas == etiqueta]
        
        # Calcular el centroide del cluster
        centroide = np.mean(puntos_cluster[:, :2], axis=0)  # Promedio en X, Y
        
        # Calcular distancia al centro del bounding box
        distancia = np.linalg.norm(centroide - np.array(centro_bb))
        
        # Seleccionar el cluster más cercano
        if distancia < distancia_minima:
            distancia_minima = distancia
            cluster_mas_cercano = puntos_cluster

    # Retornar el cluster más cercano o un array vacío si no se encuentra nada
    return cluster_mas_cercano if cluster_mas_cercano is not None else np.array([])


#####################

# Seccion 5: Secuencia de imagenes

#####################



def calcular_distancia(objeto1, objeto2):
    """
    Calcula la distancia euclídea entre dos objetos en 3D basándose en sus coordenadas x, y, z.
    """
    x1, y1, z1 = objeto1["label"]["x"], objeto1["label"]["y"], objeto1["label"]["z"]
    x2, y2, z2 = objeto2["label"]["x"], objeto2["label"]["y"], objeto2["label"]["z"]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def dimensiones_similares(objeto1, objeto2, tolerancia=0.75):
    """
    Comprueba si las dimensiones de dos objetos son similares dentro de un rango.
    """
    l1, h1, w1 = objeto1["label"]["length"], objeto1["label"]["height"], objeto1["label"]["width"]
    l2, h2, w2 = objeto2["label"]["length"], objeto2["label"]["height"], objeto2["label"]["width"]

    return (
        abs(l1 - l2) <= tolerancia * l1 and
        abs(h1 - h2) <= tolerancia * h1 and
        abs(w1 - w2) <= tolerancia * w1
    )

def mismoObjeto(objeto1, objeto2, umbral_distancia=2.0, tolerancia_dimensiones=0.2):
    """
    Determina si dos objetos son el mismo basándose en la distancia euclídea y dimensiones similares.
    """
    return (
        calcular_distancia(objeto1, objeto2) <= umbral_distancia and
        dimensiones_similares(objeto1, objeto2, tolerancia_dimensiones)
    )

def relacionar_objetos(diccionario):
    """
    Relaciona los objetos de una sección constituida por imágenes secuenciales.

    return: Diccionario donde la clave es la clase_id de un objeto en la imagen anterior,
            y el valor es la clase_id del objeto en la imagen actual.
    """
    diccionarioRelaciones = {}
    idImagenAnterior = None

    for idImagenActual, objetosActuales in diccionario.items():
        if idImagenAnterior is not None:

            # Inicializamos el diccionario de relaciones para esta imagen
            diccionarioRelaciones[idImagenAnterior] = {}

            # Buscamos relaciones entre los objetos de la imagen anterior y los actuales
            for clase_id_actual, objetoActual in objetosActuales.items():
                for clase_id_anterior, objetoAnterior in diccionario[idImagenAnterior].items():
                    if mismoObjeto(objetoAnterior, objetoActual):
                        # Relacionamos los objetos
                        diccionarioRelaciones[idImagenAnterior][clase_id_anterior] = clase_id_actual
                        break  # Una vez encontrado, no seguimos buscando para este objeto actual

        # Actualizamos la imagen anterior
        idImagenAnterior = idImagenActual

    return diccionarioRelaciones



#####################

# Seccion 6: Inferencia con YOLOv8

#####################

#--------------------------------------------

def _lidar_center_and_crop(puntos_lidar, x_min, y_min, x_max, y_max, datos_calibracion, threshold=3):
    puntos_candidatos =[]
    for punto in puntos_lidar:
        u, v = proyectar_punto_3d_a_2d(punto[:3], datos_calibracion["P2"], datos_calibracion["R0_rect"], datos_calibracion["Tr_velo_to_cam"])
        # Verifica que esté delante de la cámara
        punto_hom = np.append(punto[:3], 1)
        punto_cam = datos_calibracion["R0_rect"] @ (datos_calibracion["Tr_velo_to_cam"] @ punto_hom)[:3]
        if punto_cam[2] > 0:
            if x_min <= u <= x_max and y_min <= v <= y_max:
                puntos_candidatos.append(punto)
    if len(puntos_candidatos) == 0:
        print("No se encontraron puntos LiDAR cerca del centro del box YOLO.")
        return None, None
    puntos_candidatos = np.array(puntos_candidatos)
    # Calcula el centroide de estos puntos (centro 3D estimado del objeto)
    centro_3d = np.mean(puntos_candidatos[:, :3], axis=0)
    # Crop final: todos los puntos a menos de 'radio_crop' unidades del centro 3D
    distancias = np.linalg.norm(puntos_lidar[:, :3] - centro_3d, axis=1)
    puntos_crop = puntos_lidar[distancias < threshold]

    return centro_3d, puntos_crop

from sklearn.cluster import DBSCAN

def _lidar_center_and_crop(
    puntos_lidar, x_min, y_min, x_max, y_max, datos_calibracion,
    eps=0.6, min_samples=12,
    base_radius=1.0, radius_per_meter=0.02,
    ground_percentile=15, ground_margin=0.1, min_points=20
):
    P2  = datos_calibracion["P2"]
    R0  = datos_calibracion["R0_rect"]
    Tr  = datos_calibracion["Tr_velo_to_cam"]

    candidatos = []
    for p in puntos_lidar:
        uv = proyectar_punto_3d_a_2d(p[:3], P2, R0, Tr)
        if uv is None:
            continue
        u, v = uv
        camz = (R0 @ (Tr @ np.append(p[:3],1.0))[:3])[2]
        if camz <= 0: 
            continue
        if x_min <= u <= x_max and y_min <= v <= y_max:
            candidatos.append(p)
    if not candidatos:
        return None, None
    candidatos = np.asarray(candidatos)

    z_vals = candidatos[:,2]
    z_floor = np.percentile(z_vals, ground_percentile) + ground_margin
    sin_suelo = candidatos[z_vals > z_floor]
    if len(sin_suelo) < max(min_points, len(candidatos)//10):
        sin_suelo = candidatos

    # Clusterizar en 3D
    X = sin_suelo[:, :3]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    # Elegir el cluster mayor (ignora ruido label==-1)
    best_label = None
    best_count = 0
    for lb in set(labels):
        if lb == -1: 
            continue
        cnt = np.sum(labels == lb)
        if cnt > best_count:
            best_count = cnt
            best_label = lb
    if best_label is None:
        # Si DBSCAN no encontró cluster claro, usa todos sin_suelo
        cluster = sin_suelo
    else:
        cluster = sin_suelo[labels == best_label]

    # Centro robusto del cluster principal
    centro_3d = np.median(cluster[:, :3], axis=0)

    depth = float(np.linalg.norm(centro_3d))
    radio_crop = base_radius + radius_per_meter * depth

    dist = np.linalg.norm(puntos_lidar[:, :3] - centro_3d, axis=1)
    puntos_crop = puntos_lidar[dist < radio_crop]
    if len(puntos_crop) < min_points:
        puntos_crop = candidatos

    return centro_3d, puntos_crop

def lidar_center_and_crop(
    puntos_lidar, x_min, y_min, x_max, y_max, datos_calibracion):
    P2  = datos_calibracion["P2"]
    R0  = datos_calibracion["R0_rect"]
    Tr  = datos_calibracion["Tr_velo_to_cam"]

    centro_3d, puntos = encontrar_centro_bounding_box_y_nube(
        puntos_lidar, [x_min, y_min, x_max, y_max], P2, R0, Tr
    )

    if centro_3d is None or puntos is None or len(puntos) == 0:
        return None, None
    
    # Clusterizar en 3D
    puntos_cluster_cercano = cluster_mas_cercano(
        puntos,
        centro_bb=(centro_3d[0], centro_3d[1]),
        eps=0.6,
        min_samples=5
    )

    if puntos_cluster_cercano is None or len(puntos_cluster_cercano) == 0:
        puntos_cluster_cercano = puntos

    return centro_3d, puntos_cluster_cercano


def asociar_label(idImagen, centro_lidar, etiquetas):
    mejor_match = None
    menor_distancia = float('inf')
    for i, etiqueta in enumerate(etiquetas):
        label_lidar = np.array([etiqueta['x'], etiqueta['y'], etiqueta['z']])
        distancia = np.linalg.norm(centro_lidar[:3] - label_lidar)
        if distancia < menor_distancia:
            mejor_match = (i, etiqueta, label_lidar, distancia)
            menor_distancia = distancia
    if mejor_match and menor_distancia <=3:  # threshold configurable
        i, etiqueta, label_lidar, distancia = mejor_match
        identificador = etiqueta["id"]
        etiqueta_label = {
            "id": identificador,
            "imagen": idImagen,
            "class": etiqueta['clase'],          
            "x": etiqueta['x'],
            "y": etiqueta['y'],
            "z": etiqueta['z'],
            "length": etiqueta['length'],
            "height": etiqueta['height'],
            "width": etiqueta['width'],
            "rot_y": etiqueta['rot_y'],
            "yolo_center": centro_lidar,
            "difficulty" : etiqueta["difficulty"]
        }
        return i, etiqueta_label
    else:
        print("No se ha encontrado asociación posible para este objeto")
        return None, None

#--------------------------------------------

def images_processing(rutaKitti, idSeccion, rutaDiccionario):
    diccionario = {}
    model = YOLO('yolov8n.pt').to('cuda')
    print(next(model.model.parameters()).device) 
    ruta = f'{rutaKitti}/velodyne/{idSeccion}'
    archivos = [f for f in os.listdir(ruta) if os.path.isfile(os.path.join(ruta, f))]
    cantidad = len(archivos)
    for i in tqdm(range(cantidad)):
        try:
            idImagen = f"{i:06d}"
            diccionario[idImagen]={}
            ruta_imagen = f'{rutaKitti}/imagenes/{idSeccion}/{idImagen}.png'
            ruta_lidar = f'{rutaKitti}/velodyne/{idSeccion}/{idImagen}.bin'
            ruta_calibracion = f'{rutaKitti}/data_tracking_calib/training/calib/{idSeccion}.txt'
            ruta_label = f'{rutaKitti}/data_tracking_label_2/training/label_02/{idSeccion}.txt'
            imagen = cv2.imread(ruta_imagen)
            detecciones = model.predict(imagen, device = 0, verbose=True)
            P2, R0_rect, Tr_velo_to_cam = leer_matrices_calibracion(ruta_calibracion)
            datosCalibracion = {
                "P2": P2,
                "R0_rect": R0_rect,
                "Tr_velo_to_cam": Tr_velo_to_cam
            }
            etiquetas = leer_labels_kitti(ruta_label, int(idImagen), Tr_velo_to_cam, R0_rect)
            nubeLidar = leer_puntos_lidar(ruta_lidar)
            if os.path.exists(ruta_label) and os.path.exists(ruta_lidar) and os.path.exists(ruta_calibracion) and imagen is not None:
                diccionarioImagen = pred_extraction(idImagen, etiquetas, nubeLidar, datosCalibracion, detecciones)
                diccionario[idImagen] = diccionarioImagen
            else:
                print(f"Archivos no encontrados para idImagen {idImagen}. Verifica las rutas.")
        except Exception as e:
            print(f"Error en la iteración {idImagen}: {e}")

    with open(rutaDiccionario, 'wb') as file:
        pickle.dump(diccionario, file)
    print(f"Diccionario completo guardado en {rutaDiccionario}")


from concurrent.futures import ProcessPoolExecutor, as_completed
import os, cv2, pickle
import torch
from ultralytics import YOLO
from tqdm import tqdm

# Worker CPU: LIDAR + calibración + recorte + asociación
def _cpu_trabajo(idImagen, ruta_imagen, ruta_lidar, ruta_calibracion, ruta_label, dets_serializables):
    try:
        P2, R0_rect, Tr_velo_to_cam = leer_matrices_calibracion(ruta_calibracion)
        datosCalibracion = {"P2": P2, "R0_rect": R0_rect, "Tr_velo_to_cam": Tr_velo_to_cam}
        etiquetas = leer_labels_kitti(ruta_label, int(idImagen), Tr_velo_to_cam, R0_rect)
        nubeLidar = leer_puntos_lidar(ruta_lidar)

        dicc_img = {}
        for x1, y1, x2, y2, cls_id, conf in dets_serializables:
            centro_3D, crop_lidar = lidar_center_and_crop(nubeLidar, x1, y1, x2, y2, datosCalibracion)
            if centro_3D is not None and len(crop_lidar) > 0:
                idx, etiqueta = asociar_label(idImagen, centro_3D, etiquetas)
                if idx is not None:
                    etiquetas.pop(idx)
                    dicc_img[etiqueta["id"]] = {"points": crop_lidar, "label": etiqueta}
        return idImagen, dicc_img
    except Exception as e:
        # Devuelve vacío si algo falla en el worker, pero no rompe el flujo
        return idImagen, {}

def _images_processing(rutaKitti, idSeccion, rutaDiccionario):
    diccionario = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO('yolov8n.pt').to(device)
    print(next(model.model.parameters()).device)

    dir_imgs  = f'{rutaKitti}/imagenes/{idSeccion}'
    dir_lidar = f'{rutaKitti}/velodyne/{idSeccion}'
    ruta_calibracion = f'{rutaKitti}/data_tracking_calib/training/calib/{idSeccion}.txt'
    ruta_label       = f'{rutaKitti}/data_tracking_label_2/training/label_02/{idSeccion}.txt'

    # Mantenemos tu lógica de conteo por índice
    archivos = [f for f in os.listdir(dir_lidar) if os.path.isfile(os.path.join(dir_lidar, f))]
    cantidad = len(archivos)

    # Cola de trabajos para la parte CPU
    futures = []
    num_workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for i in tqdm(range(cantidad)):
            try:
                idImagen = f"{i:06d}"
                diccionario[idImagen] = {}  # misma inicialización que tenías

                ruta_imagen = os.path.join(dir_imgs,  f"{idImagen}.png")
                ruta_lidar  = os.path.join(dir_lidar, f"{idImagen}.bin")

                if not (os.path.exists(ruta_imagen) and os.path.exists(ruta_lidar)
                        and os.path.exists(ruta_calibracion) and os.path.exists(ruta_label)):
                    print(f"Archivos no encontrados para idImagen {idImagen}. Verifica las rutas.")
                    continue

                imagen = cv2.imread(ruta_imagen)
                if imagen is None:
                    print(f"Error al leer: {ruta_imagen}")
                    continue

                # --- YOLO en el proceso principal (GPU) ---
                results = model.predict(imagen, device=0 if device == "cuda" else None, verbose=False)

                # Serializamos solo lo necesario para el worker (ultralytics.Result no es picklable)
                dets_serializables = []
                r = results[0]
                if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes):
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                        cls_id = int(b.cls.item())
                        conf   = float(b.conf.item())
                        dets_serializables.append((x1, y1, x2, y2, cls_id, conf))

                # --- Encola la parte CPU (LIDAR + recorte + asociación) ---
                futures.append(
                    ex.submit(_cpu_trabajo, idImagen, ruta_imagen, ruta_lidar,
                              ruta_calibracion, ruta_label, dets_serializables)
                )

            except Exception as e:
                print(f"Error en la iteración {idImagen}: {e}")

        # Recoger resultados a medida que terminan
        for fut in as_completed(futures):
            idImagen, dicc_img = fut.result()
            diccionario[idImagen] = dicc_img

    # Guardado final (igual que antes)
    os.makedirs(os.path.dirname(rutaDiccionario), exist_ok=True)
    with open(rutaDiccionario, 'wb') as file:
        pickle.dump(diccionario, file)
    print(f"Diccionario completo guardado en {rutaDiccionario}")



def pred_extraction(idImagen, etiquetas, nubeLidar, datosCalibracion, detecciones):
    print(f"Procesando imagen {idImagen}")
    diccionario = {}
    for r in detecciones:
        for box in r.boxes:
            # Obtener la clase y coordenadas del bounding box
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
            centro_3D, crop_lidar = lidar_center_and_crop( nubeLidar, x_min, y_min, x_max, y_max, datosCalibracion)
            if centro_3D is not None and len(crop_lidar) > 0:
                idx, etiqueta = asociar_label(idImagen, centro_3D, etiquetas)
                if idx is not None:
                    etiquetas.pop(idx)
                    diccionario[etiqueta["id"]]={
                        "points" : crop_lidar,
                        "label" : etiqueta
                    }
    return diccionario   


from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, os, pickle, cv2
from tqdm import tqdm

def images_processing_parallel(rutaKitti, idSeccion, rutaDiccionario):
    diccionario = {}
    
    ruta = f'{rutaKitti}/velodyne/{idSeccion}'
    archivos = [f for f in os.listdir(ruta) if os.path.isfile(os.path.join(ruta, f))]
    cantidad = len(archivos)
    def worker(i):
        diccionarioThread = {}
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        try:
            idImagen = f"{i:06d}"
            diccionarioThread[idImagen]={}
            ruta_imagen = f'{rutaKitti}/imagenes/{idSeccion}/{idImagen}.png'
            ruta_lidar = f'{rutaKitti}/velodyne/{idSeccion}/{idImagen}.bin'
            ruta_calibracion = f'{rutaKitti}/data_tracking_calib/training/calib/{idSeccion}.txt'
            ruta_label = f'{rutaKitti}/data_tracking_label_2/training/label_02/{idSeccion}.txt'
            imagen = cv2.imread(ruta_imagen)
            detecciones = model.predict(imagen)
            P2, R0_rect, Tr_velo_to_cam = leer_matrices_calibracion(ruta_calibracion)
            datosCalibracion = {
                "P2": P2,
                "R0_rect": R0_rect,
                "Tr_velo_to_cam": Tr_velo_to_cam
            }
            etiquetas = leer_labels_kitti(ruta_label, int(idImagen), Tr_velo_to_cam, R0_rect)
            nubeLidar = leer_puntos_lidar(ruta_lidar)
            if os.path.exists(ruta_label) and os.path.exists(ruta_lidar) and os.path.exists(ruta_calibracion) and imagen is not None:                
                diccionarioImagen = pred_extraction(idImagen, etiquetas, nubeLidar, datosCalibracion, detecciones)
                print(f"hola {i}")
                diccionarioThread[idImagen] = diccionarioImagen
            else:
                print(f"Archivos no encontrados para idImagen {idImagen}. Verifica las rutas.")
            
            return diccionarioThread
        except Exception as e:
            print(f"Error en la iteración {idImagen}: {e}")
            return None
    
     # lanza 20 hilos y reparte frames
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(worker, i) for i in range(8)]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            diccionarioThread = fut.result()
            if diccionarioThread is not None:
                diccionario.update(diccionarioThread)

    with open(rutaDiccionario, 'wb') as file:
        pickle.dump(diccionario, file)
    print(f"Diccionario completo guardado en {rutaDiccionario}")



#### MAIN ####
from viewer3D_v1 import Visualizer3D
import sys

ruta_kitti = 'datos'
directory = "diccionarios/diccionarios_pred_tiempo"
# for num_seccion in range(15,21):
#     idSeccion = f"{num_seccion:04d}"
#     ruta_diccionario = f'{directory}/dicV2_pred{num_seccion}.pkl'
#     print(f"\n-------------- Procesando {idSeccion} -> {ruta_diccionario} --------------")
#     images_processing(ruta_kitti, idSeccion, ruta_diccionario)
num_seccion = int(sys.argv[1])
idSeccion = f"{num_seccion:04d}"
ruta_diccionario = f'{directory}/dicV2_pred{num_seccion}.pkl'
print(f"\n-------------- Procesando {idSeccion} -> {ruta_diccionario} --------------")
images_processing(ruta_kitti, idSeccion, ruta_diccionario)







