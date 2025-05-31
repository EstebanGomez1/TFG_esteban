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

def leer_labels_kitti(ruta, estructura = 1, idImagen=0):
    """ Lee un archivo KITTI y extrae las líneas y sus posiciones 3D (tx, ty, tz). """
    etiquetas = []
    with open(ruta, 'r') as file:
        for linea in file:
            datos = linea.split()
            if datos[0] != 'DontCare':  # Ignorar DontCare
                if estructura ==1:
                    etiqueta = {
                        'linea': linea.strip(),
                        'clase': datos[2].lower(),  # Tipo de objeto (ej. 'car', 'pedestrian', etc.)
                        'tx': float(datos[11]),  # Coordenada X del centro del objeto
                        'ty': float(datos[12]),  # Coordenada Y del centro del objeto
                        'tz': float(datos[13]),  # Coordenada Z del centro del objeto
                        'height': float(datos[8]),  # Altura del objeto en metros
                        'width': float(datos[9]),  # Ancho del objeto en metros
                        'length': float(datos[10]),  # Longitud del objeto en metros
                        'rot_y': float(datos[14])  # Rotación en Y del objeto (en radianes)
                    }
                    etiquetas.append(etiqueta)
                elif estructura ==2 and int(datos[0]) == idImagen:
                    etiqueta = {
                        #'linea': linea.strip(),
                        'id': datos[1].lower(),
                        'clase': datos[2].lower(),  # Tipo de objeto (ej. 'car', 'pedestrian', etc.)
                        'tx': float(datos[13]),  # Coordenada X del centro del objeto
                        'ty': float(datos[14]),  # Coordenada Y del centro del objeto
                        'tz': float(datos[15]),  # Coordenada Z del centro del objeto
                        'height': float(datos[10]),  # Altura del objeto en metros
                        'width': float(datos[11]),  # Ancho del objeto en metros
                        'length': float(datos[12]),  # Longitud del objeto en metros
                        'rot_y': float(datos[16])  # Rotación en Y del objeto (en radianes)
                    }
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
            centro_3d = np.mean(puntos_en_bounding_box, axis=0)
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
    distancias = np.linalg.norm(puntos - centro, axis=1)
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


def asociar_estimaciones_con_labels(x, y, z, etiquetas, contador, diccionario, idImagen, puntos_recortados ):
    """
    Asocia las estimaciones (x, y, z) con las líneas del archivo KITTI comparando posiciones.
    Imprime las asociaciones id= id con línea correspondiente y guarda los resultados en un diccionario.

    Parametros:
    - x, y, z: Coordenadas de la estimación a asociar (pueden ser escalares o listas).
    - etiquetas: Lista de etiquetas leídas del archivo KITTI.
    - clase: El nombre de la clase (ej. 'Car', 'Pedestrian', etc.).
    - diccionario: Diccionario donde se guardarán las asociaciones.
    - idImagen: Identificador de la imagen (ej. '000013').
    """
    # Asegurarse de que x, y, z sean listas o arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    usadas = set()  # Para no repetir asociaciones

    for idx, (coord_x, coord_y, coord_z) in enumerate(zip(x, y, z)):
        mejor_match = None
        menor_distancia = float('inf')
        for i, etiqueta in enumerate(etiquetas):
            if i in usadas:
                continue
            # Calcular distancia euclidiana entre estimaciones y etiquetas
            distancia = np.sqrt((coord_x - etiqueta['tz'])**2 +
                                (coord_y + etiqueta['tx'])**2 + # Y con signo invertido
                                (coord_z - etiqueta['ty'])**2)  
            if distancia < menor_distancia:
                mejor_match = (i, etiqueta)
                menor_distancia = distancia

        if mejor_match:
            #verificar que la asociacion no difiera mas de 10 metros euclidianos
            if menor_distancia > 10:
                print("No se ha encontrado asociacion posible para este objeto")
            else:
            
                usadas.add(mejor_match[0])
                etiqueta = mejor_match[1]
                
                # Crear el punto para almacenar en el diccionario
                puntos = np.array([coord_x, coord_y, coord_z])
                
                # Formato esperado para la etiqueta 
                etiqueta_label = {
                    "class": etiqueta['clase'],  # Tipo de objeto (ej. 'Car')
                    "x": etiqueta['tx'],  # Coordenada X del centro del objeto
                    "y": etiqueta['ty'],  # Coordenada Y del centro del objeto
                    "z": etiqueta['tz'],  # Coordenada Z del centro del objeto
                    "length": etiqueta['length'],  # Longitud del objeto en metros
                    "height": etiqueta['height'],  # Altura del objeto en metros
                    "width": etiqueta['width'],  # Ancho del objeto en metros
                    "rot_y": etiqueta['rot_y']  # Rotación del objeto en Y
                }
                
                # Guardar en el diccionario
                if idImagen not in diccionario:
                    diccionario[idImagen] = {}

                if contador not in diccionario[idImagen]:
                    diccionario[idImagen][contador] = {
                        "points": [],
                        "label": {}
                    }

                diccionario[idImagen][contador]["points"] = puntos_recortados
                diccionario[idImagen][contador]["label"] = etiqueta_label
                etiqueta_usada = etiqueta
                # Imprimir la asociación
                #print(f"id={idx} con línea: {etiqueta['linea']}"
                #etiquetas.remove(etiqueta)

def asociar_estimaciones_con_labels2(x, y, z, etiquetas, contador, diccionario, idImagen, puntos_recortados ):
    """
    Asocia las estimaciones (x, y, z) con las líneas del archivo KITTI comparando posiciones.
    Imprime las asociaciones id= id con línea correspondiente y guarda los resultados en un diccionario.

    Parametros:
    - x, y, z: Coordenadas de la estimación a asociar (pueden ser escalares o listas).
    - etiquetas: Lista de etiquetas leídas del archivo KITTI.
    - clase: El nombre de la clase (ej. 'Car', 'Pedestrian', etc.).
    - diccionario: Diccionario donde se guardarán las asociaciones.
    - idImagen: Identificador de la imagen (ej. '000013').
    """
    # Asegurarse de que x, y, z sean listas o arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    usadas = set()  # Para no repetir asociaciones
    identificador = None
    for idx, (coord_x, coord_y, coord_z) in enumerate(zip(x, y, z)):
        mejor_match = None
        menor_distancia = float('inf')
        for i, etiqueta in enumerate(etiquetas):
            if i in usadas:
                continue
            # Calcular distancia euclidiana entre estimaciones y etiquetas
            distancia = np.sqrt((coord_x - etiqueta['tz'])**2 +
                                (coord_y + etiqueta['tx'])**2 + # Y con signo invertido
                                (coord_z - etiqueta['ty'])**2)  
            if distancia < menor_distancia:
                mejor_match = (i, etiqueta)
                menor_distancia = distancia

        if mejor_match:
            #verificar que la asociacion no difiera mas de 10 metros 
            if menor_distancia > 6:
                print("No se ha encontrado asociacion posible para este objeto")
            else:
            
                usadas.add(mejor_match[0])
                etiqueta = mejor_match[1]
                identificador = etiqueta["id"]
                # Crear el punto para almacenar en el diccionario
                puntos = np.array([coord_x, coord_y, coord_z])
                
                # Formato esperado para la etiqueta 
                etiqueta_label = {
                    "id": identificador, # Id del objeto
                    "imagen": idImagen, # Id de la imagen
                    "class": etiqueta['clase'],  # Tipo de objeto (ej. 'Car')
                    "x": etiqueta['tx'],  # Coordenada X del centro del objeto
                    "y": etiqueta['ty'],  # Coordenada Y del centro del objeto
                    "z": etiqueta['tz'],  # Coordenada Z del centro del objeto
                    "length": etiqueta['length'],  # Longitud del objeto en metros
                    "height": etiqueta['height'],  # Altura del objeto en metros
                    "width": etiqueta['width'],  # Ancho del objeto en metros
                    "rot_y": etiqueta['rot_y']  # Rotación del objeto en Y
                }
                # Guardar en el diccionario
                if idImagen not in diccionario:
                    diccionario[idImagen] = {}

                if contador not in diccionario[idImagen]:
                    diccionario[idImagen][contador] = {
                        "points": [],
                        "label": {}
                    }

                diccionario[idImagen][contador]["points"] = puntos_recortados
                diccionario[idImagen][contador]["label"] = etiqueta_label
                etiqueta_usada = etiqueta
                # Imprimir la asociación
                #print(f"id={idx} con línea: {etiqueta['linea']}"
                #etiquetas.remove(etiqueta)
    return identificador

def eliminar_asociaciones_duplicadas(diccionario, diccionario_centros, idImagen):
    # Lista de objetos a eliminar (duplicados)
    eliminados = []

    # Recorrer los objetos del diccionario para buscar duplicados
    for clase_id, datos in diccionario[idImagen].items():
        # Obtener las coordenadas del centro del objeto (x, y, z) del diccionario
        x_obj, y_obj, z_obj = datos['label']['x'], datos['label']['y'], datos['label']['z']
        
        centro_obj = diccionario_centros.get(int(clase_id))
        distancia = ((centro_obj['x'] - z_obj)**2 + (centro_obj['y'] - (-x_obj))**2 + (-centro_obj['z'] - y_obj)**2)**0.5
        # si la distancia es muy grande descartamos
        #if distancia > 5: eliminados.append(clase_id)

        # Recorrer el diccionario otra vez para encontrar duplicados
        for clase_id2, datos2 in diccionario[idImagen].items():
            # Comprobar que no estamos comparando el mismo objeto

            if int(clase_id) != int(clase_id2):
  
                # Obtener las coordenadas del centro del objeto (x, y, z) del segundo objeto
                x_obj2, y_obj2, z_obj2 = datos2['label']['x'], datos2['label']['y'], datos2['label']['z']
                

                # Comparar si las coordenadas son exactamente iguales (==)
                if x_obj == x_obj2 and y_obj == y_obj2 and z_obj == z_obj2:
                    
                    centro_obj2 = diccionario_centros.get(clase_id2)

                    if centro_obj is None or centro_obj2 is None:
                        continue  # Si alguno de los centros no existe, no podemos hacer el cálculo


                    distancia2 = ((centro_obj2['x'] - z_obj2)**2 + (centro_obj2['y'] - (-x_obj2))**2 + (-centro_obj2['z'] - y_obj2)**2)**0.5


                    
                    if distancia > distancia2:
                        #candidatos[clase_id][clase_id]= distancia
                        distancia = distancia2
                        eliminados.append(clase_id)
                    else:
                        #candidatos[clase_id][clase_id2] = distancia2
                        eliminados.append(clase_id2)


    #print("==================")
    #print(diccionario)
    # Eliminar los duplicados después de terminar la comparación
    for clase_id in eliminados:
        if clase_id in diccionario[idImagen]:
            del diccionario[idImagen][clase_id]
    #print("===================")
    #print(diccionario)
    return diccionario




    
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

def inferencia(imagen, idImagen, ruta_label, ruta_lidar, diccionario, ruta_calibracion, guardarImagenes = False, info=0, estrcuturaLabel =1):
    
    # Cargar el modelo YOLOv8
    model = YOLO('yolov8n.pt')

    todos_los_puntos = []
    etiquetas = leer_labels_kitti(ruta_label, estrcuturaLabel, int(idImagen))
    diccionario_centros= {}
    # Verificar si la imagen se cargó correctamente
    if imagen is None:
        print("Error: no se pudo cargar la imagen. Verifica la ruta del archivo.")
    else:
        print(f"\n Procesando imagen: {idImagen}")
        # Detectar objetos en la imagen
        results = model.predict(imagen)

        # Leer los puntos LiDAR del archivo
        puntos_lidar = leer_puntos_lidar(ruta_lidar)

        

        # Leer las matrices de calibración
        P2, R0_rect, Tr_velo_to_cam = leer_matrices_calibracion(ruta_calibracion)

        # Procesar cada bounding box detectado por YOLO
        contador = 0
        for r in results:
            for box in r.boxes:
                # Obtener la clase y coordenadas del bounding box
                clase = r.names[int(box.cls.item())]
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()

                # Estimar el centro 3D del bounding box usando los puntos LiDAR
     
                centro_3d, puntos = encontrar_centro_bounding_box_y_nube(
                    puntos_lidar, [x_min, y_min, x_max, y_max], P2, R0_rect, Tr_velo_to_cam
                )
                if info ==1: print(f"clase= {clase}, id= {contador}, X= {centro_3d[0]}, Y= {centro_3d[1]}, Z={centro_3d[2]} ")

                if centro_3d is None or puntos is None:
                    if info ==1 : print(f"No se encontraron puntos para el bounding box de la clase {clase}. Saltando a la siguiente iteración.")
                    continue

                if centro_3d is not None:
                    # Proyectar el centro 3D a la imagen 2D
                    u, v = proyectar_punto_3d_a_2d(centro_3d, P2, R0_rect, Tr_velo_to_cam)
                    
                    # Dibujar el bounding box y el centro proyectado en la imagen
                    cv2.rectangle(imagen, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    cv2.circle(imagen, (int(u), int(v)), 5, (255, 0, 0), -1)
                    distancia = np.linalg.norm(centro_3d)
                    #cv2.putText(imagen, f"{clase} Dist: {distancia:.2f}m id: {contador}", (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(imagen, f" id: {contador}", (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Si hay puntos asociados, aplicar clustering y filtrado
                if centro_3d is not None and len(puntos) > 0:
                    # Aplicar DBSCAN para detectar el cluster más cercano al centro del bounding box
                    puntos_cluster_cercano = cluster_mas_cercano(
                        puntos, centro_bb=(centro_3d[0], centro_3d[1]), eps=0.6, min_samples=5
                    )
                    #puntos_cluster_cercano=puntos
                    if len(puntos_cluster_cercano) > 0:
                        puntos_recortados = puntos_cluster_cercano
                        if len(puntos_recortados) > 0:
                            # Agregar los puntos recortados del objeto actual a la lista global
                            todos_los_puntos.append(puntos_recortados)
                    
                    # Visualizar los puntos filtrados y recortados
                    if info ==1: visualizar_puntos_3d(puntos_recortados, titulo=f"Clase: {clase} (id: {contador})")
                    
                    # Asociar las estimaciones con las etiquetas
                    asociar_estimaciones_con_labels(centro_3d[0], centro_3d[1], centro_3d[2], etiquetas, contador, diccionario, idImagen, puntos_recortados)
                    diccionario_centros[contador] = {
                        'x':float(centro_3d[0]), 'y':float(centro_3d[1]), 'z':float(centro_3d[2])
                    }
                    #print(diccionario[idImagen][contador]['label'])
                contador += 1

        # Guardar el diccionario actualizado en el archivo
        #with open(ruta_diccionario, 'wb') as file:
        #    pickle.dump(diccionario, file)

        # Mostrar la imagen con los bounding boxes y los centros proyectados
        if (info == 1 or info) == 2: mostrar_imagen(imagen)
        if guardarImagenes: guardar_imagenes_procesadas(imagen,idImagen)
    # Combinar todos los puntos en un único array
    todos_los_puntos = np.vstack(todos_los_puntos) if len(todos_los_puntos) > 0 else np.array([])

    # Visualizar todos los puntos en un único espacio 3D
    if info ==1: visualizar_puntos_3d(todos_los_puntos, titulo="Todos los Objetos Detectados en el Espacio 3D")
    if info ==1: visualizar_puntos_filtrados(todos_los_puntos)
    #print(diccionario_centros)
    #diccionario = eliminar_asociaciones_duplicadas(diccionario, diccionario_centros, idImagen)
    return diccionario

def inferencia2(imagen, idImagen, ruta_label, ruta_lidar, diccionario, ruta_calibracion, guardarImagenes = False, info=0, estrcuturaLabel =1):
    
    # Cargar el modelo YOLOv8
    model = YOLO('yolov8n.pt')

    todos_los_puntos = []
    etiquetas = leer_labels_kitti(ruta_label, estrcuturaLabel, int(idImagen))
    diccionario_centros= {}
    datos_imagen = {}
    # Verificar si la imagen se cargó correctamente
    if imagen is None:
        print("Error: no se pudo cargar la imagen. Verifica la ruta del archivo.")
    else:
        print(f"\n Procesando imagen: {idImagen}")
        # Detectar objetos en la imagen
        results = model.predict(imagen)

        # Leer los puntos LiDAR del archivo
        puntos_lidar = leer_puntos_lidar(ruta_lidar)

        

        # Leer las matrices de calibración
        P2, R0_rect, Tr_velo_to_cam = leer_matrices_calibracion(ruta_calibracion)

        # Procesar cada bounding box detectado por YOLO
        contador = 0
        for r in results:
            for box in r.boxes:
                # Obtener la clase y coordenadas del bounding box
                clase = r.names[int(box.cls.item())]
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()

                # Estimar el centro 3D del bounding box usando los puntos LiDAR
     
                centro_3d, puntos = encontrar_centro_bounding_box_y_nube(
                    puntos_lidar, [x_min, y_min, x_max, y_max], P2, R0_rect, Tr_velo_to_cam
                )
                if info ==1: print(f"clase= {clase}, id= {contador}, X= {centro_3d[0]}, Y= {centro_3d[1]}, Z={centro_3d[2]} ")

                if centro_3d is None or puntos is None:
                    if info ==1 : print(f"No se encontraron puntos para el bounding box de la clase {clase}. Saltando a la siguiente iteración.")
                    continue

                if centro_3d is not None:
                    # Proyectar el centro 3D a la imagen 2D
                    u, v = proyectar_punto_3d_a_2d(centro_3d, P2, R0_rect, Tr_velo_to_cam)
                    
                    # Dibujar el bounding box y el centro proyectado en la imagen
                    #cv2.rectangle(imagen, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    #cv2.circle(imagen, (int(u), int(v)), 5, (255, 0, 0), -1)
                    #distancia = np.linalg.norm(centro_3d)
                    #cv2.putText(imagen, f"{clase} Dist: {distancia:.2f}m id: {contador}", (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    #cv2.putText(imagen, f" id: {contador}", (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    datos_imagen[contador]={}
                    datos_imagen[contador]["x_min"] = x_min
                    datos_imagen[contador]["x_max"] = x_max
                    datos_imagen[contador]["y_min"] = y_min
                    datos_imagen[contador]["y_max"] = y_max
                    datos_imagen[contador]["u"] = u
                    datos_imagen[contador]["v"] = v
                # Si hay puntos asociados, aplicar clustering y filtrado
                if centro_3d is not None and len(puntos) > 0:
                    # Aplicar DBSCAN para detectar el cluster más cercano al centro del bounding box
                    puntos_cluster_cercano = cluster_mas_cercano(
                        puntos, centro_bb=(centro_3d[0], centro_3d[1]), eps=0.6, min_samples=5
                    )
                    #puntos_cluster_cercano=puntos
                    if len(puntos_cluster_cercano) > 0:
                        puntos_recortados = puntos_cluster_cercano
                        if len(puntos_recortados) > 0:
                            # Agregar los puntos recortados del objeto actual a la lista global
                            todos_los_puntos.append(puntos_recortados)
                    
                        # Visualizar los puntos filtrados y recortados
                        if info ==1: visualizar_puntos_3d(puntos_recortados, titulo=f"Clase: {clase} (id: {contador})")
                        
                        # Asociar las estimaciones con las etiquetas
                        identifiador = asociar_estimaciones_con_labels2(centro_3d[0], centro_3d[1], centro_3d[2], etiquetas, contador, diccionario, idImagen, puntos_recortados)
                        datos_imagen[contador]["id"] = identifiador
                        diccionario_centros[contador] = {
                            'x':float(centro_3d[0]), 'y':float(centro_3d[1]), 'z':float(centro_3d[2])
                        }
                        #print(diccionario[idImagen][contador]['label'])
                contador += 1

        # Guardar el diccionario actualizado en el archivo
        #with open(ruta_diccionario, 'wb') as file:
        #    pickle.dump(diccionario, file)

        # Mostrar la imagen con los bounding boxes y los centros proyectados
        if (info == 1 or info) == 2: mostrar_imagen(imagen)
        
    # Combinar todos los puntos en un único array
    todos_los_puntos = np.vstack(todos_los_puntos) if len(todos_los_puntos) > 0 else np.array([])

    # Visualizar todos los puntos en un único espacio 3D
    if info ==1: visualizar_puntos_3d(todos_los_puntos, titulo="Todos los Objetos Detectados en el Espacio 3D")
    if info ==1: visualizar_puntos_filtrados(todos_los_puntos)
    #print(diccionario_centros)
    #print(diccionario)
    diccionario = eliminar_asociaciones_duplicadas(diccionario, diccionario_centros, idImagen)
    # pintar los id en la imagen
    # for elemento in diccionario[idImagen]:
    #     for elem in datos_imagen:
    #         if elemento == elem:
    #             cv2.putText(imagen, f" id: {datos_imagen[elem]["id"]}", (int(datos_imagen[elem]["x_min"]), int(datos_imagen[elem]["y_min"] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #             diccionario[idImagen][datos_imagen[elem]["id"]] = diccionario[idImagen].pop(elemento)
    # Crear un diccionario auxiliar para almacenar los cambios
    cambios = {}
    #print(datos_imagen)
    for elemento in list(diccionario[idImagen]):  # Usar list() para evitar modificar el diccionario durante la iteración
        #print(f"elemento= {elemento}")
        for elem in datos_imagen:
            #print(f"toca= {elem}")
            if int(elemento) == int(elem):
                #print(f"elem= {elem} y elemento = {elemento}")
                cv2.putText(
                    imagen,
                    f" id: {datos_imagen[elem]['id']}",
                    (int(datos_imagen[elem]["x_min"]), int(datos_imagen[elem]["y_min"] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                # Dibujar el bounding box y el centro proyectado en la imagen
                cv2.rectangle(imagen, (int(datos_imagen[elem]["x_min"]), int(datos_imagen[elem]["y_min"])), (int(datos_imagen[elem]["x_max"]), int(datos_imagen[elem]["y_max"])), (0, 255, 0), 2)
                cv2.circle(imagen, (int(datos_imagen[elem]["u"]), int(datos_imagen[elem]["v"])), 5, (255, 0, 0), -1)
                cambios[datos_imagen[elem]["id"]] = diccionario[idImagen][elem]

    # Aplicar los cambios después de la iteración
    diccionario[idImagen].clear()
    diccionario[idImagen].update(cambios)
    #print(diccionario)
    if guardarImagenes: guardar_imagenes_procesadas(imagen,idImagen)
    return diccionario







