import torch
import torch.nn as nn
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
#from PTv3.model import PointTransformerV3
from typing import List
import math
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
from pprint import pprint
from torch.utils.data import ConcatDataset, DataLoader

def load_dictionary(file_pkl):
    try:
        with open(file_pkl, 'rb') as file:
            diccionario = pickle.load(file)
        print(f"Diccionario {file_pkl} cargado exitosamente.")
    except FileNotFoundError:
        print(f"Error: El archivo {file_pkl} no fue encontrado.")
        diccionario = {}
    return diccionario

class_to_idx = {
    'car': 0,
    'pedestrian': 1,
    'van': 2,
    'cyclist': 3,
    'truck': 4,
    'person': 5,
    'tram': 6,
    'misc': 7
}

def parse_label(label_dict):
    import math  # por si aún no estaba importado

    # Obtener rot_y original (en coordenadas KITTI)
    rot_y_kitti = label_dict.get("rot_y", 0.0)
    # Convertir a yaw en coordenadas LiDAR
    yaw_lidar = ((-rot_y_kitti - math.pi / 2 + math.pi) % (2 * math.pi)) - math.pi
    #yolo_center = label_dict.get("yolo_center", [0.0, 0.0, 0.0])
    if "yolo_center" in label_dict:
        yolo_center = label_dict["yolo_center"]
    else:
        # Usa las keys x, y, z del label
        yolo_center = [
            label_dict.get("x", 0.0),
            label_dict.get("y", 0.0),
            label_dict.get("z", 0.0),
        ]
    label_tensor = torch.tensor([
        class_to_idx.get(label_dict.get("clase", "car"), -1),
        label_dict.get("x", 0.0),
        label_dict.get("y", 0.0),
        label_dict.get("z", 0.0)*0.5,
        label_dict.get("width", 0.0),
        label_dict.get("length", 0.0),
        label_dict.get("height", 0.0),
        yaw_lidar,
        *yolo_center
    ], dtype=torch.float32)

    return label_tensor

import json
import pprint

class WindowedDataset(Dataset):
    def __init__(self, objetos_dict, ventana=3):
        self.ventana = ventana
        self.inputs = []
        self.outputs = []
        skip_classes = {4, 6, 7}
        for objeto_id in sorted(objetos_dict.keys()):
            imagenes = objetos_dict[objeto_id]
            img_ids = sorted(imagenes.keys())
            puntos = [imagenes[img_id]["points"] for img_id in img_ids]
            labels = [imagenes[img_id]["labels"] for img_id in img_ids]
            num_imagenes = len(puntos)

            for i in range(num_imagenes - ventana + 1):
                ventana_actual = puntos[i:i + ventana]
                labels_ventana = labels[i:i + ventana]
                salida = labels[i + ventana - 1]
                clase_idx = class_to_idx.get(salida.get("class", "car"), -1)
                if clase_idx in skip_classes:
                    continue

                tensor_ventana = []
                target_box = parse_label(salida)  # última label, destino de alineación
                center_ref = target_box[1:4]  # x, y, z

                for j in range(ventana):
                    nube = torch.tensor(ventana_actual[j])
                    label_j = parse_label(labels_ventana[j])
                    center_j = label_j[1:4]
                    delta = center_ref - center_j
                    nube[:, :3] += delta  # mover nube al centro de la última label
                    tensor_ventana.append(nube)

                self.inputs.append(tensor_ventana)
                self.outputs.append(target_box)


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]





def custom_collate_yolo_center_fn(batch):
    entradas = []
    salidas = []
    centros = []

    for item in batch:
        crop_tensors = []
        label_tensor = item[1]

        # Extraer centro yolo del label: asumimos que los últimos 3 valores son x,y,z centro
        centro_yolo = label_tensor[-3:]
        label_sin_centro = label_tensor[:-3]
        label_sin_centro[1:4] -= centro_yolo  
        for tensor in item[0]:  # Lista de puntos del crop
            puntos = tensor.clone()
            puntos[:, :3] -= centro_yolo  # Normaliza restando el centro estimado
            crop_tensors.append(puntos)

        entradas.extend(crop_tensors)
        salidas.append(label_sin_centro)
        centros.append(centro_yolo)

    return entradas, salidas, centros

def custom_collate_yolo_center_fn_smooth(batch):
    entradas = []
    salidas = []
    centros = []
    saltos = []
    for item in batch:
        crop_tensors = []
        label_tensor = item[1]
        centro_yolo = label_tensor[-3:]
        # Calcular distancia entre el centro original del label y centro del crop
        centro_label = label_tensor[1:4]  # x, y, z reales
        distancia = torch.norm(centro_label - centro_yolo)
        salto = distancia.item() > 3.0
        saltos.append(salto)

        # Extraer centro yolo del label: asumimos que los últimos 3 valores son x,y,z centro
        #centro_yolo = label_tensor[-3:]
        label_sin_centro = label_tensor[:-3].clone()
        label_sin_centro[1:4] -= centro_yolo  

        for tensor in item[0]:  # Lista de puntos del crop
            puntos = tensor.clone()
            puntos[:, :3] -= centro_yolo  # Normaliza restando el centro estimado
            crop_tensors.append(puntos)

        entradas.extend(crop_tensors)
        salidas.append(label_sin_centro)
        centros.append(centro_yolo)

    return entradas, salidas, centros, saltos

def custom_collate_yolo_center_fn_smooth_sinNorm(batch):
    entradas = []
    salidas = []
    centros = []
    saltos = []
    for item in batch:
        crop_tensors = []
        label_tensor = item[1]
        centro_yolo = label_tensor[-3:]
        # Calcular distancia entre el centro original del label y centro del crop
        centro_label = label_tensor[1:4]  # x, y, z reales
        distancia = torch.norm(centro_label - centro_yolo)
        salto = distancia.item() > 3.0
        salto = False
        saltos.append(salto)

        # Extraer centro yolo del label: asumimos que los últimos 3 valores son x,y,z centro
        #centro_yolo = label_tensor[-3:]
        #label_sin_centro = label_tensor[:-3].clone()
        #label_sin_centro[1:4] -= centro_yolo  

        for tensor in item[0]:  # Lista de puntos del crop
            puntos = tensor.clone()
            #puntos[:, :3] -= centro_yolo  # Normaliza restando el centro estimado
            crop_tensors.append(puntos)

        entradas.extend(crop_tensors)
        salidas.append(label_tensor)
        centros.append([0,0,0])

    return entradas, salidas, centros, saltos

def custom_collate_yolo_center_fn_smooth_offset(batch):
    entradas = []
    salidas = []
    centros = []
    saltos = []
    for item in batch:
        crop_tensors = []
        label_tensor = item[1]
        centro_yolo = label_tensor[-3:]
        # Calcular distancia entre el centro original del label y centro del crop
        centro_label = label_tensor[1:4]  # x, y, z reales
        # offset aleatorio de máximo 0.2m en cada eje (por ejemplo)
        offset = torch.empty(3).uniform_(-0.2, 0.2)
        centro_label = centro_label + offset  # centro desplazado
        distancia = torch.norm(centro_label - centro_yolo)
        salto = distancia.item() > 3.0
        saltos.append(salto)

        # Extraer centro yolo del label: asumimos que los últimos 3 valores son x,y,z centro
        #centro_yolo = label_tensor[-3:]
        label_sin_centro = label_tensor[:-3]
        label_sin_centro[1:4] -= centro_yolo  

        for tensor in item[0]:  # Lista de puntos del crop
            puntos = tensor.clone()
            puntos[:, :3] -= centro_yolo  # Normaliza restando el centro estimado
            crop_tensors.append(puntos)

        entradas.extend(crop_tensors)
        salidas.append(label_sin_centro)
        centros.append(centro_yolo)

    return entradas, salidas, centros, saltos

import random

def rotate_points_z(points, angle_rad):
    """
    Rota una nube de puntos en el plano XY (alrededor de Z).
    """
    cos_val = math.cos(angle_rad)
    sin_val = math.sin(angle_rad)
    R = torch.tensor([
        [cos_val, -sin_val, 0],
        [sin_val,  cos_val, 0],
        [0,        0,       1]
    ], dtype=points.dtype, device=points.device)

    xyz = points[:, :3] @ R.T
    return torch.cat([xyz, points[:, 3:]], dim=1)

def custom_collate_yolo_center_fn_smooth_rot(batch):
    entradas = []
    salidas = []
    centros = []
    saltos = []

    for item in batch:
        crop_tensors = []
        label_tensor = item[1]
        centro_yolo = label_tensor[-3:]

        # --- 1. Rotación aleatoria ---
        #angle = random.uniform(-math.pi, math.pi)  # entre -π y π
        angle = random.uniform(-math.pi / 4, math.pi / 4)  # ±45°


        # --- 2. Calcular distancia real entre centro label y centro yolo
        centro_label = label_tensor[1:4]  # x, y, z reales
        distancia = torch.norm(centro_label - centro_yolo)
        salto = distancia.item() > 3.0
        saltos.append(salto)

        # --- 3. Ajustar label ---
        label_sin_centro = label_tensor[:-3].clone()
        label_sin_centro[1:4] -= centro_yolo  # normalizar coordenadas

        # ROTAR POSICIÓN (X, Y) Y ÁNGULO (YAW)
        xy = label_sin_centro[1:3]
        x_rot =  xy[0] * math.cos(angle) - xy[1] * math.sin(angle)
        y_rot =  xy[0] * math.sin(angle) + xy[1] * math.cos(angle)
        label_sin_centro[1] = x_rot
        label_sin_centro[2] = y_rot
        label_sin_centro[7] += angle  # yaw += rotación aplicada
        label_sin_centro[7] = ((label_sin_centro[7] + math.pi) % (2 * math.pi)) - math.pi  # normaliza a [-π, π]

        # --- 4. Rotar nubes de puntos y normalizar ---
        for tensor in item[0]:  # Lista de puntos del crop
            puntos = tensor.clone()
            puntos[:, :3] -= centro_yolo
            puntos = rotate_points_z(puntos, angle)
            crop_tensors.append(puntos)

        entradas.extend(crop_tensors)
        salidas.append(label_sin_centro)
        centros.append(centro_yolo)

    return entradas, salidas, centros, saltos







def format_dicc(diccionario):
    """
    Cambia el formato del diccionario para tener un diccionario de objetos donde cada objeto 
    tiene un diccionario de imagenes donde aparece con sus nubes y labels asociados
    """
    objetos_dict = {}
    for img_id in diccionario:
        for obj in diccionario[img_id]:
            puntos = diccionario[img_id][obj]["points"]  # Nube de puntos
            label = diccionario[img_id][obj]["label"]    # Info del objeto
            obj_id = label["id"]
            if( obj_id not in objetos_dict):
                objetos_dict[obj_id]={}
            objetos_dict[obj_id][img_id] = {
                "points": puntos,
                "labels": label
            }           
    return objetos_dict

def correr_prueba(ventana_dataloader):
    print("Entrando a correr_prueba...")
    batches_leidos = 0
    for batch in ventana_dataloader:
        print("Batch completo de entradas:")
        for i, entrada in enumerate(batch["entrada"]):
            print(f"\n🔹 Entrada {i}:")
            for j, tensor in enumerate(entrada):
                # Mostrar la forma solo si es tensor/array, si es lista muestra el tamaño
                if hasattr(tensor, "shape"):
                    print(f" Tensor {j}: shape={tensor.shape}\n")
                elif hasattr(tensor, "size"):
                    print(f" Tensor {j}: size={tensor.size()}\n")
                elif isinstance(tensor, list):
                    print(f" Tensor {j}: list len={len(tensor)}\n")
                else:
                    print(f" Tensor {j}: type={type(tensor)} value={tensor}\n")
        print("\nBatch de salidas:")
        print(batch["salida"])
        break  # Solo mostramos un batch para no llenar la consola
    if batches_leidos == 0:
        print("⚠️ No se leyó ningún batch (el DataLoader está vacío o no itera)")

def getDataLoader(diccionario):
    dataset = WindowedDataset(format_dicc(diccionario))
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    




def ver_dicc(diccionario):
    for imagen in diccionario:
        print(f"\n---------------------------------------{imagen}-----------------------------------------------\n")
        pprint(diccionario[imagen])

def ver_format_dicc(diccionario):
    for obj_id in diccionario:
        print(f"\n---------------------------------------{obj_id}-----------------------------------------------\n")
        pprint(diccionario[obj_id])

def ver_dataSet(ventana_dataset):
    for i in range(3):  # Mira las primeras 3 muestras
        ventana, label = ventana_dataset[i]
        print(f"Sample {i}:")
        print(f"  Ventana (lista de {len(ventana)} tensores):")
        for j, t in enumerate(ventana):
            print(f"    Tensor {j}: shape={t.shape}, dtype={t.dtype}")
        print(f"  Label tensor: shape={label.shape}, values={label}")
        print("-"*40)
            
        

def data_compiler(numDiccs=2):
    diccionarios =[]
    #for i in range(0, numDiccs):
    #    #nombre_archivo = f"diccionario{str(i).zfill(4)}.pkl"
    nombre_archivo = 'dicc3.pkl'
    if os.path.exists(nombre_archivo):
        diccionario = load_dictionary(nombre_archivo)
        diccionarios.append(diccionario)
    else:
        print(f"Archivo {nombre_archivo} no encontrado, se salta.")
    return diccionarios

##### Fusionar diccionario #####

def getMultiDataLoader2(diccionarios):
    datasets = []
    for diccionario in diccionarios:
        datos_formateados = format_dicc(diccionario)
        dataset = WindowedDataset(datos_formateados)
        datasets.append(dataset)

    dataset_total = ConcatDataset(datasets)
    return dataset_total #DataLoader(dataset_total, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

def getMultiDataLoader(diccionarios):
    datasets = []
    for diccionario in diccionarios:
        datos_formateados = format_dicc(diccionario)
        dataset = WindowedDataset(datos_formateados)
        datasets.append(dataset)

    dataset_total = ConcatDataset(datasets)
    return DataLoader(dataset_total, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

def load_dicctionaries(name="dic_perf", total=6):
    lista_diccionarios = []
    for i in range(total):  # de 0 a 5
        #dicc_name = f"diccionarios/diccionario{i}.pkl"
        #dicc_name = f"dic_format{i}.pkl"
        dicc_name = f"{name}{i}.pkl"
        dicc = load_dictionary(dicc_name)
        lista_diccionarios.append(dicc)
    return lista_diccionarios
