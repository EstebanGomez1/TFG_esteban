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
    'truck': 4
}

def generar_ventanas2(objetos_dict, ventana=3):
    """
    Aplica una ventana deslizante sobre las imágenes de cada objeto y obtiene como salida la etiqueta de la última imagen de la ventana.
    """
    ventanas = []

    for objeto_id, imagenes in objetos_dict.items():
        img_ids = sorted(imagenes.keys()) 
        puntos = [imagenes[img_id]["points"] for img_id in img_ids]
        labels = [imagenes[img_id]["labels"] for img_id in img_ids]
        num_imagenes = len(puntos)

        for i in range(num_imagenes - ventana + 1):
            ventana_actual = puntos[i:i + ventana]
            salida = labels[i + ventana - 1]  # Último label de la ventana
            ventanas.append((ventana_actual, salida))

    print(ventanas)
    return ventanas

def generar_ventanas3(objetos_dict, ventana=3):
    ventanas_por_objeto = []

    # Ordenar los objetos por su id (importante para consistencia)
    for objeto_id in sorted(objetos_dict.keys()):
        imagenes = objetos_dict[objeto_id]
        # Ordenar las imagenes por su id (importante para consistencia temporal)
        img_ids = sorted(imagenes.keys())
        puntos = [imagenes[img_id]["points"] for img_id in img_ids]
        labels = [imagenes[img_id]["labels"] for img_id in img_ids]
        num_imagenes = len(puntos)
        
        ventanas = []
        for i in range(num_imagenes - ventana + 1):
            ventana_actual = puntos[i:i + ventana]
            salida = labels[i + ventana - 1]
            ventanas.append((ventana_actual, salida))

        ventanas_por_objeto.append(ventanas)
    print(ventanas_por_objeto)
    return ventanas_por_objeto

def generar_ventanas(objetos_dict, ventana=3):
    inputs = []
    outputs = []
    # Ordenar los objetos por su id (importante para consistencia)
    for objeto_id in sorted(objetos_dict.keys()):
        imagenes = objetos_dict[objeto_id]
        # Ordenar las imagenes por su id (importante para consistencia temporal)
        img_ids = sorted(imagenes.keys())
        puntos = [imagenes[img_id]["points"] for img_id in img_ids]
        labels = [imagenes[img_id]["labels"] for img_id in img_ids]
        num_imagenes = len(puntos)
        
        ventanas_objeto = []
        labels_objeto = []
        for i in range(num_imagenes - ventana + 1):
            ventana_actual = puntos[i:i + ventana]   # Lista de nubes de puntos
            salida = labels[i + ventana - 1]         # Etiqueta correspondiente
            ventanas_objeto.append(ventana_actual)
            labels_objeto.append(salida)

        inputs.append(ventanas_objeto)
        outputs.append(labels_objeto)

    dataset = {
        "inputs": inputs,   # List[List[window]]
        "outputs": outputs  # List[List[label]]
    }
    return dataset

"""class VentanaDataset(Dataset):
    def __init__(self, ventanas):
        self.ventanas = ventanas
        

    def __len__(self):
        return len(self.ventanas)

    def __getitem__(self, idx):
        entrada, salida = self.ventanas[idx]

        # Convertir entrada a tensores correctamente
        entrada = [e.clone().detach().float() if isinstance(e, torch.Tensor) else torch.tensor(e, dtype=torch.float32) for e in entrada]

        # Extraer solo los valores relevantes de salida
        if isinstance(salida, dict):
            salida = torch.tensor([
                class_to_idx.get(salida.get("class", 0), -1),
                salida.get("x", 0.0),
                salida.get("y", 0.0),
                salida.get("z", 0.0),
                salida.get("length", 0.0),
                salida.get("height", 0.0),
                salida.get("width", 0.0),
                salida.get("rot_y", 0.0)
            ], dtype=torch.float32)
        else:
            raise ValueError(f"Salida en índice {idx} no es un diccionario. Datos: {self.ventanas[idx]}")

        return {"entrada": entrada, "salida": salida}"""


"""class WindowedDataset(Dataset):
    def __init__(self, data_dict):
        self.data = []

        inputs = data_dict['inputs']
        outputs = data_dict['outputs']

        for i in range(len(inputs)):
            input_seq = inputs[i]
            output_seq = outputs[i]
            self.data.append((input_seq, output_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, output_seq = self.data[idx]

        # Convertir a tensores
        points = torch.tensor(points, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        return points, labels"""

class WindowedDataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = data_dict['inputs']
        self.outputs = data_dict['outputs']

        all_tokens = set()

        for window in self.inputs:
            for seq in window:
                if isinstance(seq, np.ndarray):
                    seq = seq.tolist()

                for token in seq:
                    # Si token es array, convertirlo a lista y tomar el primer elemento
                    if isinstance(token, np.ndarray):
                        token = token.tolist()

                        # Si el resultado es una lista, tomar el primer string real
                        if isinstance(token, list):
                            token = token[0]

                    all_tokens.add(str(token))  # convertir todo a string por seguridad

        for label_seq in self.outputs:
            if isinstance(label_seq, np.ndarray):
                label_seq = label_seq.tolist()

            for token in label_seq:
                if isinstance(token, np.ndarray):
                    token = token.tolist()
                    if isinstance(token, list):
                        token = token[0]
                all_tokens.add(str(token))

        self.vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_window = self.inputs[idx]
        output_seq = self.outputs[idx]

        input_ids = []
        for sublist in input_window:
            row = []
            for token in sublist:
                if isinstance(token, np.ndarray):
                    token = token.tolist()
                    if isinstance(token, list):
                        token = token[0]
                token_str = str(token)
                row.append(self.vocab[token_str])
            input_ids.append(row)

        output_ids = []
        for token in output_seq:
            if isinstance(token, np.ndarray):
                token = token.tolist()
                if isinstance(token, list):
                    token = token[0]
            token_str = str(token)
            output_ids.append(self.vocab[token_str])

        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        output_tensor = torch.tensor(output_ids, dtype=torch.long)

        return input_tensor, output_tensor


# Función de collation para batches
"""def ventana_collate(batch):
    entradas = [item["entrada"] for item in batch]  # Lista de listas de tensores
    salidas = torch.stack([item["salida"] for item in batch])  # Labels en batch

    # Encontrar la longitud máxima de las nubes de puntos en todas las entradas del batch
    max_len = max(len(seq) for entrada in entradas for seq in entrada)  # La longitud máxima de la secuencia dentro de cada entrada

    # Rellenar las secuencias con ceros para que tengan la misma longitud
    entradas_padded = [
        [torch.cat([seq, torch.ones(max_len - len(seq), seq.size(1))], dim=0) if len(seq) < max_len else seq for seq in entrada]
        for entrada in entradas
    ]

    # Apilar las secuencias rellenas
    entradas_padded = [torch.stack(entrada) for entrada in entradas_padded]

    return {"entrada": entradas_padded, "salida": salidas}"""
def ventana_collate(batch):
    # Filtrar entradas con salidas vacías
    batch = [item for item in batch if item[1].numel() > 0]
    if len(batch) == 0:
        raise ValueError("Batch vacío: todas las salidas están vacías.")

    entradas = [item[0] for item in batch]  # Lista de [W, L_i, C]
    salidas = [item[1] for item in batch]   # Lista de [W] (o variable)

    # Paso 1: calcular longitudes máximas
    max_ventanas = max(entrada.size(0) for entrada in entradas)       # W
    max_puntos = max(entrada.size(1) for entrada in entradas)         # L
    num_canales = entradas[0].size(2)                                 # C

    # Paso 2: rellenar con padding cada entrada → [W, L, C]
    entradas_padded = []
    for entrada in entradas:
        ventanas = []
        for ventana in entrada:
            if ventana.size(0) < max_puntos:
                padding = torch.zeros(max_puntos - ventana.size(0), num_canales)
                ventana = torch.cat([ventana, padding], dim=0)
            ventanas.append(ventana)
        if len(ventanas) < max_ventanas:
            # Rellenar con ventanas vacías si faltan
            dummy_ventana = torch.zeros(max_puntos, num_canales)
            ventanas += [dummy_ventana] * (max_ventanas - len(ventanas))
        entrada_tensor = torch.stack(ventanas)  # [W, L, C]
        entradas_padded.append(entrada_tensor)

    entradas_batch = torch.stack(entradas_padded)  # [B, W, L, C]

    # Paso 3: pad también salidas si varían en longitud
    max_len_salidas = max(salida.size(0) for salida in salidas)
    salidas_padded = [
        torch.cat([salida, torch.full((max_len_salidas - salida.size(0),), 0, dtype=torch.long)])
        if salida.size(0) < max_len_salidas else salida
        for salida in salidas
    ]
    salidas_batch = torch.stack(salidas_padded)

    return {"entrada": entradas_batch, "salida": salidas_batch}


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
    ventanas_generadas = generar_ventanas(format_dicc(diccionario), ventana=3)
    ventana_dataset = WindowedDataset(ventanas_generadas)
    return DataLoader(ventana_dataset, batch_size=4, shuffle=False, collate_fn=ventana_collate)
    
### Uso del modelo ###

# diccionario = load_dictionary("diccionario0000.pkl")
# ventanas_generadas = generar_ventanas(format_dicc(diccionario), ventana=3)
# ventana_dataset = VentanaDataset(ventanas_generadas)
# ventana_dataloader = DataLoader(ventana_dataset, batch_size=4, shuffle=False, collate_fn=ventana_collate)


def ver_dicc(diccionario):
    for imagen in diccionario:
        print(f"\n---------------------------------------{imagen}-----------------------------------------------\n")
        pprint(diccionario[imagen])

def ver_format_dicc(diccionario):
    for obj_id in diccionario:
        print(f"\n---------------------------------------{obj_id}-----------------------------------------------\n")
        pprint(diccionario[obj_id])
            
        

def data_compiler(numDiccs=2):
    diccionarios =[]
    #for i in range(0, numDiccs):
    #    #nombre_archivo = f"diccionario{str(i).zfill(4)}.pkl"
    nombre_archivo = 'dicc1.pkl'
    if os.path.exists(nombre_archivo):
        diccionario = load_dictionary(nombre_archivo)
        diccionarios.append(diccionario)
    else:
        print(f"Archivo {nombre_archivo} no encontrado, se salta.")
    return diccionarios


# Ejemplo de diccionario
diccionario_prueba = {
    "1": {
        "img_01": {"points": "nube1", "labels": "etiquetas1-1"},
        "img_02": {"points": "nube2", "labels": "etiquetas2-1"},
        "img_03": {"points": "nube3", "labels": "etiquetas3-1"},
        "img_04": {"points": "nube4", "labels": "etiquetas4-1"},
        "img_05": {"points": "nube5", "labels": "etiquetas5-1"},
    },
    "2": {
        "img_01": {"points": "nube1", "labels": "etiquetas1-2"},
        "img_02": {"points": "nube2", "labels": "etiquetas2-2"},
        "img_03": {"points": "nube3", "labels": "etiquetas3-2"},
        "img_04": {"points": "nube4", "labels": "etiquetas4-2"},
        "img_05": {"points": "nube5", "labels": "etiquetas5-2"},
    }
}



print("-------------------")
#datal = getDataLoader(data_compiler()[0])
#print("Longitud del dataset:", len(datal.dataset))
#correr_prueba(datal)

diccNormal = data_compiler()[0]
#ver_dicc(dicc)
#dicc_format = format_dicc(dicc)
#ver_format_dicc(dicc_format)
#pprint(dicc_format)

"""dicc = diccionario_prueba
dicc = format_dicc(diccNormal)

ventanas_generadas = generar_ventanas(dicc, ventana=3)
print(f"long outs: {len(ventanas_generadas['outputs'])}")
print(ventanas_generadas['outputs'][1][0])
print(len(ventanas_generadas['inputs'][1][0]))"""

ventanas_generadas = generar_ventanas(format_dicc(diccNormal), ventana=3)
ventana_dataset = WindowedDataset(ventanas_generadas)
dataloader = DataLoader(ventana_dataset, batch_size=4, shuffle=False, collate_fn=ventana_collate)
for batch_inputs, batch_labels in dataloader:
    print(batch_inputs, batch_labels)

print("-------------------")