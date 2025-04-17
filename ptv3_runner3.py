import torch
import torch.nn as nn
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
from PTv3.model import PointTransformerV3
from typing import List
import math
import warnings



#### dataset

# Cargar el archivo .pkl que contiene el diccionario
archivo_pickle = 'diccionario0000.pkl'

try:
    with open(archivo_pickle, 'rb') as file:
        diccionario = pickle.load(file)
    print("Diccionario cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: El archivo {archivo_pickle} no fue encontrado.")
    diccionario = {}

class_to_idx = {
    'car': 0,
    'pedestrian': 1,
    'van': 2,
    'cyclist': 3,
    'truck': 4
}

def generar_ventanas(objetos_dict, ventana=3):
    """
    Aplica una ventana deslizante sobre las imágenes de cada objeto y obtiene como salida la etiqueta de la última imagen de la ventana.
    """
    ventanas = []

    for objeto_id, datos in objetos_dict.items():
        puntos = datos["points"]
        labels = datos["labels"]
        num_imagenes = len(puntos)

        for i in range(num_imagenes - ventana + 1):
            ventana_actual = puntos[i:i + ventana]
            entrada = ventana_actual  # Lista de nubes de puntos
            salida = labels[i + ventana - 1]  # Último label de la ventana
            ventanas.append((entrada, salida))

    return ventanas

class VentanaDataset(Dataset):
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

        return {"entrada": entrada, "salida": salida}


# Función de collation para batches
def ventana_collate(batch):
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

    return {"entrada": entradas_padded, "salida": salidas}

objetos_dict = {}

for img_id, objects in diccionario.items():
    for obj_id, obj_data in objects.items():
        puntos = obj_data["points"]  # Nube de puntos
        label = obj_data["label"]  # Info del objeto

        # Guardar todos los puntos y labels asociados a este objeto
        if obj_id not in objetos_dict:
            objetos_dict[obj_id] = {"points": [], "labels": []}

        objetos_dict[obj_id]["points"].append(puntos)
        objetos_dict[obj_id]["labels"].append(label)

ventanas_generadas = generar_ventanas(objetos_dict, ventana=3)
ventana_dataset = VentanaDataset(ventanas_generadas)
ventana_dataloader = DataLoader(ventana_dataset, batch_size=4, shuffle=True, collate_fn=ventana_collate)

class PTv3_deteccion(nn.Module):
    def __init__(self, grid_size: tuple):
        super(PTv3_deteccion, self).__init__()
        self.grid_size = grid_size

        self.point_encoder = PointTransformerV3(
            in_channels=1,
            enc_depths=(1, 1, 1, 1, 1),
            enc_num_head=(1, 2, 4, 8, 16),
            enc_patch_size=(64, 64, 64, 64, 64),
            enc_channels=(32, 64, 128, 128, 256),
            dec_depths=(1, 1, 1, 1),
            dec_channels=(128, 64, 64, 64),
            dec_num_head=(4, 4, 4, 8),
            dec_patch_size=(64, 64, 64, 64),
            mlp_ratio=4,
            qkv_bias=True,
        )

        # usamos una MLP para pasar a 8 caracteristicas
        self.feature_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
    
    def forward(self, batch_ventanas: List[torch.Tensor]) -> torch.Tensor:

        offset = torch.tensor([i.shape[0] for i in batch_ventanas], device=device).cumsum(0)
        batch_ventanas = torch.cat(batch_ventanas, 0).to(device)

        points_dict = {
            "feat": torch.ones(batch_ventanas.shape[0], 1,  device=device),
            "coord": batch_ventanas,
            "offset": offset,
            "grid_size": 15.0,
        }

        point_features = self.point_encoder(points_dict)

        print("---finalizada la obtencion de caracteristicas ---")
        """feat_out = self.feature_layer(point_features["feat"])

        offset = [0] + points_dict["offset"].tolist()
        mean_feats = [feat_out[start:end].mean(dim=0, keepdim=True) for start, end in zip(offset[:-1], offset[1:])]
        resultado_final.append(torch.cat(mean_feats, dim=0))"""

        return True #torch.stack(resultado_final)



    
### Uso del modelo ###


model = PTv3_deteccion(grid_size=(30, 30))


# Verificar si CUDA está disponible
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"device= {device}")
model.to(device)

# Inferencia #
print("\n -----> Inferencia <-----\n")
model.eval()

# Entrenamiento #
#print("\n -----> Entrenamiento <-----\n")
#model.train()

for batch in ventana_dataloader:
    entrada = batch["entrada"][0]
    salida = batch["salida"][0]
    #print(entrada)
    #print(salida)
    entrada = list(torch.unbind(entrada, dim=0))

    s = model(entrada)
        
    """for entrada in batch["entrada"]:
        print(entrada)
        print("Tipo de 'entrada':", type(entrada)) 
        print("Longitud de entradas:", len(entrada))
        # Pasamos las entradas al modelo
        #with torch.no_grad(): 
        salida = model(entrada, batch["salida"][0])  # Con esto llamamos al forward

        # Mostrar la salida del modelo
        print("Salida:", salida)
        break  # Solo la primera iteracion"""
    break
