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
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time


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


def vecinos_proximos_agregar(tensor, k=3):
    # Calcular el centroide (media de los vectores)
    centroide = tensor.mean(dim=0, keepdim=True)

    # Calcular la distancia euclidiana entre cada vector y el centroide
    distancias = F.pairwise_distance(tensor, centroide)

    # Seleccionar los índices de los k vecinos más cercanos
    indices_k_cercanos = torch.topk(distancias, k=k, largest=False).indices

    # Obtener los k vecinos más cercanos
    vecinos_cercanos = tensor[indices_k_cercanos]

    # Agregar los vecinos más cercanos
    vector_agregado = vecinos_cercanos.mean(dim=0)

    return vector_agregado


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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
    
    def forward(self, batch_ventanas: List[torch.Tensor]) -> torch.Tensor:

        offset = torch.tensor([i.shape[0] for i in batch_ventanas], device=device).cumsum(0)
        batch_ventanas = torch.cat(batch_ventanas, 0).to(device)

        points_dict = {
            "feat": batch_ventanas[:,3:],
            "coord": batch_ventanas[:,:3],
            "offset": offset,
            "grid_size": 5.0,
        }

        point_features = self.point_encoder(points_dict)

        #print("---finalizada la obtencion de caracteristicas ---")
        #print(point_features)
        feat_out = self.feature_layer(point_features["feat"])

        """resultado_final = [];
        offset = [0] + points_dict["offset"].tolist()
        mean_feats = [feat_out[start:end].mean(dim=0, keepdim=True) for start, end in zip(offset[:-1], offset[1:])]
        resultado_final.append(torch.cat(mean_feats, dim=0))"""

        return feat_out #torch.stack(resultado_final)


# Modelos de regresión
class RegresionLineal(nn.Module):
    def __init__(self, input_dim):
        super(RegresionLineal, self).__init__()
        self.linear = nn.Linear(input_dim, 7)  # Regresión para 7 valor

    def forward(self, x):
        return self.linear(x)

class RegresionCiclica(nn.Module):
    def __init__(self, input_dim):
        super(RegresionCiclica, self).__init__()
        self.linear_sin = nn.Linear(input_dim, 1)  # Para componente seno
        self.linear_cos = nn.Linear(input_dim, 1)  # Para componente coseno

    def forward(self, x):
        sin_output = self.linear_sin(torch.sin(x))
        cos_output = self.linear_cos(torch.cos(x))
        return sin_output + cos_output  # Combinación de ambos




### Uso del modelo ###
ventanas_generadas = generar_ventanas(objetos_dict, ventana=3)
ventana_dataset = VentanaDataset(ventanas_generadas)
ventana_dataloader = DataLoader(ventana_dataset, batch_size=4, shuffle=False, collate_fn=ventana_collate)

model = PTv3_deteccion(grid_size=(30, 30))


# Verificar si CUDA está disponible
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"device= {device}")
model.to(device)

# Inferencia #
#print("\n -----> Inferencia <-----\n")
#model.eval()

# Entrenamiento #
print("\n -----> Entrenamiento <-----\n")
model.train()

# Crear los modelos
modelo_lineal = RegresionLineal(input_dim=7).to('cuda')
modelo_ciclico = RegresionCiclica(input_dim=1).to('cuda')

# Definir los optimizadores
opt_lineal = optim.Adam(modelo_lineal.parameters(), lr=0.001)
opt_ciclico = optim.Adam(modelo_ciclico.parameters(), lr=0.001)

# Definir la función de pérdida
criterio = nn.MSELoss()

# Lista para almacenar los resultados finales de todos los lotes


epochs = 20
# Entrenamiento
for epoch in range(epochs):
    print(f"epoch {epoch+1}/{epochs}")
    resultados_finales = []
    for batch in tqdm(ventana_dataloader):
        entrada = batch["entrada"][0]
        salida = batch["salida"][0]
        #print(entrada)
        #print(salida)
        entrada = list(torch.unbind(entrada, dim=0))

        s = model(entrada)
        s_reducida = vecinos_proximos_agregar(s,3) # k-vecinos 3
        #print(s_reducida)

        # Paso 2: Dividir en regresión lineal y cíclica
        s_lineal = s_reducida[:7].view(1, -1)  # Asegúrate de que tenga forma [1, 7]
        s_ciclico = s_reducida[7:].view(1, -1)  # Asegúrate de que tenga forma [1, 1]

        salida_lineal = salida[:7].view(1, -1)  # Asegúrate de que tenga forma [1, 7]
        salida_ciclica = salida[7:].view(1, -1)  # Asegúrate de que tenga forma [1, 1]

        s_lineal = s_lineal.to(device)  # Mover s_lineal a la GPU
        s_ciclico = s_ciclico.to(device)  # Mover s_ciclico a la GPU
        salida_lineal = salida_lineal.to(device)  # Mover salida_lineal a la GPU
        salida_ciclica = salida_ciclica.to(device)  # Mover salida_ciclica a la GPU

        # Paso 3: Regresión lineal
        pred_lineal = modelo_lineal(s_lineal)
        loss_lineal = criterio(pred_lineal, salida_lineal)

        # Paso 4: Regresión cíclica
        pred_ciclico = modelo_ciclico(s_ciclico)
        loss_ciclico = criterio(pred_ciclico, salida_ciclica)

        loss_total = loss_lineal + loss_ciclico

        # Paso hacia atrás
        opt_lineal.zero_grad()
        opt_ciclico.zero_grad()
        loss_total.backward()
        opt_lineal.step()
        opt_ciclico.step()

        # Paso 7: Combinar resultados y almacenar
        resultado_final = torch.cat((pred_lineal.flatten(), pred_ciclico.flatten()), dim=0)
        resultados_finales.append(resultado_final)

        # Imprimir resultados por lote
        #print(f"Resultado final del lote: {resultado_final}")
        #print(f"Pérdida Lineal: {loss_lineal.item()}, Pérdida Cíclica: {loss_ciclico.item()}")

        

    # Guardar los modelos al final de cada época
    #torch.save(modelo_lineal.state_dict(), f"modelo_lineal_epoch_{epoch+1}.pt")
    #torch.save(modelo_ciclico.state_dict(), f"modelo_ciclico_epoch_{epoch+1}.pt")


print("__________________ Resultado del Entrenamiento _____________________")

print(resultados_finales)


def correr_prueba():
    ventanas_generadas = generar_ventanas(objetos_dict, ventana=3)
    ventana_dataset = VentanaDataset(ventanas_generadas)
    ventana_dataloader = DataLoader(ventana_dataset, batch_size=4, shuffle=False, collate_fn=ventana_collate)
    for batch in ventana_dataloader:
        print("Batch completo de entradas:")
        for i, entrada in enumerate(batch["entrada"]):
            print(f"\n🔹 Entrada {i}:")
            for j, tensor in enumerate(entrada):
                print(f" Tensor {j}: {tensor.shape}\n")  # Convertir a numpy para más claridad
        print("\nBatch de salidas:")
        print(batch["salida"])
        break  # Solo mostramos un batch para no llenar la consola


#correr_prueba()


