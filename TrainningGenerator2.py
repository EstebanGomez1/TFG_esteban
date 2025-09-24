import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from collections import defaultdict

# Cargar el archivo .pkl que contiene el diccionario
archivo_pickle = 'diccionario0000.pkl'

try:
    with open(archivo_pickle, 'rb') as file:
        diccionario = pickle.load(file)
    print("Diccionario cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: El archivo {archivo_pickle} no fue encontrado.")
    diccionario = {}

class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data = defaultdict(lambda: {"points": [], "labels": []})
        
        # Agrupar por objeto (class_id)
        for img_id, objects in data_dict.items():
            for obj_id, obj_data in objects.items():
                points = obj_data["points"]  # Nube de puntos
                label = obj_data["label"]  # Info del objeto
                
                # Guardar todos los puntos y labels asociados a este objeto
                self.data[obj_id]["points"].append(torch.tensor(points, dtype=torch.float32))
                self.data[obj_id]["labels"].append(label)

        # Convertimos el diccionario en una lista para el Dataset
        self.data = list(self.data.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj_id, obj_info = self.data[idx]
        return {
            "id": obj_id,
            "points": obj_info["points"],  # Lista de nubes de puntos
            "labels": obj_info["labels"]   # Lista de labels
        }

# Función para agrupar en batches 
def custom_collate(batch):
    ids = [item["id"] for item in batch]
    points = [item["points"] for item in batch]  # Lista de listas de tensores
    labels = [item["labels"] for item in batch]  # Lista de listas de labels

    return {"id": ids, "points": points, "labels": labels}


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


# Crear dataset y dataloader
dataset = CustomDataset(diccionario)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

# Convertir DataLoader en una estructura de diccionario
objetos_dict = {}

for batch in dataloader:
    for obj_id, points_list, labels_list in zip(batch["id"], batch["points"], batch["labels"]): 
        objetos_dict[obj_id] = {
            "points": points_list,  # Lista de nubes de puntos
            "labels": labels_list   # Lista de labels
        }

# Aplicar ventanas deslizantes
ventanas_generadas = generar_ventanas(objetos_dict, ventana=3)

# Imprimir sin ventana deslizante
for batch in dataloader:
    print("--------------")
    
    for obj_id, points_list, labels_list in zip(batch["id"], batch["points"], batch["labels"]): 
        print(f"Objeto ID: {obj_id}")
        
        for i, (points, label) in enumerate(zip(points_list, labels_list)):
            num_puntos = points.shape[0] 
            print(f"  Imagen {i}: {num_puntos} puntos - Label: {label}")

print("\n ////////////// Ventanas Deslizantes \\\\\\\\\\\\\\\ \n")

# Imprimir con ventana deslizante
for entrada, salida in ventanas_generadas:
    print(f"Entrada: {[e.shape for e in entrada]} -> Salida: {salida}")

print("\n ////////////// DataLoader \\\\\\\\\\\\\\\ \n")

class VentanaDataset(Dataset):
    def __init__(self, ventanas):
        self.ventanas = ventanas

    def __len__(self):
        return len(self.ventanas)

    def __getitem__(self, idx):
        entrada, salida = self.ventanas[idx]
        entrada = [torch.tensor(e, dtype=torch.float32) for e in entrada]  # Convertir a tensores
        salida = torch.tensor(salida, dtype=torch.long)  # Asegurar que salida sea tensor

        return {"entrada": entrada, "salida": salida}

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

    # Padding para nubes de puntos dentro de cada batch
    max_len = max(len(seq) for seq in entradas)
    entradas_padded = [seq + [torch.zeros_like(seq[0])] * (max_len - len(seq)) for seq in entradas]

    return {"entrada": entradas_padded, "salida": salidas}

# Crear el dataset y el dataloader
ventana_dataset = VentanaDataset(ventanas_generadas)
ventana_dataloader = DataLoader(ventana_dataset, batch_size=4, shuffle=True, collate_fn=ventana_collate)

# Verificación del DataLoader
for batch in ventana_dataloader:
    print("Batch completo de entradas:")
    for i, entrada in enumerate(batch["entrada"]):
        print(f"\n🔹 Entrada {i}:")
        for j, tensor in enumerate(entrada):
            print(f" Tensor {j}: {tensor.shape}\n")  # Convertir a numpy para más claridad
    print("\nBatch de salidas:")
    print(batch["salida"])
    break  # Solo mostramos un batch para no llenar la consola


