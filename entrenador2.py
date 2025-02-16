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
    # Cargar el diccionario desde el archivo .pkl
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

# Función para agrupar en batches (manteniendo estructuras variables)
def custom_collate(batch):
    ids = [item["id"] for item in batch]
    points = [item["points"] for item in batch]  # Lista de listas de tensores
    labels = [item["labels"] for item in batch]  # Lista de listas de labels

    return {"id": ids, "points": points, "labels": labels}

def ventana_deslizante(dataloader, ventana_size=3):
    # Recorremos las iteraciones del dataloader
    for batch in dataloader:
        print("--------------")
        
        # Imprimir la información completa de la iteración actual
        print(f"Objeto ID: {batch['id']}")
        
        # Recorrer las imágenes, sus puntos y etiquetas 
        for imagen_id, points_list, labels_list in zip(batch['id'], batch['points'], batch['labels']):
            print(f"Imagen ID: {imagen_id}")
            
            # Aplicamos la ventana deslizante
            for start_idx in range(len(points_list) - ventana_size + 1):
                # Obtenemos los puntos y etiquetas correspondientes a la ventana deslizante
                points_ventana = points_list[start_idx:start_idx + ventana_size]
                labels_ventana = labels_list[start_idx:start_idx + ventana_size]

                # Imprimir la ventana deslizante
                print(f"  Ventana deslizante de puntos {start_idx + 1} a {start_idx + ventana_size}:")
                
                for i in range(len(points_ventana)):
                    print(f"    Imagen {start_idx + i}: {len(points_ventana[i])} puntos - Label: {labels_ventana[i]}")
                
            print("----")
        print("====================")


dataset = CustomDataset(diccionario)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)


# Ejemplo de iteración
for batch in dataloader:
    print("--------------")
    
    for obj_id, points_list, labels_list in zip(batch["id"], batch["points"], batch["labels"]): 
        print(f"Objeto ID: {obj_id}")
        
        for i, (points, label) in enumerate(zip(points_list, labels_list)):
            num_puntos = points.shape[0] 
            print(f"  Imagen {i}: {num_puntos} puntos - Label: {label}")

print("/////////////////////////////////")
ventana_deslizante(dataloader, ventana_size=3)

