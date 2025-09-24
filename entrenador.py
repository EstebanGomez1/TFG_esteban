import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

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

# Definir el Dataset de PyTorch
class NubePuntosDataset(Dataset):
    def __init__(self, diccionario):
        """
        Args:
            diccionario (dict): Diccionario que contiene las nubes de puntos y etiquetas.
        """
        self.datos = []
        
        for idImagen, clases in diccionario.items():
            imagen_objetos = []
            for clase, datos in clases.items():
                puntos = np.array(datos["points"])
                etiqueta = datos["label"]

                # Convertir la etiqueta a un tensor
                etiqueta_tensor = torch.tensor([
                    etiqueta['x'], 
                    etiqueta['y'], 
                    etiqueta['z'], 
                    etiqueta['length'], 
                    etiqueta['height'], 
                    etiqueta['width'], 
                    etiqueta['rot_y']
                ], dtype=torch.float32)

                # Convertir los puntos a un tensor
                puntos_tensor = torch.tensor(puntos, dtype=torch.float32)

                # Añadir los puntos y la etiqueta como una tupla
                imagen_objetos.append((puntos_tensor, etiqueta_tensor))
            
            # Añadir la imagen con todos sus objetos
            self.datos.append(imagen_objetos)

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        return self.datos[idx]

# Función custom_collate_fn para manejar el tamaño variable de los tensores
def custom_collate_fn(batch):
    imagenes_batch = []
    
    # Iteramos por cada imagen en el batch
    for imagen_objetos in batch:
        objetos_batch = []
        for puntos, etiqueta in imagen_objetos:
            objetos_batch.append((puntos, etiqueta))
        
        imagenes_batch.append(objetos_batch)
    
    return imagenes_batch

# Crear el Dataset y DataLoader
dataset = NubePuntosDataset(diccionario)
dataloader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=custom_collate_fn)

# Iterar sobre el DataLoader y procesar los datos
for i, imagen in enumerate(dataloader):
    print(f"Batch {i + 1}:")
    
    for j, objeto in enumerate(imagen):
        print(f"  imagen: {i+1}.{j + 1}:")
        
        # Comprobar si el objeto es una lista de tuplas (puntos, etiquetas)
        if isinstance(objeto, list):
            print(f"    Número de elementos en la imagen : {len(objeto)}")
            for k, (puntos, etiqueta) in enumerate(objeto):
                print(f"      Objeto {k + 1}:")
                print(f"        Puntos: {puntos.shape}")
                print(f"        Etiqueta: {etiqueta}")
        else:
            print("    Error: El objeto no es una lista de tuplas (puntos, etiqueta).")
    
    #break  # El break es solo para mostrar el primer batch. Elimina esta línea para iterar por completo.






