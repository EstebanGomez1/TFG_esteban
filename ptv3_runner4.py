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
import matplotlib.pyplot as plt
import os
import dataCompiler

class_to_idx = {
    'car': 0,
    'pedestrian': 1,
    'van': 2,
    'cyclist': 3,
    'truck': 4
}

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
        #print(point_features["feat"].size())
        #feat_out = self.feature_layer(point_features["feat"])

        feats_mean = torch.mean(point_features["feat"], dim=0, keepdim=True)  # shape: [1, 128]
        #print(f"feats mean: {feats_mean.size()}")
        feat_out = self.feature_layer(feats_mean)

        return feat_out #torch.stack(resultado_final)


# Modelos de regresión
class RegresionLineal(nn.Module):
    def __init__(self, input_dim):
        super(RegresionLineal, self).__init__()
        self.linear = nn.Linear(input_dim, 6)  # Regresión para 6 valores

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
    
class ClaseClasificacion(nn.Module):
    def __init__(self, input_dim):
        super(ClaseClasificacion, self).__init__()
        self.linear = nn.Linear(input_dim, 5)

    def forward(self, x):
        return self.linear(x)

def performance_loss(
    pred_clase, target_clase,
    pred_lineal, target_lineal,
    pred_ciclico, target_ciclico,
    alpha=1.0, beta=1.0, gamma=1.0, delta=0.0
):
    
    # Clasificación
    loss_class = F.cross_entropy(pred_clase, target_clase)
    # Regresión lineal
    loss_lineal = F.smooth_l1_loss(pred_lineal, target_lineal)
    # Regresión cíclica
    loss_ciclico = F.mse_loss(pred_ciclico, target_ciclico)
    # Penalización extra: error en centroide (x, y, z)
    penalizacion = torch.mean(torch.norm(pred_lineal[..., :3] - target_lineal[..., :3], dim=1))

    # Pérdida total combinada
    loss_total = (
        alpha * loss_class +
        beta * loss_lineal +
        gamma * loss_ciclico +
        delta * penalizacion
    )

    return loss_total, loss_class, loss_lineal, loss_ciclico, penalizacion

# Verificar si CUDA está disponible
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

### Uso del modelo ###
# ventanas_generadas = generar_ventanas(objetos_dict, ventana=3)
# ventana_dataset = VentanaDataset(ventanas_generadas)
# ventana_dataloader = DataLoader(ventana_dataset, batch_size=4, shuffle=False, collate_fn=ventana_collate)


diccionario = dataCompiler.load_dictionary("dicc3.pkl")
ventana_dataloader = dataCompiler.getDataLoader(diccionario)

# Cargar el modelo principal
model = PTv3_deteccion(grid_size=(30, 30))

print(f"device= {device}")
model.to(device)

# Inferencia #
#print("\n -----> Inferencia <-----\n")
#model.eval()

# Entrenamiento #
print("\n -----> Entrenamiento <-----\n")
model.train()

# Crear los modelos
num_clases = len(class_to_idx)
modelo_clase = ClaseClasificacion(input_dim=1).to(device)
modelo_lineal = RegresionLineal(input_dim=6).to('cuda')
modelo_ciclico = RegresionCiclica(input_dim=1).to('cuda')

# Definir el optimizador
opt = optim.Adam(
    list(modelo_lineal.parameters()) +
    list(modelo_ciclico.parameters()) +
    list(modelo_clase.parameters()),
    lr=0.001
)

# Definir la función de pérdida
criterio = nn.MSELoss()

# Lista para almacenar los resultados finales de todos los lotes


criterio_clase = nn.CrossEntropyLoss()

losses = [] 
losses_total = []
losses_class = []
losses_lineal = []
losses_ciclico = []
penalizaciones = []
epochs = 1
# Entrenamiento
for epoch in range(epochs):
    epoch_total = []
    epoch_class = []
    epoch_lineal = []
    epoch_ciclico = []
    epoch_penal = []
    

    print(f"epoch {epoch+1}/{epochs}")
    resultados_finales = []
    epoch_losses = [] 
    for entrada, salida in tqdm(ventana_dataloader):
        print(f"input entrada: {entrada}")
        print(f"target salida: {salida}")

        #entrada = list(torch.unbind(entrada, dim=0))

        s = model(entrada)
        #print(f"prediccion: {s}")

        # Separación de características
        s_clase   = s[:, 0].view(1, 1)         
        s_lineal  = s[:, 1:7]                  
        s_ciclico = s[:, 7].view(1, 1)         

        # Targets
        target_clase   = salida[0].long().view(1).to(device)     
        target_lineal  = salida[1:7].view(1, -1).to(device)      
        target_ciclico = salida[7].view(1, 1).to(device)         

        #predicciones
        pred_clase = modelo_clase(s_clase).to(device)  
        pred_lineal = modelo_lineal(s_lineal).to(device)  
        pred_ciclico = modelo_ciclico(s_ciclico).to(device)  

        # Regresiones
        loss_class = criterio_clase(pred_clase, target_clase)
        loss_lineal = criterio(pred_lineal, target_lineal)
        loss_ciclico = criterio(pred_ciclico, target_ciclico)

        #loss_total = loss_class + loss_lineal + loss_ciclico

        # Performance loss
        loss_total, loss_class, loss_lineal, loss_ciclico, penalizacion = performance_loss(
            pred_clase, target_clase, pred_lineal, target_lineal, pred_ciclico, target_ciclico,
            alpha=1.0, beta=1.0, gamma=1.0, delta=0.1
        )

        # backward
        opt.zero_grad()
        loss_total.backward()
        opt.step()

        # Guarda las métricas
        epoch_total.append(loss_total.item())
        epoch_class.append(loss_class.item())
        epoch_lineal.append(loss_lineal.item())
        epoch_ciclico.append(loss_ciclico.item())
        epoch_penal.append(penalizacion.item())


        #epoch_losses.append(loss_total.item())   
    # Guarda el promedio de cada métrica por epoch
    losses_total.append(sum(epoch_total) / len(epoch_total))
    losses_class.append(sum(epoch_class) / len(epoch_class))
    losses_lineal.append(sum(epoch_lineal) / len(epoch_lineal))
    losses_ciclico.append(sum(epoch_ciclico) / len(epoch_ciclico))
    penalizaciones.append(sum(epoch_penal) / len(epoch_penal))
            
    # avg_loss = sum(epoch_losses) / len(epoch_losses)
    # losses.append(sum(epoch_losses)/len(epoch_losses))



#Wrapper de modelos
wrapper = {
    "model_principal_state_dict": model.state_dict(),
    "modelo_lineal_state_dict": modelo_lineal.state_dict(),
    "modelo_ciclico_state_dict": modelo_ciclico.state_dict(),
    "modelo_clase_state_dict": modelo_clase.state_dict(),
    "optimizer_state_dict": opt.state_dict(),
}

# Guardar los modelos
torch.save(wrapper, "modelos_combinados.pth")

print("__________________ Resultado del Entrenamiento _____________________")


plt.figure(figsize=(10,5))
plt.plot(losses_total, label="Performance Loss Total")
plt.plot(losses_class, label="Clasificación")
plt.plot(losses_lineal, label="Regresión")
plt.plot(losses_ciclico, label="Cíclica")
plt.plot(penalizaciones, label="Penalización")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("performance_loss.png")
print("Gráfica guardada como performance_loss.png")

#print(resultados_finales[-1])

