import torch
import torch.nn as nn
import sys
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
from sklearn.metrics import precision_score, recall_score, average_precision_score
import numpy as np
from torch.utils.data import random_split, DataLoader
sys.path.append('IoU/2D-3D-IoUs-main')
from IoU import IoU3D, IoUs2D
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

class PTv3_deteccion(nn.Module):
    def __init__(self, grid_size: int):
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

        self.clf_head = nn.Linear(8, 8)      # 8 clases
        self.reg_head = nn.Linear(8, 6)      # 6 valores
        self.cyc_head = nn.Linear(8, 1)      # 1 valor

        # usamos una MLP para pasar a 8 caracteristicas
        self.feature_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )

    def forward(self, ventana: list):
        device = next(self.parameters()).device
        ventana = [f.to(device) for f in ventana]
        ventana = [f.squeeze(0) if f.ndim == 3 and f.shape[0] == 1 else f for f in ventana]
        all_points = torch.cat(ventana, dim=0)
        offset = torch.tensor([f.shape[0] for f in ventana], device=device).cumsum(0)
        points_dict = {
            "feat": all_points[:, 3:],
            "coord": all_points[:, :3],
            "offset": offset,
            "grid_size": self.grid_size,
        }
        point_features = self.point_encoder(points_dict)
        feat = points_dict["feat"]
        if feat.shape[1] == 0:
            feat = torch.zeros((feat.shape[0], 1), device=feat.device)
            points_dict["feat"] = feat
        feats_mean = torch.mean(point_features["feat"], dim=0, keepdim=True)  # [1, D]
        embedding = self.feature_layer(feats_mean)
        logits = self.clf_head(embedding)     # [batch, 8]
        reg_out = self.reg_head(embedding)    # [batch, 6]
        cyc_out = self.cyc_head(embedding)    # [batch, 1]
        return logits, reg_out, cyc_out
    

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
        self.linear = nn.Linear(input_dim, 8)

    def forward(self, x):
        return self.linear(x)


# Verificar si CUDA está disponible
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



diccionario = dataCompiler.load_dicctionaries(1)
ventana_dataloader = dataCompiler.getMultiDataLoader(diccionario) 
# == SPLIT DEL DATALOADER ==
dataset = ventana_dataloader.dataset
total_len = len(dataset)
val_len = int(0.2 * total_len)  # 20% validación
train_len = total_len - val_len

train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

train_dataloader = DataLoader(train_dataset, batch_size=ventana_dataloader.batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=ventana_dataloader.batch_size, shuffle=False)

# == PARAMETROS ==
num_clases = len(class_to_idx)
epochs = 2

# == MODELOS ==
model = PTv3_deteccion(grid_size=0.4).to(device)
modelo_clase = ClaseClasificacion(input_dim=1).to(device)
modelo_lineal = RegresionLineal(input_dim=6).to(device)
modelo_ciclico = RegresionCiclica(input_dim=1).to(device)

# == OPTIMIZADOR Y LOSSES ==
opt = optim.Adam(
    list(model.parameters()) + 
    list(modelo_lineal.parameters()) +
    list(modelo_ciclico.parameters()) +
    list(modelo_clase.parameters()),
    lr=0.001
)
criterio = nn.MSELoss()
criterio_clase = nn.CrossEntropyLoss(ignore_index=-1)

precisions = []
recalls = []
mean_IoI3Ds  = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # ----------- ENTRENAMIENTO -----------
    model.train(); modelo_clase.train(); modelo_lineal.train(); modelo_ciclico.train()
    running_loss = 0.0
    
    for entrada, salida in tqdm(train_dataloader):
        try:
            # Tu modelo ahora devuelve (logits_clase, salida_regresion, salida_ciclica)
            logits, reg_out, cyc_out = model(entrada)

            # Prepara los targets
            target_clase   = salida[:, 0].long().to(device)
            target_lineal  = salida[:, 1:7].to(device)
            target_ciclico = salida[:, 7].unsqueeze(1).to(device)

            # Calcula las pérdidas
            loss_class   = criterio_clase(logits, target_clase)
            loss_lineal  = criterio(reg_out, target_lineal)
            loss_ciclico = criterio(cyc_out, target_ciclico)
            loss_total   = loss_class + loss_lineal + loss_ciclico

            # Backpropagation & update
            opt.zero_grad()
            loss_total.backward()
            opt.step()

            running_loss += loss_total.item()
        except Exception as e:
            print(f"Error en entrenamiento: {e}")
            continue
    print(f"Loss entrenamiento: {running_loss/len(train_dataloader):.4f}")

    # ----------- VALIDACIÓN Y MÉTRICAS -----------
    model.eval(); modelo_clase.eval(); modelo_lineal.eval(); modelo_ciclico.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    all_iou3d = []
    with torch.no_grad():
        for entrada, salida in tqdm(val_dataloader):
            try:
                logits, reg_out, cyc_out = model(entrada)
                target_clase   = salida[:, 0].long().to(device)
                target_lineal  = salida[:, 1:7].to(device)
                target_ciclico = salida[:, 7].unsqueeze(1).to(device)

                loss_class = criterio_clase(logits, target_clase)
                loss_lineal = criterio(reg_out, target_lineal)
                loss_ciclico = criterio(cyc_out, target_ciclico)
                loss_total = loss_class + loss_lineal + loss_ciclico
                val_loss += loss_total.item()

                # Para métricas
                pred_classes = logits.argmax(dim=1).cpu().numpy()
                true_classes = target_clase.cpu().numpy()
                all_preds.extend(pred_classes)
                all_targets.extend(true_classes)

                #print("target_clase:", target_clase)
                #print("pred_clase:", pred_clase.argmax(dim=1))

                #IoU 3d
                # Concatenar para formar el label KITTI [x, y, z, l, w, h, alpha]
                pred_boxes = torch.cat([reg_out, cyc_out], dim=1)  # [batch, 7]
                true_boxes = torch.cat([target_lineal, target_ciclico], dim=1)  # [batch, 7]

                # Calcula IoU 3D por cada par predicho-real
                ious = IoU3D(pred_boxes, true_boxes)  # [batch]
                all_iou3d.extend(ious.cpu().numpy())
            except Exception as e:
                print(f"Error en validación: {e}")
                continue

    # Métricas de validación
    # --- Filtra los ignorados de las métricas si usas ignore_index en CrossEntropyLoss ---
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mask = (all_targets >= 0) & (all_targets < num_clases)
    all_preds = all_preds[mask]
    all_targets = all_targets[mask]

    # Métricas de validación
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    try:
        ap = average_precision_score(
            np.eye(num_clases)[all_targets],
            np.eye(num_clases)[all_preds],
            average=None
        )
    except Exception:
        ap = [0.0] * num_clases

    mean_IoI3D = np.mean(all_iou3d) if all_iou3d else 0.0

    print(f"Loss validación: {val_loss/len(val_dataloader):.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | Mean 3D IoU: {mean_IoI3D:.4f}")
    for i, class_name in enumerate(class_to_idx):
        print(f'AP ({class_name}): {ap[i]:.4f}')

    precisions.append(precision)
    recalls.append(recall)
    mean_IoI3Ds.append(mean_IoI3D)


# --- Graficar métricas ---
epochs_range = range(1, len(precisions) + 1)
plt.figure(figsize=(10,6))
plt.plot(epochs_range, precisions, label='Precision', marker='o')
plt.plot(epochs_range, recalls, label='Recall', marker='o')
plt.plot(epochs_range, mean_IoI3Ds, label='Mean 3D IoU', marker='o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Valor')
plt.title('Precision, Recall y Mean 3D IoU por Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("metricas.png")
plt.show()


# codigo antigo:

# for entrada, salida in tqdm(train_dataloader):
    #     try:
    #         #s = model(entrada)
    #         logits, reg_out, cyc_out = model(entrada)
    #         s_clase   = s[:,0].unsqueeze(1)
    #         s_lineal  = s[:,1:7]
    #         s_ciclico = s[:,7].unsqueeze(1)

    #         target_clase   = salida[:,0].long().to(device)
    #         target_lineal  = salida[:,1:7].to(device)
    #         target_ciclico = salida[:,7].unsqueeze(1).to(device)

    #         pred_clase = modelo_clase(s_clase)
    #         pred_lineal = modelo_lineal(s_lineal)
    #         pred_ciclico = modelo_ciclico(s_ciclico)

    #         loss_class = criterio_clase(pred_clase, target_clase)
    #         loss_lineal = criterio(pred_lineal, target_lineal)
    #         loss_ciclico = criterio(pred_ciclico, target_ciclico)
    #         loss_total = loss_class + loss_lineal + loss_ciclico

    #         opt.zero_grad()
    #         loss_total.backward()
    #         opt.step()

    #         running_loss += loss_total.item()           
    #     except Exception as e:
    #         print(f"Error en entrenamiento: {e}")
    #         continue


"""""s = model(entrada)
                s_clase   = s[:,0].unsqueeze(1)
                s_lineal  = s[:,1:7]
                s_ciclico = s[:,7].unsqueeze(1)

                target_clase   = salida[:,0].long().to(device)
                target_lineal  = salida[:,1:7].to(device)
                target_ciclico = salida[:,7].unsqueeze(1).to(device)

                pred_clase = modelo_clase(s_clase)
                pred_lineal = modelo_lineal(s_lineal)
                pred_ciclico = modelo_ciclico(s_ciclico)

                loss_class = criterio_clase(pred_clase, target_clase)
                loss_lineal = criterio(pred_lineal, target_lineal)
                loss_ciclico = criterio(pred_ciclico, target_ciclico)
                loss_total = loss_class + loss_lineal + loss_ciclico
                val_loss += loss_total.item()

                # Para métricas
                pred_classes = pred_clase.argmax(dim=1).cpu().numpy()
                true_classes = target_clase.cpu().numpy()
                all_preds.extend(pred_classes)
                all_targets.extend(true_classes)"""


