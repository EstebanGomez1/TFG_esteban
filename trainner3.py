import os
#os.environ["TORCH_USE_CUDA_DSA"] = "1"
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
from torch.cuda.amp import autocast
import dataCompiler
from sklearn.metrics import precision_score, recall_score, average_precision_score
import numpy as np
from torch.utils.data import random_split, DataLoader
sys.path.append('IoU/2D-3D-IoUs-main')
from IoU import IoU3D, IoUs2D
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import logging

logging.basicConfig(
    filename='errores.log',          # Nombre del archivo de log
    filemode='w',
    level=logging.ERROR,             # Nivel mínimo a registrar
    format='%(asctime)s %(levelname)s: %(message)s',  # Formato de cada línea
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
        self.cyc_head_sin = nn.Linear(8, 1)
        self.cyc_head_cos = nn.Linear(8, 1)      # 1 valor

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
        sin_out = self.cyc_head_sin(torch.sin(embedding))
        cos_out = self.cyc_head_cos(torch.cos(embedding))
        cyc_out = sin_out + cos_out   # [batch, 1]
        return logits, reg_out, cyc_out
    


# Verificar si CUDA está disponible
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


diccionario = dataCompiler.load_dicctionaries(15)
dataset_total = dataCompiler.getMultiDataLoader2(diccionario)

# 3. Split en train y val
total_len = len(dataset_total)
val_len = int(0.2 * total_len)
train_len = total_len - val_len
train_dataset, val_dataset = random_split(dataset_total, [train_len, val_len])

# 4. DataLoaders finales: **IMPORTANTE: pasa tu custom_collate_fn**
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=1, 
    shuffle=True, 
    num_workers=8, 
    pin_memory=True, 
    persistent_workers=True,
    collate_fn=dataCompiler.custom_collate2_fn
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=8, 
    pin_memory=True, 
    persistent_workers=True,
    collate_fn=dataCompiler.custom_collate2_fn
)

# == PARAMETROS ==
num_clases = len(class_to_idx)
epochs = 3

# == MODELOS ==
model = PTv3_deteccion(grid_size=0.4).to(device)

# == OPTIMIZADOR Y LOSSES ==
opt = optim.AdamW(
    list(model.parameters()),
    lr=0.001
)

criterio = nn.MSELoss()
criterio_clase = nn.CrossEntropyLoss(ignore_index=-1)
num_classes = 8
precisions = []
recalls = []
mean_IoI3Ds  = []
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # ----------- ENTRENAMIENTO -----------
    model.train();
    running_loss = 0.0
    
    for entrada, salida in tqdm(train_dataloader, leave=False):
        try:           
            opt.zero_grad()
            logits, reg_out, cyc_out = model(entrada)
            # Prepara los targets
            target_clase   = salida[0][0].unsqueeze(0).long().to(device)
            target_clase = torch.where(
                (target_clase >= 0) & (target_clase < num_classes),
                target_clase,
                torch.full_like(target_clase, -1)
            )
            target_lineal  = salida[0][1:7].unsqueeze(0).to(device)
            target_ciclico = salida[0][7].view(1,1).to(device)

            # Calcula las pérdidas  
            loss_class   = criterio_clase(logits, target_clase)
            loss_lineal  = criterio(reg_out, target_lineal)
            loss_ciclico = criterio(cyc_out, target_ciclico)
            #loss_total   = loss_class + loss_lineal + loss_ciclico
            loss_total = loss_class + 5.0 * loss_lineal + 5.0 * loss_ciclico
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                print("Loss anómala:", loss_total.item())
                continue
            loss_total.backward()
            opt.step()
            running_loss += loss_total.item()

        except Exception as e:
            print(e)
            logging.exception(e)
            continue
    print(f"Loss entrenamiento: {running_loss/len(train_dataloader):.4f}")

    # ----------- VALIDACIÓN Y MÉTRICAS -----------
    model.eval();
    val_loss = 0.0
    all_preds, all_targets = [], []
    all_iou3d = []
    with torch.no_grad():
        for entrada, salida in tqdm(val_dataloader, leave=False):
            try:
                logits, reg_out, cyc_out = model(entrada)
                # Prepara los targets
                target_clase   = salida[0][0].unsqueeze(0).long().to(device)
                #target_clase   = salida[:, 0].long().to(device)
                target_clase = torch.where(
                    (target_clase >= 0) & (target_clase < num_classes),
                    target_clase,
                    torch.full_like(target_clase, -1)
                )
                target_lineal  = salida[0][1:7].unsqueeze(0).to(device)
                target_ciclico = salida[0][7].view(1,1).to(device)
                #target_lineal  = salida[:, 1:7].to(device)
                #target_ciclico = salida[:, 7].unsqueeze(1).to(device)
                # Calcula las pérdidas
                # Backpropagation & update

                loss_class   = criterio_clase(logits, target_clase)
                loss_lineal  = criterio(reg_out, target_lineal)
                loss_ciclico = criterio(cyc_out, target_ciclico)
                loss_total   = loss_class + loss_lineal + loss_ciclico

                # Para métricas
                pred_classes = logits.argmax(dim=1).cpu().numpy()
                true_classes = target_clase.cpu().numpy()
                all_preds.extend(pred_classes)
                all_targets.extend(true_classes)

                # ========== IoU 3D por elemento ==========
                # Crea [x, y, z, l, w, h, alpha] para cada predicción y ground truth
                pred_boxes = torch.cat([reg_out, cyc_out], dim=1)   # [batch, 7]
                true_boxes = torch.cat([target_lineal, target_ciclico], dim=1) # [batch, 7]
                # Añade las dimensiones batch para IoU3D
                pred_boxes = pred_boxes.unsqueeze(0)
                true_boxes = true_boxes.unsqueeze(0)
                # Calcula IoU 3D para todo el batch (devuelve [1, batch])
                ious = IoU3D(pred_boxes, true_boxes).squeeze(0).detach().cpu().numpy() # [batch]
                # Opcional: clamp para evitar valores negativos por errores numéricos
                ious = np.clip(ious, 0, 1)
                all_iou3d.extend(ious.tolist())

            except Exception as e:
                print(e)
                logging.exception(e)
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