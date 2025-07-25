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
import dataCompiler2
from sklearn.metrics import precision_score, recall_score, average_precision_score
import numpy as np
from torch.utils.data import random_split, DataLoader
sys.path.append('IoU/2D-3D-IoUs-main')
from IoU import IoU3D, IoUs2D
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import funciones4
import logging
from viewer3D_v1 import Visualizer3D
from datetime import datetime
import matplotlib.pyplot as plt

# Logger para errores
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler('errores.log', mode='w')
error_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
error_handler.setFormatter(error_formatter)
error_logger.addHandler(error_handler)

# Logger para métricas
metrics_logger = logging.getLogger('metrics_logger')
metrics_logger.setLevel(logging.INFO)
metrics_handler = logging.FileHandler('metricas.log', mode='w')
metrics_formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
metrics_handler.setFormatter(metrics_formatter)
metrics_logger.addHandler(metrics_handler)

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

        self.feature_layer = nn.Sequential(      
            nn.Linear(64, 64),
            nn.ReLU(),      
            nn.Linear(64, 32)
        )

        # 64*23*23   18432
        self.clf_head = nn.Sequential(
            nn.Linear(36864, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        ) 
        self.reg_head = nn.Sequential(
            nn.Linear(36864, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )  
        self.cyc_head_sin = nn.Sequential(
            nn.Linear(36864, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self.cyc_head_cos = nn.Sequential(
            nn.Linear(36864, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
            #nn.AdaptiveAvgPool2d((1,1))
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1)
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
        #feats_mean = torch.mean(point_features["feat"], dim=0, keepdim=True)  # [1, D]
        #embedding = self.feature_layer(feats_mean)
        #_-------_
        grid_feat = self.scatter_grid_pooling(
            point_features["coord"].detach(),
            point_features["feat"].detach(),  # 👈 aquí están las features
            res_x=0.25,
            grid_size= 24, #6/0.25
            device=device
        )
        grid_feat = grid_feat.unsqueeze(0)  # [1, F]

        #############
        """# Visualizar y guardar
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"heatmap_{timestamp}.png"
        grid_np = grid_feat[0, 0].detach().cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(grid_np, cmap='hot', origin='lower')
        plt.colorbar(label='Número de puntos')
        plt.title("Distribución de puntos en la rejilla")

        # Asegúrate de que la carpeta exista
        folder = "grids"
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        plt.savefig(filepath)
        plt.close()

        # Extraer coordenadas XY
        points_np = all_points[:, :3].detach().cpu().numpy()
        x, y = points_np[:, 0], points_np[:, 1]

        # Crear figura 2D
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, c='blue', s=1)  # puedes cambiar color o tamaño de punto
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Nube de puntos proyectada en XY")
        plt.axis('equal')  # para que los ejes tengan la misma escala

        # Guardar imagen
        cloud_img_filename = f"cloud_{timestamp}.png"
        cloud_img_path = os.path.join(folder, cloud_img_filename)
        plt.savefig(cloud_img_path)
        plt.close()"""
        #############



        #embedding = self.feature_layer(grid_feat)
        embedding = self.conv_block(grid_feat)  
        embedding = embedding.view(1, -1)  
        #_-------_
        normalized_embedding = embedding
        logits = self.clf_head(normalized_embedding)     # [batch, 8]
        reg_out = self.reg_head(normalized_embedding)    # [batch, 6]
        sin_out = self.cyc_head_sin(normalized_embedding)
        cos_out = self.cyc_head_cos(normalized_embedding)
        cyc_out = torch.cat([sin_out, cos_out], dim=1) 
        #cyc_out = F.normalize(cyc_out, dim=1)
        return logits, reg_out, cyc_out
    
    def scatter_grid_pooling(self, coords, feats, res_x=0.25, grid_size=24, device="cpu"):
        assert coords.shape[0] == feats.shape[0], "coords y feats deben tener el mismo número de puntos"

        x = coords[:, 0]
        y = coords[:, 1]

        F = feats.shape[1]
        grid = torch.zeros((F, grid_size, grid_size), device=device)

        half_extent = (grid_size * res_x) / 2

        cx = ((x + half_extent) / res_x).long()
        cy = ((y + half_extent) / res_x).long()

        mask = (cx >= 0) & (cx < grid_size) & (cy >= 0) & (cy < grid_size)
        cx = cx[mask]
        cy = cy[mask]
        feats = feats[mask]

        indices = cx * grid_size + cy  
        grid = torch.zeros((F, grid_size * grid_size), device=device)

        grid = grid.index_add(1, indices, feats.T) 

        grid = grid.view(F, grid_size, grid_size)

        return grid





    
#seed = 10
seed = int(time.time())

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Si usas CUDA:
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Verificar si CUDA está disponible
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


diccionario = dataCompiler2.load_dicctionaries("dic_perf",5)
dataset_total = dataCompiler2.getMultiDataLoader2(diccionario)

#seed = 30
seed = int(time.time())

generator = torch.Generator().manual_seed(seed)
total_len = len(dataset_total)
val_len = int(0.1 * total_len)
train_len = total_len - val_len
train_dataset, val_dataset = random_split(dataset_total, [train_len, val_len], generator=generator)

# 4. DataLoaders finales: **IMPORTANTE: pasa tu custom_collate_fn**
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=1, 
    shuffle=True, 
    collate_fn=dataCompiler2.custom_collate_yolo_center_fn_smooth
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=1, 
    shuffle=False, 
    collate_fn=dataCompiler2.custom_collate_yolo_center_fn_smooth
)

# == PARAMETROS ==
num_clases = len(class_to_idx)
epochs = 20

# == MODELOS ==
model = PTv3_deteccion(grid_size=0.2).to(device)
# == OPTIMIZADOR Y LOSSES ==
opt = optim.AdamW(
    list(model.parameters()),
    lr=0.001
)

start_epoch = 0
class CosineAngularLoss(nn.Module):
    def __init__(self):
        super(CosineAngularLoss, self).__init__()

    def forward(self, pred_angle, target_angle):
        return torch.mean(1 - torch.cos(pred_angle - target_angle))


class StableAngularLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cyc_out, target_angle):
        delta = cyc_out - target_angle
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        return torch.mean(torch.sin(delta / 2) ** 2)

#criterio_ciclico = CosineAngularLoss()
criterio_ciclico = StableAngularLoss()

criterio = nn.MSELoss()
criterio_clase = nn.CrossEntropyLoss(ignore_index=-1)
num_classes = 8
precisions = []
recalls = []
mean_IoI3Ds  = []


# Variables para seguimiento
best_iou = 0.0
last_checkpoint_path = None
min_improvement = 0.01  # Mejora mínima requerida de 0.1 IoU
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Para reiniciar desde checkpoints
start_epoch = 0
cont_optimizer = False  # Flag para saber si continuamos entrenamiento

vis = Visualizer3D()

for epoch in range(start_epoch, epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # ----------- ENTRENAMIENTO -----------
    model.train();
    running_loss = 0.0
    contador = 0
    contador_saltos = 0
    contador_clip = 0
    #TODO: añadir checkpoints
    total_class_loss = 0.0
    total_lineal_loss = 0.0
    total_ciclico_loss = 0.0
    num_batches = 0
    #opt.zero_grad()
    for entrada, salida, centro, salto in tqdm(train_dataloader):       
        loss_total = 0
        if salto[0]:
            contador_saltos += 1
            continue
        try:           
            logits, reg_out, cyc_out = model(entrada)
            # Prepara los targets
            target_clase   = salida[0][0].unsqueeze(0).long().to(device)
            target_clase = torch.where(
                (target_clase >= 0) & (target_clase < num_classes),
                target_clase,
                torch.full_like(target_clase, -1)
            )
            # Coordenadas relativas al centro del crop
            target_lineal  = salida[0][1:7].unsqueeze(0).to(device) #retocar para solo las dimensiones
            target_ciclico = salida[0][7].view(1,1).to(device)
            # Calcula las pérdidas  
            loss_class   = criterio_clase(logits, target_clase)
            loss_lineal  = criterio(reg_out, target_lineal)

            pred_yaw = torch.atan2(cyc_out[:, 0], cyc_out[:, 1])
            loss_ciclico = criterio_ciclico(pred_yaw, target_ciclico)           
            # gamma = 1 para lr 0.0005
            # alpha = 0.75
            # beta = 2
            gamma = 1.5
            alpha = 1
            beta = 1.5
            loss_total   = gamma*loss_class + loss_lineal*alpha + loss_ciclico*beta
            if loss_total > 150:
                metrics_logger.info(f"loss: {loss_total}")
                metrics_logger.info(f"clase: {loss_class*gamma}")
                metrics_logger.info(f"lineal: {loss_lineal*alpha}")
                metrics_logger.info(f"ciclico: {beta*loss_ciclico}")
            #loss_total = loss_class + 5.0 * loss_lineal + 5.0 * loss_ciclico
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                metrics_logger.info(f"clase: {loss_class*gamma}")
                metrics_logger.info(f"lineal: {loss_lineal}")
                metrics_logger.info(f"ciclico: {loss_ciclico}")
                opt.zero_grad() 
                sys.exit(1)
                continue
            (loss_total / 1).backward()
            if (contador+1) % 1 == 0:
                 total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
                 if(total_norm >50):
                    contador_clip += 1
                 opt.step()
                 opt.zero_grad()
            running_loss += loss_total.item()
            contador = contador +1
            total_class_loss += loss_class*gamma
            total_lineal_loss += loss_lineal*alpha
            total_ciclico_loss += beta*loss_ciclico
            num_batches += 1

        except Exception as e:
            error_logger.exception(e)
            continue

    metrics_logger.info(f"----------------------------------------------------------------fin etapa----------------------------------------------")
    print(f"Loss entrenamiento: {running_loss/len(train_dataloader):.4f}")
    metrics_logger.info(f"Loss entrenamiento: {running_loss/len(train_dataloader):.4f}")
    metrics_logger.info(f"Clipping en epoch: {contador_clip}")
    metrics_logger.info(f"saltos en epoch: {contador_saltos}")
    # ----------- VALIDACIÓN Y MÉTRICAS -----------
    model.eval();
    val_loss = 0.0
    all_preds, all_targets = [], []
    all_iou3d = []
    contador_val =0
    with torch.no_grad():
        for entrada, salida, centro, salto in tqdm(val_dataloader):
            contador_val = contador_val +1
            
            try:
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
                # Backpropagation & update

                loss_class   = criterio_clase(logits, target_clase)
                loss_lineal  = criterio(reg_out, target_lineal)   
                pred_yaw = torch.atan2(cyc_out[:, 0], cyc_out[:, 1])
                loss_ciclico = criterio_ciclico(pred_yaw, target_ciclico)   
                gamma = 1.5
                alpha = 1
                beta = 1.5
                loss_total   =  gamma*loss_class + loss_lineal*alpha + loss_ciclico*beta
                val_loss += loss_total.item()
                # Para métricas
                pred_classes = logits.argmax(dim=1).cpu().numpy()
                true_classes = target_clase.cpu().numpy()
                all_preds.extend(pred_classes)
                all_targets.extend(true_classes)
                #metrics_logger.info(f"prediccion de clase: {pred_classes} vs true clase {true_classes}")

                # ========== IoU 3D por elemento ==========
                # 1. Calcula coordenadas globales sumando el centro del crop
                pred_xyz_global = reg_out[:, :3]         # [1, 3]
                true_xyz_global = target_lineal[:, :3] # [1, 3]
                # 2. Crea cajas [x, y, z, w, h, l, alpha] para predicción y ground truth
                # cyc_out --> [alpha]
                pred_box = torch.cat([pred_xyz_global, reg_out[:, 3:], cyc_out], dim=1)   # [1, 7]
                true_box = torch.cat([true_xyz_global, target_lineal[:, 3:], target_ciclico], dim=1) # [1, 7]

                # 3. Añade dimensión batch extra si lo pide tu función IoU3D (usualmente espera [B, N, 7])
                pred_boxes = pred_box.unsqueeze(0)   # [1, 1, 7]
                true_boxes = true_box.unsqueeze(0)   # [1, 1, 7]

                # 4. Calcula IoU3D
                ious = IoU3D(pred_boxes, true_boxes).squeeze(0).detach().cpu().numpy()   # [N] o [batch]
                ious = np.clip(ious, 0, 1)
                all_iou3d.extend(ious.tolist())


                # Guarda visualizacion 3D
                if contador_val % 25 == 1:
                    # Desnormalizar puntos
                    centro_crop = centro[0].to(device).view(-1)
                    entrada_denorm = []
                    for nube in entrada:
                        nube_denorm = nube.clone().to(device)
                        nube_denorm[:, :3] += centro_crop
                        entrada_denorm.append(nube_denorm)
                    
                    all_points = torch.cat(entrada_denorm, dim=0)
                    
                    # ---- Obtener predicción del modelo ----
                    pred_xyz = reg_out[:, :3].squeeze(0) + centro_crop         # x, y, z
                    pred_dims = reg_out[:, 3:].squeeze(0)          # w, l, h (en ese orden)
                    pred_yaw = torch.atan2(cyc_out[:, 0], cyc_out[:, 1]).squeeze(0)  # ángulo en radianes
                    box_pred_7 = torch.cat([pred_xyz, pred_dims, pred_yaw.view(1)], dim=0)
                    # ---- Obtener ground truth ----
                    gt_xyz = target_lineal[:, :3].squeeze(0) + centro_crop
                    gt_dims = target_lineal[:, 3:].squeeze(0)
                    gt_yaw = target_ciclico.squeeze(0)
                    box_gt_7 = torch.cat([gt_xyz, gt_dims, gt_yaw.view(1)], dim=0)


                    # ---- Visualizar ----
                    vis.plot(all_points, pred_box=box_pred_7, target_box=box_gt_7, filename=f"debug_demo{contador_val}.png")


            except Exception as e:
                #print(e)
                #error_logger.info(f"contador = {contador}")
                error_logger.info(entrada)
                error_logger.info(salida)
                error_logger.exception(e)
                continue

    # Métricas de validación

    mean_IoI3D = np.mean(all_iou3d) if all_iou3d else 0.0

    iou_threshold = 0.5
    tp = 0
    fp = 0
    fn = 0

    for pred_cls, true_cls, iou in zip(all_preds, all_targets, all_iou3d):
        if pred_cls == true_cls:
            if iou >= iou_threshold:
                tp += 1
            else:
                fn += 1  # predijo bien la clase pero mal la caja
        else:
            if iou >= iou_threshold:
                fp += 1  # predijo mal la clase aunque ubicó bien
            else:
                fp += 1  # predijo mal todo

    # Evitar divisiones por cero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"Loss validación: {val_loss/len(val_dataloader):.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | Mean 3D IoU: {mean_IoI3D:.4f}")
    
    metrics_logger.info(f"Loss validación: {val_loss/len(val_dataloader):.4f}")
    metrics_logger.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | Mean 3D IoU: {mean_IoI3D:.4f}")
    metrics_logger.info(f"TP: {tp} | FP: {fp} | FN: {fn}")
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    metrics_logger.info(f"F1-score: {f1:.4f}")  
    print(f"F1-score: {f1:.4f}")

    precisions.append(precision)
    recalls.append(recall)
    mean_IoI3Ds.append(mean_IoI3D)

    mean_class_loss = total_class_loss / num_batches
    mean_lineal_loss = total_lineal_loss / num_batches
    mean_ciclico_loss = total_ciclico_loss / num_batches
    metrics_logger.info(f"Loss media clase: {mean_class_loss}")
    metrics_logger.info(f"Loss media lineal: {mean_lineal_loss}")
    metrics_logger.info(f"Loss media ciclica: {mean_ciclico_loss}")
