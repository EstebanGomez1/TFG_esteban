import torch
import torch.nn as nn
import torch.optim as optim
from utils import set_seed, get_device, setup_loggers
from model import PTv3_deteccion
from loader import load_dicctionaries, getMultiDataLoader2, custom_collate_yolo_center_fn_smooth_rot
from torch.utils.data import DataLoader, random_split
from trainner import run_epoch
from viewer3D_v1 import Visualizer3D

# 1. Preparación
set_seed()
device = get_device()
error_logger, metrics_logger, trace_logger  = setup_loggers()
visualizer = Visualizer3D()

# 2. Datos
diccionarios = load_dicctionaries("diccionarios/dic_perf", 1)
dataset_total = getMultiDataLoader2(diccionarios)
val_len = int(0.1 * len(dataset_total))
train_len = len(dataset_total) - val_len
train_dataset, val_dataset = random_split(dataset_total, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_yolo_center_fn_smooth_rot)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_yolo_center_fn_smooth_rot)

# 3. Modelo y configuración
model = PTv3_deteccion(grid_size=0.2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion_clf = nn.CrossEntropyLoss(ignore_index=-1)
criterion_reg = nn.MSELoss()

class StableAngularLoss(nn.Module):
    def forward(self, cyc_out, target_angle):
        delta = cyc_out - target_angle
        delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.mean(torch.sin(delta / 2) ** 2)

criterion_yaw = StableAngularLoss()

# 4. Entrenamiento
epochs = 20
num_classes = 8

for epoch in range(epochs):
    print(f"\n Epoch {epoch+1}/{epochs}")

    train_loss = run_epoch(
        model, train_loader, optimizer, device,
        criterion_clf, criterion_reg, criterion_yaw,
        num_classes, error_logger, mode="train", trace_logger=trace_logger
    )
    print(f" Train loss: {train_loss:.4f}")
    metrics_logger.info(f"Epoch {epoch+1}")
    metrics_logger.info(f"Train Loss={train_loss:.4f}")

    val_loss, precision, recall, f1, mean_iou = run_epoch(
        model, val_loader, optimizer, device,
        criterion_clf, criterion_reg, criterion_yaw,
        num_classes, error_logger, mode="val", trace_logger=trace_logger,
        visualizer=visualizer, epoch_idx=epoch
    )
    print(f" Val loss: {val_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | IoU3D: {mean_iou:.4f}")
    metrics_logger.info(f"Val Loss={val_loss:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, IoU3D={mean_iou:.4f}")
