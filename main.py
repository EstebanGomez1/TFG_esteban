import torch
import torch.nn as nn
import torch.optim as optim
from utils import set_seed, get_device, setup_loggers
from model import PTv3_deteccion
import loader
from torch.utils.data import DataLoader, random_split
from trainner2 import run_epoch
from viewer3D_v1 import Visualizer3D

# 1. Preparación
set_seed()
device = get_device()
results_directory = "visualizacion3D/prueba"
error_logger, metrics_logger, trace_logger  = setup_loggers(results_directory)
visualizer = Visualizer3D(save_path=results_directory)

# 2. Datos
#diccionarios = loader.load_dicctionaries("diccionarios/dic_perf", 4)
#diccionarios = loader.load_dicctionaries("dic_pred", 1)
#diccionarios = loader.load_dicctionaries("diccionariosOld/actuales/dicc", 15)
diccionarios = loader.load_dicctionaries("diccionarios/diccionarios_pred/dicV2_pred", 1) 
#diccionarios = loader.load_dicctionaries("diccionarios/diccionarios_gt/dic2_perf", 5) 
dataset_total = loader.getMultiDataLoader2(diccionarios)
val_len = int(0.01 * len(dataset_total))
train_len = len(dataset_total) - val_len
train_dataset, val_dataset = random_split(dataset_total, [train_len, val_len])

#train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=loader.custom_collate_yolo_center_fn_smooth_rot)
#val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=loader.custom_collate_yolo_center_fn_smooth)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=loader.custom_collate_yolo_center_fn_smooth)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=loader.custom_collate_yolo_center_fn_smooth)

# 3. Modelo y configuración
model = PTv3_deteccion(grid_size=0.2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion_clf = nn.CrossEntropyLoss(ignore_index=-1)
#criterion_reg = nn.MSELoss()
criterion_reg = nn.SmoothL1Loss()
class StableAngularLoss(nn.Module):
    def forward(self, cyc_out, target_angle):
        delta = cyc_out - target_angle
        delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.mean(torch.sin(delta / 2) ** 2)

criterion_yaw = StableAngularLoss()

# 4. Entrenamiento
epochs = 50
num_classes = 8

for epoch in range(epochs):
    print(f"\n Epoch {epoch+1}/{epochs}")

    train_loss = run_epoch(
        model, train_loader, optimizer, device,
        criterion_clf, criterion_reg, criterion_yaw,
        num_classes, error_logger, mode="train", trace_logger=trace_logger, visualizer=None, epoch_idx=epoch
    )
    print(f" Train loss: {train_loss:.4f}")
    metrics_logger.info(f"Epoch {epoch+1}")
    metrics_logger.info(f"Train Loss={train_loss:.4f}")

    # val_loss, precision, recall, f1, mean_iou = run_epoch(
    #     model, val_loader, optimizer, device,
    #     criterion_clf, criterion_reg, criterion_yaw,
    #     num_classes, error_logger, mode="val", trace_logger=trace_logger,
    #     visualizer=visualizer, epoch_idx=epoch
    # )
    val_loss, class_accuracy, iou_accuracy, det_accuracy, mean_iou = run_epoch(
        model, val_loader, optimizer, device,
        criterion_clf, criterion_reg, criterion_yaw,
        num_classes, error_logger, mode="val", trace_logger=trace_logger,
        visualizer=visualizer, epoch_idx=epoch
    )
    # print(f" Val loss: {val_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | IoU3D: {mean_iou:.4f}")
    # metrics_logger.info(f"Val Loss={val_loss:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, IoU3D={mean_iou:.4f}")
    print(f" Val loss: {val_loss:.4f} | Class Precision: {class_accuracy:.4f} | IoU Accuracy: {iou_accuracy:.4f} | Detection Accuracy: {det_accuracy:.4f} | IoU3D: {mean_iou:.4f}")
    metrics_logger.info(f"Val Loss={val_loss:.4f}, Class Precision={class_accuracy:.4f}, IoU Accuracy={iou_accuracy:.4f}, Detection Accuracy={det_accuracy:.4f}, IoU3D={mean_iou:.4f}")
      
        # --- Guardar modelo por época ---
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'iou': mean_iou
    }
    torch.save(checkpoint, f"{results_directory}/model_epoch{epoch+1}.pth")

