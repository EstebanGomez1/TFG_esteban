import torch
import numpy as np
import random
import time
from tqdm import tqdm
import sys
sys.path.append('IoU/2D-3D-IoUs-main')
from IoU import IoU3D, IoUs2D


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

def run_epoch(model, dataloader, optimizer, device, criterion_clf, criterion_reg, criterion_yaw,
              num_classes, logger, mode="train", trace_logger=None, visualizer=None, epoch_idx=None):
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    total_class_loss = 0.0
    total_lineal_loss = 0.0
    total_ciclico_loss = 0.0
    num_batches = 0

    all_preds, all_targets = [], []
    all_iou3d = []

    contador_saltos = 0
    contador_clip = 0
    contador_val = 0

    with torch.set_grad_enabled(is_train):
        for entrada, salida, centro, salto in tqdm(dataloader):
            if salto[0]:
                contador_saltos += 1
                continue
            try:
                logits, reg_out, cyc_out = model(entrada)
                target_clase = salida[0][0].unsqueeze(0).long().to(device)
                target_clase = torch.where(
                    (target_clase >= 0) & (target_clase < num_classes),
                    target_clase,
                    torch.full_like(target_clase, -1)
                )
                target_lineal = salida[0][1:7].unsqueeze(0).to(device)
                target_ciclico = salida[0][7].view(1,1).to(device)

                loss_class = criterion_clf(logits, target_clase)
                loss_lineal = criterion_reg(reg_out, target_lineal)
                pred_yaw = torch.atan2(cyc_out[:, 0], cyc_out[:, 1])
                loss_ciclico = criterion_yaw(pred_yaw, target_ciclico)

                gamma, alpha, beta = 1, 1.25, 1.25
                loss_total = gamma*loss_class + alpha*loss_lineal + beta*loss_ciclico

                if is_train:
                    loss_total.backward()
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
                    if total_norm > 50:
                        contador_clip += 1
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss_total.item()
                total_class_loss += gamma * loss_class.item()
                total_lineal_loss += alpha * loss_lineal.item()
                total_ciclico_loss += beta * loss_ciclico.item()
                num_batches += 1

                # Para métricas solo si es validación
                if not is_train:
                    pred_classes = logits.argmax(dim=1).cpu().numpy()
                    true_classes = target_clase.cpu().numpy()
                    all_preds.extend(pred_classes)
                    all_targets.extend(true_classes)

                    pred_box = torch.cat([reg_out[:, :3], reg_out[:, 3:], cyc_out], dim=1)
                    true_box = torch.cat([target_lineal[:, :3], target_lineal[:, 3:], target_ciclico], dim=1)
                    ious = IoU3D(pred_box.unsqueeze(0), true_box.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                    all_iou3d.extend(np.clip(ious, 0, 1).tolist())

                    if visualizer and contador_val % 25 == 1:
                        centro_crop = centro[0].to(device).view(-1)
                        entrada_denorm = [nube.clone().to(device) for nube in entrada]
                        for n in entrada_denorm:
                            n[:, :3] += centro_crop
                        all_points = torch.cat(entrada_denorm, dim=0)
                        pred_xyz = reg_out[:, :3].squeeze(0) + centro_crop
                        pred_dims = reg_out[:, 3:].squeeze(0)
                        pred_yaw = pred_yaw.squeeze(0)
                        gt_xyz = target_lineal[:, :3].squeeze(0) + centro_crop
                        gt_dims = target_lineal[:, 3:].squeeze(0)
                        gt_yaw = target_ciclico.squeeze(0)
                        box_pred_7 = torch.cat([pred_xyz, pred_dims, pred_yaw.view(1)], dim=0)
                        box_gt_7 = torch.cat([gt_xyz, gt_dims, gt_yaw.view(1)], dim=0)
                        visualizer.plot(all_points, pred_box=box_pred_7, target_box=box_gt_7,
                                        filename=f"img{contador_val}_{np.clip(ious, 0, 1).tolist()}.png")

                    contador_val += 1

            except Exception as e:
                logger.exception(e)
                continue

    avg_loss = running_loss / num_batches if num_batches > 0 else 0

    if not is_train:
        iou_threshold = 0.5
        tp = fp = fn = 0
        for pred_cls, true_cls, iou in zip(all_preds, all_targets, all_iou3d):
            if pred_cls == true_cls:
                if iou >= iou_threshold:
                    tp += 1
                else:
                    fn += 1
            else:
                fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = np.mean(all_iou3d) if all_iou3d else 0.0
        return avg_loss, precision, recall, f1, mean_iou
    
    if is_train and trace_logger is not None:
        mean_class_loss = total_class_loss / num_batches if num_batches > 0 else 0.0
        mean_lineal_loss = total_lineal_loss / num_batches if num_batches > 0 else 0.0
        mean_ciclico_loss = total_ciclico_loss / num_batches if num_batches > 0 else 0.0
        trace_logger.info("--------------------------------------")
        trace_logger.info(f"Loss media clase: {mean_class_loss:.6f}")
        trace_logger.info(f"Loss media lineal: {mean_lineal_loss:.6f}")
        trace_logger.info(f"Loss media cíclica: {mean_ciclico_loss:.6f}")
        trace_logger.info("--------------------------------------")

    return avg_loss


