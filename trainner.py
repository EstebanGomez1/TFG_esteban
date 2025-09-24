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

def confusion_matrix(all_preds, all_targets, all_iou3d, iou_thr=0.5):
    tp = fp = fn = tn = 0
    for pred, true, iou in zip(all_preds, all_targets, all_iou3d):
        if pred == true:
            if iou >= iou_thr:
                tp += 1
            else:
                fp += 1
        else:
            if iou >= iou_thr:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn

def metricas_clase(all_preds, all_targets, all_iou3d, num_classes, class_to_idx):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    counts = np.zeros(num_classes, dtype=int)
    sum_iou = np.zeros(num_classes, dtype=float)
    correct_class = np.zeros(num_classes, dtype=int)

    for pc, tc, iou in zip(all_preds, all_targets, all_iou3d):
        if tc < 0 or tc >= num_classes:
            continue
        counts[tc] += 1
        iou = float(np.clip(iou, 0.0, 1.0))
        sum_iou[tc] += iou
        if pc == tc:
            correct_class[tc] += 1
    metricas_clase = {}
    for i in range(num_classes):
        if counts[i] > 0:
            iou_media = 100.0 * (sum_iou[i] / counts[i])
            clasificacion = 100.0 * (correct_class[i] / counts[i])
        else:
            iou_media = 0.0
            clasificacion = 0.0
        metricas_clase[idx_to_class.get(i, f"class_{i}")] = {
            "iou_media": round(iou_media, 2),
            "clasificacion": round(clasificacion, 2)
        }
    return metricas_clase

import numpy as np

def metricas_iou2d(all_targets, all_iou2d, num_classes, class_to_idx):
    """
    Calcula la IoU2D media por clase.
    
    all_targets: lista con las clases reales
    all_iou2d: lista con las IoU2D asociadas
    num_classes: número total de clases
    class_to_idx: diccionario {nombre_clase: idx}
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    counts = np.zeros(num_classes, dtype=int)
    sum_iou = np.zeros(num_classes, dtype=float)

    for tc, iou in zip(all_targets, all_iou2d):
        if tc < 0 or tc >= num_classes:
            continue
        counts[tc] += 1
        sum_iou[tc] += float(np.clip(iou, 0.0, 1.0))

    metricas = {}
    for i in range(num_classes):
        if counts[i] > 0:
            iou_media = 100.0 * (sum_iou[i] / counts[i])
        else:
            iou_media = 0.0
        metricas[idx_to_class.get(i, f"class_{i}")] = round(iou_media, 2)

    return metricas

import numpy as np


def media_iou_car_ped_cyc(all_targets, all_iou2d, all_iou3d, class_to_idx):
    """
    Calcula la IoU2D y IoU3D medias globales solo para car, pedestrian y cyclist.
    Devuelve un diccionario con IoU2D_media y IoU3D_media (en %).
    """
    clases_interes = {"car", "pedestrian", "cyclist"}
    idx_interes = {class_to_idx[c] for c in clases_interes}

    iou2d_vals = [np.clip(iou, 0, 1) for t, iou in zip(all_targets, all_iou2d) if t in idx_interes]
    iou3d_vals = [np.clip(iou, 0, 1) for t, iou in zip(all_targets, all_iou3d) if t in idx_interes]

    mean_2d = 100.0 * np.mean(iou2d_vals) if iou2d_vals else 0.0
    mean_3d = 100.0 * np.mean(iou3d_vals) if iou3d_vals else 0.0

    return {
        "IoU2D_media": round(mean_2d, 2),
        "IoU3D_media": round(mean_3d, 2),
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
    all_iou2d = []

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
                
                sin, cos = cyc_out[:, 0], cyc_out[:, 1]
                norm = torch.sqrt(sin**2 + cos**2 + 1e-6)
                sin = sin / norm
                cos = cos / norm
                pred_yaw = torch.atan2(cyc_out[:, 0], cyc_out[:, 1])
                loss_ciclico = criterion_yaw(pred_yaw, target_ciclico)

                gamma, alpha, beta = 0.5, 1.25, 1.75 #para 
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

                    pred_box = torch.cat([reg_out[:, :3], reg_out[:, 3:], pred_yaw.view(-1, 1)], dim=1)
                    true_box = torch.cat([target_lineal[:, :3], target_lineal[:, 3:], target_ciclico], dim=1)
                    ious = IoU3D(pred_box.unsqueeze(0), true_box.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                    all_iou3d.extend(np.clip(ious, 0, 1).tolist())
                    pred_box2d = torch.cat([reg_out[:, 0:2], reg_out[:, 3:5], pred_yaw.view(-1, 1)], dim=1)
                    true_box2d = torch.cat([target_lineal[:, 0:2], target_lineal[:, 3:5], target_ciclico], dim=1)
                    res_2d = IoUs2D(pred_box2d.unsqueeze(0), true_box2d.unsqueeze(0))
                    ious2d_tensor = res_2d[0] if isinstance(res_2d, tuple) else res_2d  # <-- clave
                    ious2d = ious2d_tensor.squeeze(0).detach().cpu().numpy()
                    all_iou2d.extend(np.clip(ious2d, 0, 1).tolist())

                    if visualizer and contador_val % 25 == 10:
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
                                        filename=f"img{contador_val}_{epoch_idx}.png")

                    contador_val += 1

            except Exception as e:
                logger.exception(e)
                continue

    avg_loss = running_loss / num_batches if num_batches > 0 else 0
  
    if not is_train:
        metricas_class = metricas_clase(all_preds, all_targets, all_iou3d, 8, class_to_idx)
        metricas_class_2d = metricas_iou2d(all_targets, all_iou2d, num_classes, class_to_idx)
        metricas_globales_cpc = media_iou_car_ped_cyc(all_targets, all_iou2d, all_iou3d, class_to_idx)
        tp, fp, fn, tn = confusion_matrix(all_preds, all_targets, all_iou3d, 0.5)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        trace_logger.info(f"tp: {tp:.6f}")
        trace_logger.info(f"fp: {fp:.6f}")
        trace_logger.info(f"fn: {fn:.6f}")
        trace_logger.info(f"tn: {tn:.6f}") 
        trace_logger.info(f"precision: {precision:.6f}")
        trace_logger.info(f"recall: {recall:.6f}")
        trace_logger.info(f"f1: {f1:.6f}")
        for class_name, stats in metricas_class.items():
            trace_logger.info(
                f"{class_name}: iou_media={stats['iou_media']:.2f}%, "
                f"clasificacion={stats['clasificacion']:.2f}%"
            )
        for class_name, iou_media in metricas_class_2d.items():
            trace_logger.info(f"{class_name}: IoU2D media={iou_media:.2f}%")
        mean_iou2d = np.mean(all_iou2d) if all_iou2d else 0.0
        trace_logger.info(f"iou2d_mean: {mean_iou2d:.6f}")
        mean_iou = np.mean(all_iou3d) if all_iou3d else 0.0
        trace_logger.info(f"iou3d_mean: {mean_iou:.6f}")
        trace_logger.info(
            f"Media global (Car+Pedestrian+Cyclist): "
            f"IoU2D={metricas_globales_cpc['IoU2D_media']:.2f}%, "
            f"IoU3D={metricas_globales_cpc['IoU3D_media']:.2f}%"
        )
        trace_logger.info(f"Saltos: {contador_saltos:.6f}") 
        trace_logger.info("----------------==============------------------")

    if not is_train:
        thr = 0.5
        n = len(all_targets)
        # 1) Accuracy de clase (pred_cls == true_cls)
        class_correct = sum(pc == tc for pc, tc in zip(all_preds, all_targets))
        class_accuracy = class_correct / n if n > 0 else 0.0

        # 2) Accuracy de IoU (iou >= threshold)
        iou_correct = sum(iou >= thr for iou in all_iou3d)
        iou_accuracy = iou_correct / n if n > 0 else 0.0

        # 3) Accuracy de detección (clase correcta + IoU suficiente)
        det_correct = sum(
            (pc == tc) and (iou >= thr)
            for pc, tc, iou in zip(all_preds, all_targets, all_iou3d)
        )
        det_accuracy = det_correct / n if n > 0 else 0.0
        mean_iou = np.mean(all_iou3d) if all_iou3d else 0.0
        return avg_loss, class_accuracy, iou_accuracy, det_accuracy, mean_iou
    

    
    if is_train and trace_logger is not None:
        mean_class_loss = total_class_loss / num_batches if num_batches > 0 else 0.0
        mean_lineal_loss = total_lineal_loss / num_batches if num_batches > 0 else 0.0
        mean_ciclico_loss = total_ciclico_loss / num_batches if num_batches > 0 else 0.0      
        trace_logger.info("--------------------------------------")
        trace_logger.info(f"Loss media clase: {mean_class_loss:.6f}")
        trace_logger.info(f"Loss media lineal: {mean_lineal_loss:.6f}")
        trace_logger.info(f"Loss media cíclica: {mean_ciclico_loss:.6f}") 
        trace_logger.info(f"Saltos: {contador_saltos:.6f}") 
        trace_logger.info("--------------------------------------")

    return avg_loss


