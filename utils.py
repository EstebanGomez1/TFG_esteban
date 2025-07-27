# core/utils.py
import torch
import numpy as np
import random
import time
import logging

def set_seed(seed=None):
    if seed is None:
        seed = int(time.time())

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return seed

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_loggers():
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

    # Logger para seguimiento
    trace_logger = logging.getLogger('trace_logger')
    trace_logger.setLevel(logging.DEBUG)
    trace_handler = logging.FileHandler('seguimiento.log', mode='w')
    trace_formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    trace_handler.setFormatter(trace_formatter)
    trace_logger.addHandler(trace_handler)

    return error_logger, metrics_logger, trace_logger