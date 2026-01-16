# -*- coding: utf-8 -*-
"""
Created on Thu May 22 01:25:48 2025

@author: Julian
"""

import os
import torch
from torch.utils.data import DataLoader
from dataset import LetrasDataset
from train.letras import LetrasCNN
import numpy as np
from train.iteraciones import ModeloIteraciones  # Importa la clase del modelo
from dataset import IteracionesDataset          # Asumo que tienes un dataset para iteraciones
from train.dpi import DpiCNN       # Importa la clase del modelo DPI
from dataset import DpiDataset     # Asumo que tienes un dataset para DPI
from train.angulos import AngulosCNN   # Importa el modelo AngulosCNN
from dataset import AngulosDataset     # Dataset para ángulos

def evaluar_angulos(model_path, data_dir, n_eval, batch_size=32 , start=int):
    """
    Evalúa el modelo de ángulos en un conjunto de datos.

    Args:
        model_path (str): Ruta al archivo .pth con pesos del modelo entrenado.
        data_dir (str): Carpeta donde están los datos de evaluación.
        n_eval (int): Número de muestras para evaluación.
        batch_size (int): Tamaño de batch para evaluación.

    Returns:
        float: Error cuadrático medio promedio en el conjunto de evaluación.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_eval = AngulosDataset(data_dir, count=n_eval, start=start)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)

    modelo = AngulosCNN().to(device)
    modelo.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    modelo.eval()

    criterio = torch.nn.MSELoss(reduction='sum')

    loss_total = 0.0
    n_samples = 0

    with torch.no_grad():
        for imgs, targets in dataloader_eval:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = modelo(imgs)
            loss = criterio(outputs, targets)

            loss_total += loss.item()
            n_samples += imgs.size(0)

    mse_promedio = loss_total / n_samples
    print(f"Error cuadrático medio (MSE) promedio Ángulos: {mse_promedio:.6f}")

    return mse_promedio

def evaluar_dpi(model_path, data_dir, n_eval, batch_size=32,start=int):
    """
    Evalúa el modelo DPI en un conjunto de datos.

    Args:
        model_path (str): Ruta al archivo .pth con pesos del modelo entrenado.
        data_dir (str): Carpeta donde están los datos de evaluación.
        n_eval (int): Número de muestras para evaluación.
        batch_size (int): Tamaño de batch para evaluación.

    Returns:
        float: Error cuadrático medio promedio en el conjunto de evaluación.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando DPI en dispositivo: {device}")
    
    dataset_eval = DpiDataset(data_dir, count=n_eval, start=start)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)

    modelo = DpiCNN().to(device)
    modelo.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    modelo.eval()

    criterio = torch.nn.MSELoss(reduction='sum')

    loss_total = 0.0
    n_samples = 0

    with torch.no_grad():
        for imgs, targets in dataloader_eval:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = modelo(imgs)
            loss = criterio(outputs, targets)

            loss_total += loss.item()
            n_samples += imgs.size(0)

    mse_promedio = loss_total / n_samples
    print(f"Error cuadrático medio (MSE) promedio DPI: {mse_promedio:.6f}")

    return mse_promedio

def evaluar_iteraciones(model_path, data_dir, n_eval, batch_size=32,start=int):
    """
    Evalúa el modelo de iteraciones en un conjunto de datos.

    Args:
        model_path (str): Ruta al archivo .pth con pesos del modelo entrenado.
        data_dir (str): Carpeta donde están los datos de evaluación.
        n_eval (int): Número de muestras para evaluación.
        batch_size (int): Tamaño de batch para evaluación.

    Returns:
        float: Error cuadrático medio promedio en el conjunto de evaluación.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando en dispositivo: {device}")

    # Carga el dataset de evaluación
    dataset_eval = IteracionesDataset(data_dir, count=n_eval, start=start)
    print(f"Cantidad de datos en dataset de evaluación: {len(dataset_eval)}")
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)

    # Instancia el modelo y carga pesos
    modelo = ModeloIteraciones().to(device)
    modelo.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    modelo.eval()

    criterio = torch.nn.MSELoss(reduction='sum')  # Para sumar errores y luego promediar

    loss_total = 0.0
    n_samples = 0

    with torch.no_grad():
        for imgs, targets in dataloader_eval:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = modelo(imgs)
            loss = criterio(outputs, targets)

            loss_total += loss.item()
            n_samples += imgs.size(0)

    mse_promedio = loss_total / n_samples
    print(f"Error cuadrático medio (MSE) promedio en evaluación: {mse_promedio:.6f}")

    return mse_promedio

def evaluar_letras(model_path, data_dir, n_datos, batch_size=16, max_len=10, num_classes=7,start=int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar dataset de evaluación
    dataset = LetrasDataset(data_dir, count=n_datos, start=start)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Cargar modelo
    modelo = LetrasCNN(max_len=max_len, num_classes=num_classes).to(device)
    modelo.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    modelo.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)  # (batch_size, max_len)

            outputs = modelo(imgs)  # (batch_size, max_len, num_classes)
            preds = torch.argmax(outputs, dim=2)  # Predicción (batch_size, max_len)

            # Comparamos ignorando los padding (-1)
            mask = targets != -1
            total += mask.sum().item()
            correct += (preds == targets).masked_select(mask).sum().item()

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy total ignorando padding: {accuracy*100:.2f}%")

    
    
    
def evaluar_todo():
    data_dir = "./data"
    # Pregunta cuántos datos evaluar
    n_eval = int(input("Ingrese la cantidad de datos para evaluar: "))
    batch_size = int(input("Ingrese tamaño de batch (default 32): ") or 32)
    
    total_datos = len([d for d in os.listdir(data_dir) if d.startswith("cortes_iterativos_")])
    start_idx = max(0, total_datos - n_eval)
    

    print("\n--- Evaluando modelo Iteraciones ---")
    mse_iter = evaluar_iteraciones(
        model_path="modelos/modelo_iteraciones.pth",
        data_dir=data_dir,
        n_eval=n_eval,
        batch_size=batch_size,
        start=start_idx

    )

    print("\n--- Evaluando modelo DPI ---")
    mse_dpi = evaluar_dpi(
        model_path="Modelos/modelo_dpi.pth",
        data_dir=data_dir,
        n_eval=n_eval,
        batch_size=batch_size,
        start=start_idx

    )

    print("\n--- Evaluando modelo Ángulos ---")
    mse_angulos = evaluar_angulos(
        model_path="modelos/modelo_angulos.pth",
        data_dir=data_dir,
        n_eval=n_eval,
        batch_size=batch_size,
        start=start_idx

    )

    print("\n--- Evaluando modelo Letras ---")
    evaluar_letras(
        model_path="modelos/modelo_letras.pth",
        data_dir=data_dir,
        n_datos=n_eval,
        batch_size=batch_size,
        start=start_idx

    )

    print("\nEvaluación final completa.")
    return