# -*- coding: utf-8 -*-
"""
Created on Thu May 22 02:12:43 2025

@author: Julian
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess import cargar_datos_batch, normalize_image
from dataset import LindenmayerDataset
from train.iteraciones import IteracionesCNN
from train.dpi import DPIModelCNN
from train.angulos import AngulosCNN
from train.letras import LetrasCNN
import os

def train_model(model, dataloader, criterion, optimizer, epochs=5, device='cpu', save_path=None):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Modelo guardado en: {save_path}")

def entrenar_todo(base_path, start_idx=0, batch_size=64, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Usando dispositivo:", device)

    os.makedirs("models", exist_ok=True)

    # === ITERACIONES ===
    dataset_iter = LindenmayerDataset(base_path, start_idx=start_idx, batch_size=batch_size,
                                      target='iteraciones', transform=normalize_image)
    loader_iter = DataLoader(dataset_iter, batch_size=batch_size, shuffle=True)
    model_iter = IteracionesCNN()
    crit_iter = nn.CrossEntropyLoss()
    opt_iter = torch.optim.Adam(model_iter.parameters(), lr=1e-3)
    print("Entrenando modelo: ITERACIONES")
    train_model(model_iter, loader_iter, crit_iter, opt_iter, epochs=epochs, device=device,
                save_path="models/iteraciones.pth")

    # === DPI ===
    dataset_dpi = LindenmayerDataset(base_path, start_idx=start_idx, batch_size=batch_size,
                                     target='dpi', transform=normalize_image)
    loader_dpi = DataLoader(dataset_dpi, batch_size=batch_size, shuffle=True)
    model_dpi = DPIModelCNN()
    crit_dpi = nn.MSELoss()
    opt_dpi = torch.optim.Adam(model_dpi.parameters(), lr=1e-3)
    print("Entrenando modelo: DPI")
    train_model(model_dpi, loader_dpi, crit_dpi, opt_dpi, epochs=epochs, device=device,
                save_path="models/dpi.pth")

    # === ANGULOS ===
    dataset_ang = LindenmayerDataset(base_path, start_idx=start_idx, batch_size=batch_size,
                                     target='angulos', transform=normalize_image)
    loader_ang = DataLoader(dataset_ang, batch_size=batch_size, shuffle=True)
    model_ang = AngulosCNN()
    crit_ang = nn.MSELoss()
    opt_ang = torch.optim.Adam(model_ang.parameters(), lr=1e-3)
    print("Entrenando modelo: ÁNGULOS")
    train_model(model_ang, loader_ang, crit_ang, opt_ang, epochs=epochs, device=device,
                save_path="models/angulos.pth")

    # === LETRAS ===
    dataset_let = LindenmayerDataset(base_path, start_idx=start_idx, batch_size=batch_size,
                                     target='letras_vec', transform=normalize_image)
    loader_let = DataLoader(dataset_let, batch_size=batch_size, shuffle=True)
    model_let = LetrasCNN()
    crit_let = nn.CrossEntropyLoss()
    opt_let = torch.optim.Adam(model_let.parameters(), lr=1e-3)
    print("Entrenando modelo: LETRAS")
    train_model(model_let, loader_let, crit_let, opt_let, epochs=epochs, device=device,
                save_path="models/letras.pth")

def main():
    base_path = "./data"
    entrenar_todo(base_path, start_idx=0, batch_size=64, epochs=5)

if __name__ == "__main__":
    main()