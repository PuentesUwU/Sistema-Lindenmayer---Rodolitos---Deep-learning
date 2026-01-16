# -*- coding: utf-8 -*-
"""
Created on Thu May 22 01:25:05 2025

@author: Julian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DpiDataset

class DpiCNN(nn.Module):
    def __init__(self):
        super(DpiCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)  # 2 canales: mayor + menor
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 128)  
        self.fc2 = nn.Linear(128, 1)  # salida escalar continua para dpi

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, H/4, W/4)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 64, H/8, W/8)
        x = self.global_pool(x)          # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)        # (B, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # salida sin activación para regresión

        return x.squeeze(1)  # salida (B,)


def entrenar_dpi(n_datos, epocas, batch_size, data_dir="./data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset = DpiDataset(data_dir, count=n_datos)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    modelo = DpiCNN().to(device)
    criterio = nn.MSELoss()  # Error cuadrático medio para regresión
    optimizador = optim.Adam(modelo.parameters(), lr=0.001)

    modelo.train()
    for epoca in range(epocas):
        running_loss = 0.0
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.float().to(device)

            optimizador.zero_grad()
            outputs = modelo(imgs)
            loss = criterio(outputs, targets)
            loss.backward()
            optimizador.step()

            running_loss += loss.item() * imgs.size(0)

        loss_promedio = running_loss / len(dataset)
        print(f"Época {epoca+1}/{epocas} - Loss: {loss_promedio:.6f}")

    # Guardar el modelo entrenado
    torch.save(modelo.state_dict(), "modelos/modelo_dpi.pth")
    print("Modelo DPI guardado en modelo_dpi.pth")
