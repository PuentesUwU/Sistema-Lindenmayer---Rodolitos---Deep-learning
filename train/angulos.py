# -*- coding: utf-8 -*-
"""
Created on Thu May 22 01:25:16 2025

@author: Julian
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import AngulosDataset  # Dataset específico para ángulos

class AngulosCNN(nn.Module):
    def __init__(self):
        super(AngulosCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 2)  # Dos salidas: ang1 y ang2

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (2, 128, 128) → (32, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))  # (64, 64, 64) → (64, 32, 32)
        x = self.pool(F.relu(self.conv3(x)))  # (128, 32, 32) → (128, 16, 16)
        x = self.global_pool(x)          # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)        # (B, 128)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
    
def entrenar_angulos(n_datos, epocas, batch_size, data_dir="./data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset = AngulosDataset(data_dir, count=n_datos)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    modelo = AngulosCNN().to(device)
    criterio = nn.MSELoss()  # Regresión para los dos ángulos
    optimizador = optim.Adam(modelo.parameters(), lr=0.001)

    modelo.train()
    for epoca in range(epocas):
        running_loss = 0.0
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizador.zero_grad()
            outputs = modelo(imgs)
            loss = criterio(outputs, targets)
            loss.backward()
            optimizador.step()

            running_loss += loss.item() * imgs.size(0)

        loss_promedio = running_loss / len(dataset)
        print(f"Época {epoca+1}/{epocas} - Loss: {loss_promedio:.6f}")

    torch.save(modelo.state_dict(), "modelos/modelo_angulos.pth")
    print("Modelo Ángulos guardado en modelo_angulos.pth")