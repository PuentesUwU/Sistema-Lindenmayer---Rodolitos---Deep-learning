# -*- coding: utf-8 -*-
"""
Created on Thu May 22 01:24:40 2025

@author: Julian
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LindenmayerDataset  # Asegúrate que esté bien importado

class ModeloIteraciones(nn.Module):
    def __init__(self):
        super(ModeloIteraciones, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(1)

def entrenar_iteraciones(num_datos, epocas, batch_size):
    print("Entrenando modelo de ITERACIONES...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset y DataLoader
    dataset = LindenmayerDataset("./data", start=0, count=num_datos, modo="iteraciones")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    model = ModeloIteraciones().to(device)
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoca in range(epocas):
        total_loss = 0.0

        for batch in dataloader:
            entradas, etiquetas = batch
            
            entradas, etiquetas = entradas.to(device), etiquetas.float().to(device)

            optimizador.zero_grad()
            salidas = model(entradas)
            loss = criterio(salidas, etiquetas)
            loss.backward()
            optimizador.step()

            total_loss += loss.item()

        print(f"Época {epoca+1}/{epocas} - Loss: {total_loss:.4f}")
    os.makedirs("modelos", exist_ok=True)
    torch.save(model.state_dict(), "modelos/modelo_iteraciones.pth")
    print("Modelo de iteraciones guardado en 'modelos/modelo_iteraciones.pth'")