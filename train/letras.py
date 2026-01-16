# -*- coding: utf-8 -*-
"""
Created on Thu May 22 01:25:27 2025

@author: Julian
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import LetrasDataset  # Dataset específico para letras codificadas

class LetrasCNN(nn.Module):
    def __init__(self, max_len=10, num_classes=7):
        super(LetrasCNN, self).__init__()
        self.max_len = max_len
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, max_len * num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_pool(x)           # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)         # (B, 128)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        # Salida: (batch_size, max_len, num_classes)
        return x.view(-1, self.max_len, self.num_classes)
    
def entrenar_letras(n_datos, epocas, batch_size, data_dir="./data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset = LetrasDataset(data_dir, count=n_datos)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    modelo = LetrasCNN().to(device)
    criterio = nn.CrossEntropyLoss(ignore_index=-1)  # Clasificación múltiple por posición
    optimizador = optim.Adam(modelo.parameters(), lr=0.001)

    modelo.train()
    for epoca in range(epocas):
        running_loss = 0.0
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)  # targets shape: (batch_size, max_len)

            optimizador.zero_grad()
            outputs = modelo(imgs)  # (batch_size, max_len, num_classes)

            # CrossEntropyLoss espera (N,C) y targets (N) así que aplanamos
            outputs_reshaped = outputs.view(-1, modelo.num_classes)  # (batch_size*max_len, num_classes)
            targets_reshaped = targets.view(-1)  # (batch_size*max_len)

            loss = criterio(outputs_reshaped, targets_reshaped)
            loss.backward()
            optimizador.step()

            running_loss += loss.item() * imgs.size(0)

        loss_promedio = running_loss / len(dataset)
        print(f"Época {epoca+1}/{epocas} - Loss: {loss_promedio:.6f}")

    torch.save(modelo.state_dict(), "modelos/modelo_letras.pth")
    print("Modelo Letras guardado en modelo_letras.pth")