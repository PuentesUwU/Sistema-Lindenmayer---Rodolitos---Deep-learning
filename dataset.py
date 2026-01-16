# -*- coding: utf-8 -*-
"""
Created on Thu May 22 02:10:19 2025

@author: Julian
"""
import re
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class LindenmayerDataset(Dataset):
    def __init__(self, data_dir, start=0, count=None, modo="iteraciones"):
        self.data_dir = data_dir
        self.start = start
        self.count = count
        self.modo = modo

        self.transforms = transforms.Compose([
            transforms.Grayscale(),  # convierte a 1 canal
            transforms.ToTensor(),  # convierte a [0,1]
        ])

        # Lista de carpetas ordenadas (0, 1, 2, ...)
        todas = sorted([
            os.path.join(self.data_dir, carpeta)
            for carpeta in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, carpeta))
        ])
        if count is None:
            self.carpetas = todas[start:]
        else:
            self.carpetas = todas[start:start+count]

    def __len__(self):
        return len(self.carpetas)

    def __getitem__(self, idx):
        carpeta = self.carpetas[idx]
        img_mayor = Image.open(os.path.join(carpeta, "mayor.png"))
        img_menor = Image.open(os.path.join(carpeta, "menor.png"))

        # Procesamos y unimos en un solo tensor de 2 canales
        tensor_mayor = self.transforms(img_mayor)
        tensor_menor = self.transforms(img_menor)
        imagen = torch.cat([tensor_mayor, tensor_menor], dim=0)  # (2, H, W)

        # Cargar y parsear l_system.txt
        path_ = os.path.join(carpeta, "l_system.txt")
        with open(path_, "r") as f:
            linea = f.read().strip()
            
            
        matches = re.match(r"\[(\d+)\]\[(\d+\.\d+)\]\[(\d+\.\d+),(\d+\.\d+)\]([A-F]+)", linea)
        if not matches:
            raise ValueError(f"No se pudo parsear l_system.txt en {path_}: formato inválido")

        # Formato esperado: "3[0.5][45,90]ABCDEF"
        try:
            iteraciones = int(matches.group(1))
            dpi = float(matches.group(2))
            angulos=(float(matches.group(3)),float(matches.group(4)))
            letras = matches.group(5)
        except Exception as e:
            raise ValueError(f"Error al parsear l_system.txt en {carpeta}: {e}")

        if self.modo == "iteraciones":
            etiqueta = iteraciones
        elif self.modo == "dpi":
            etiqueta = dpi
        elif self.modo == "angulos":
            etiqueta = torch.tensor(angulos, dtype=torch.float32)
        elif self.modo == "letras":
            # Codificamos A–F como 0–5, y FALTANTES como -1 (relleno)
            letras_codificadas = [-1]*10
            for i, letra in enumerate(letras[:10]):
                letras_codificadas[i] = ord(letra) - ord('A')
            etiqueta = torch.tensor(letras_codificadas, dtype=torch.long)
        else:
            raise ValueError(f"Modo no reconocido: {self.modo}")

        return imagen, etiqueta
    

class IteracionesDataset(LindenmayerDataset):
    def __init__(self, data_dir, start=0, count=None):
        super().__init__(data_dir, start, count, modo="iteraciones")

class DpiDataset(LindenmayerDataset):
    def __init__(self, data_dir, start=0, count=None):
        super().__init__(data_dir, start, count, modo="dpi")

class AngulosDataset(LindenmayerDataset):
    def __init__(self, data_dir, start=0, count=None):
        super().__init__(data_dir, start, count, modo="angulos")

class LetrasDataset(LindenmayerDataset):
    def __init__(self, data_dir, start=0, count=None):
        super().__init__(data_dir, start, count, modo="letras")