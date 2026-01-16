# -*- coding: utf-8 -*-
"""
Created on Thu May 22 01:26:03 2025

@author: Julian
"""
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from train.letras import LetrasCNN
from train.iteraciones import ModeloIteraciones
from train.dpi import DpiCNN
from train.angulos import AngulosCNN

def predecir_iteraciones(imagen):
    """
    Realiza una predicción del número de iteraciones a partir de un tensor de imagen.

    Args:
        imagen (Tensor): Tensor de entrada de forma (2, 400, 400)
        model_path (str): Ruta al modelo entrenado (.pth)

    Returns:
        int: Predicción de iteraciones
    """
    model_path="modelos/modelo_iteraciones.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar modelo
    modelo = ModeloIteraciones().to(device)
    modelo.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    modelo.eval()

    # Preparar imagen para batch
    imagen = imagen.unsqueeze(0).to(device)  # (1, 2, 400, 400)

    with torch.no_grad():
        salida = modelo(imagen)  # salida debería tener forma (1, 1) o similar
        prediccion = salida.item()  # Convertir tensor a float

    return int(round(prediccion))

def predecir_dpi(imagen):
    """
    Realiza una predicción del número de iteraciones a partir de un tensor de imagen.

    Args:
        imagen (Tensor): Tensor de entrada de forma (2, 400, 400)
        model_path (str): Ruta al modelo entrenado (.pth)

    Returns:
        int: Predicción de iteraciones
    """
    model_path="modelos/modelo_dpi.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar modelo
    modelo = DpiCNN().to(device)
    modelo.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    modelo.eval()

    # Preparar imagen para batch
    imagen = imagen.unsqueeze(0).to(device)  # (1, 2, 400, 400)

    with torch.no_grad():
        salida = modelo(imagen)  # salida debería tener forma (1, 1) o similar
        prediccion = salida.item()  # Convertir tensor a float

    return round(prediccion,3)

def predecir_angulos(imagen):
    """
    Predice dos ángulos a partir de una imagen tensorial.

    Args:
        imagen (Tensor): Tensor de entrada de forma (2, 400, 400)
        model_path (str): Ruta al modelo entrenado (.pth)

    Returns:
        tuple: Predicción de dos ángulos (float, float)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path="modelos/modelo_angulos.pth"
    # Cargar modelo
    modelo = AngulosCNN().to(device)
    modelo.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    modelo.eval()

    # Añadir dimensión de batch
    imagen = imagen.unsqueeze(0).to(device)  # (1, 2, 400, 400)

    with torch.no_grad():
        salida = modelo(imagen)  # salida de forma (1, 2)
        angulos = salida.squeeze(0).cpu().tolist()  # [angulo1, angulo2]
    print(angulos)
    angulos = tuple(round(x, 3) for x in tuple(angulos))
    print(angulos)
    return angulos


def predecir_letras(imagen):
    """
    Predice una secuencia de letras (A-F) a partir de una imagen.

    Args:
        imagen (Tensor): Imagen tensorial (2, 400, 400)
        model_path (str): Ruta al archivo del modelo
        max_len (int): Longitud máxima de secuencia
        num_classes (int): Cantidad de clases (7 incluyendo padding)

    Returns:
        str: Secuencia de letras predicha (ej. "ACF")
    """
    num_classes=7
    max_len=10
    model_path="modelos/modelo_letras.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo = LetrasCNN(max_len=max_len, num_classes=num_classes).to(device)
    modelo.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    modelo.eval()

    imagen = imagen.unsqueeze(0).to(device)  # (1, 2, 400, 400)

    with torch.no_grad():
        salida = modelo(imagen)  # (1, max_len, num_classes)
        predicciones = torch.argmax(salida, dim=2)  # (1, max_len)
        indices = predicciones.squeeze(0).tolist()  # lista de int

    # Mapeo de índices (0 a 6) -> letras A a F (padding es -1 o 6 si usas num_classes=7)
    mapa_letras = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
    letras = ''.join([mapa_letras[i] for i in indices if i in mapa_letras])

    return letras

def procesar_y_predecir():
    """
    Recorre las carpetas dentro de data_dir, carga las imágenes 'mayor.png' y 'menor.png',
    realiza predicciones con modelos previamente definidos, y guarda los resultados
    en archivos de texto individuales dentro de cada carpeta.
    """
    data_dir = "./data_pre"
    
    
    # Verificar existencia de la carpeta
    if not os.path.isdir(data_dir):
        print("⚠ La carpeta 'data_pre' está vacía o no existe.")
        print("👉 Por favor créala y suministra las imágenes 'mayor.png' y 'menor.png' para la correcta predicción.")
        return

    if len(os.listdir(data_dir)) == 0:
        print("⚠ La carpeta 'data_pre' está vacía.")
        print("👉 Por favor suministra las imágenes para la correcta predicción.")
        return
    
    # Transforms para las imágenes
    transform = transforms.Compose([
        transforms.Resize((400, 400)),   # Ajustar al tamaño usado en el entrenamiento
        transforms.ToTensor()
    ])

    # Itera sobre cada carpeta en data_dir
    for nombre_carpeta in os.listdir(data_dir):
        ruta_carpeta = os.path.join(data_dir, nombre_carpeta)

        # Verifica que sea una carpeta
        if not os.path.isdir(ruta_carpeta):
            continue

        try:
            # Cargar imágenes
            img_mayor = Image.open(os.path.join(ruta_carpeta, "mayor.png")).convert("L")
            img_menor = Image.open(os.path.join(ruta_carpeta, "menor.png")).convert("L")

            # Aplicar transformaciones
            tensor_mayor = transform(img_mayor)
            tensor_menor = transform(img_menor)

            # Unir en un tensor de 2 canales
            imagen = torch.cat([tensor_mayor, tensor_menor], dim=0)  # Añade batch dimension

            # Llamar funciones de predicción
            iteraciones = predecir_iteraciones(imagen)
            dpi = predecir_dpi(imagen)
            angulo1, angulo2 = predecir_angulos(imagen)
            letras = predecir_letras(imagen)

            # Formatear predicción
            prediccion = f"[{iteraciones}][{dpi}][{angulo1},{angulo2}]{letras}"

            # Guardar predicción en archivo
            ruta_archivo = os.path.join(ruta_carpeta, "prediccion.txt")
            with open(ruta_archivo, "w") as f:
                f.write(prediccion)

            print(f"✔ Predicción guardada en {ruta_archivo}")

        except Exception as e:
            print(f"⚠ Error procesando carpeta '{nombre_carpeta}': {e}")

if __name__ == "__main__":
    procesar_y_predecir()
    
#jummm