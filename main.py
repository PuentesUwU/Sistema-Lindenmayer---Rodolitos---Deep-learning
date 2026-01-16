# -*- coding: utf-8 -*-
"""
Created on Tue May 20 16:01:32 2025

@author: Julian
"""
import os
import predict
from predict import procesar_y_predecir

def entrenar_modelos():
    print("\n=== ENTRENAMIENTO DE MODELOS ===")
    try:
        n_datos = int(input("¿Cuántas carpetas usar para entrenar? "))
        epocas = int(input("¿Cuántas épocas quieres entrenar? "))
        batch_size = int(input("¿Qué batch size deseas usar? "))

        # Aquí llamaremos funciones específicas para cada modelo
        from train.iteraciones import entrenar_iteraciones
        from train.dpi import entrenar_dpi
        from train.angulos import entrenar_angulos
        from train.letras import entrenar_letras

        entrenar_iteraciones(n_datos, epocas, batch_size)
        entrenar_dpi(n_datos, epocas, batch_size)
        entrenar_angulos(n_datos, epocas, batch_size)
        entrenar_letras(n_datos, epocas, batch_size)

        print("----Entrenamiento completado.----")
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")

def evaluar_modelos():
    print("\n=== EVALUACIÓN DE MODELOS ===")
    try:
        from evaluate import evaluar_todo
        evaluar_todo()
        print("Evaluación completada.")
    except Exception as e:
        print(f"Error durante la evaluación: {e}")

def usar_modelos():
    print("\n=== USO DE MODELOS CON NUEVAS IMÁGENES ===")
    try:
        path_mayor = input("Ruta de la imagen MAYOR: ")
        path_menor = input("Ruta de la imagen MENOR: ")
        
        from predict import predecir_lsystem
        resultado = predecir_lsystem(path_mayor, path_menor)
        print(f"\n Sistema L predicho: {resultado}")
        
        with open("resultado.txt", "w") as f:
            f.write(resultado)
        print("Guardado en resultado.txt")
    except Exception as e:
        print(f"Error al predecir: {e}")

def menu_principal():
    while True:
        print("\n=== SISTEMA L Deep Learning — CONSOLA ===")
        print("1. Entrenar modelos")
        print("2. Evaluar modelos")
        print("3. Usar modelos con imágenes externas")
        print("4. Salir")
        opcion = input("Elige una opción (1-4): ")

        if opcion == '1':
            try:
                print("\n=== ENTRENAMIENTO DE MODELOS ===")
                entrenar_modelos()
            except Exception as e:
                print(f"Error durante el entrenamiento: {e}")

        elif opcion == '2':
            evaluar_modelos()

        elif opcion == '3':
            procesar_y_predecir()

        elif opcion == '4':
            print("Adiós.")
            break

        else:
            print("Opción no válida.")

if __name__ == "__main__":
    menu_principal()
