# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 12:56:35 2025

@author: Julian
"""
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from skimage.transform import resize
from PIL import Image, ImageChops
import numpy as np
import trimesh
import math
import os
import io
from io import BytesIO
import re
import random
import base64
from main import menu_principal
from pathlib import Path



def mismo_lado(nuevo_vertice, centro_masa, face_i,mesh):
    # Obtener vectores del plano
    p1,p2,p3 = mesh.vertices[face_i[0]].tolist(),mesh.vertices[face_i[1]].tolist(),mesh.vertices[face_i[2]].tolist()
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    
    # Calcular la normal del plano
    normal = np.cross(v1, v2)
    
    # Evaluar los signos para ambos puntos
    signo_x = np.dot(normal, np.array(nuevo_vertice) - np.array(p1))
    signo_y = np.dot(normal, np.array(centro_masa) - np.array(p1))
    
    # Devuelve True si están en el mismo lado, False si están en lados opuestos
    return (signo_x * signo_y) >= 0  

def booleano_crece_o_disminuye(mesh, centroid, nuevo_vertice,face_i):
    centro_masa = mesh.center_mass
    if mismo_lado(nuevo_vertice,centro_masa,face_i,mesh)==False:
        
        vertices = mesh.vertices
        faces=mesh.faces
        direccion = nuevo_vertice - centroid
        longitud = np.linalg.norm(direccion)
        direccion /= longitud  # Normalizar dirección
        epsilon = 1e-6
        for face in faces:
            v0, v1, v2 = vertices[face]
            
            # Möller–Trumbore optimizado
            edge1, edge2 = v1 - v0, v2 - v0
            h = np.cross(direccion, edge2)
            a = np.dot(edge1, h)
            
            if -epsilon < a < epsilon:
                continue  # Rayo paralelo, no hay intersección
                
            f = 1.0 / a
            s = centroid - v0
            u = f * np.dot(s, h)
                
            if u < 0.0 or u > 1.0:
                    continue
                
            q = np.cross(s, edge1)
            v = f * np.dot(direccion, q)

            if v < 0.0 or u + v > 1.0:
                continue
            
            t = f * np.dot(edge2, q)
            if epsilon < t < longitud:
                return False  # La línea atraviesa el objeto
        return True
    return False    



def transformacion_1 (mesh, dpi, light):
    nuevas_caras=[]
    caras_eliminadas=[]
    nuevos_vertices=[]
    for i in mesh.faces:
        vertices_cara = np.array([mesh.vertices[i[0]].tolist(),
                             mesh.vertices[i[1]].tolist(),
                             mesh.vertices[i[2]].tolist()])
        centroid = np.mean(vertices_cara, axis=0)
        nuevo_vertice = [centroid[0]+(dpi*np.cos(light[1]*np.cos(light[0])))
                         ,centroid[1]+(dpi*np.sin(light[0]))
                         ,centroid[2]]+(dpi*np.sin(light[1]))
        if booleano_crece_o_disminuye(mesh,centroid,nuevo_vertice,i):
            
            nuevos_vertices.append(nuevo_vertice)
            index_nuevo_vertice = len(mesh.vertices)-1+len(nuevos_vertices)
            new_faces = np.array([
                [i[0], i[1], index_nuevo_vertice],
                [i[1], i[2], index_nuevo_vertice],
                [i[2], i[0], index_nuevo_vertice]])
            
            nuevas_caras.append(new_faces)
            caras_eliminadas.append(i.tolist())
    nuevas_caras = np.vstack(nuevas_caras)
    mask = np.ones(len(mesh.faces), dtype=bool)
    for face in caras_eliminadas:
        mask &= ~np.all(mesh.faces == face, axis=1)
    mesh.faces = mesh.faces[mask]
    mesh.vertices = np.vstack([mesh.vertices, nuevos_vertices])
    mesh.faces = np.vstack([mesh.faces, nuevas_caras])
    
    return



def transformacion_2(mesh, dpi, light):
    nuevas_caras=[]
    caras_eliminadas=[]
    nuevos_vertices=[]
    for i in mesh.faces:
        vertices_cara = np.array([mesh.vertices[i[0]].tolist(),
                             mesh.vertices[i[1]].tolist(),
                             mesh.vertices[i[2]].tolist()])
        centroid = np.mean(vertices_cara, axis=0)
        centro_masa = mesh.center_mass
        vertice_luz = [centroid[0]+(dpi*np.cos(light[1]*np.cos(light[0])))
                         ,centroid[1]+(dpi*np.sin(light[0]))
                         ,centroid[2]]+(dpi*np.sin(light[1]))
        nuevo_vertice = centroid + ((np.dot(vertice_luz - centroid, centro_masa - centroid) / np.dot(centro_masa - centroid, centro_masa - centroid)) * (centro_masa - centroid))

        """nuevo_vertice = (centroid + (dpi * (centroid- centro_masa) / np.linalg.norm(centroid- centro_masa)))"""
        if booleano_crece_o_disminuye(mesh,centroid,nuevo_vertice,i):
            nuevos_vertices.append(nuevo_vertice)
            index_nuevo_vertice = len(mesh.vertices)-1+len(nuevos_vertices)
            new_faces = np.array([
                [i[0], i[1], index_nuevo_vertice],
                [i[1], i[2], index_nuevo_vertice],
                [i[2], i[0], index_nuevo_vertice]])
            nuevas_caras.append(new_faces)
            caras_eliminadas.append(i.tolist())
    nuevas_caras = np.vstack(nuevas_caras)
    mask = np.ones(len(mesh.faces), dtype=bool)
    for face in caras_eliminadas:
        mask &= ~np.all(mesh.faces == face, axis=1)
    mesh.faces = mesh.faces[mask]
    mesh.vertices = np.vstack([mesh.vertices, nuevos_vertices])
    mesh.faces = np.vstack([mesh.faces, nuevas_caras])
    
    return

def transformacion_3(mesh, dpi, light):
    direccion = np.array([
        np.cos(light[1]*np.cos(light[0])),  # Componente X
        np.sin(light[0]),  # Componente Y
        np.sin(light[1])                   # Componente Z
    ])
    new_mesh = mesh
    for i in range(len(mesh.vertices)):
        nuevo_vertice = mesh.vertices[i] + (dpi * direccion)
        if booleano_vertices(mesh,nuevo_vertice):  # Aplicar solo si el nuevo vértice cumple la condición
            new_mesh.vertices[i] = nuevo_vertice
        
    # Aplicar desplazamiento
    
    mesh.vertices = new_mesh.vertices
    return

def booleano_vertices(mesh, nuevo_vertice):
    return not mesh.contains([nuevo_vertice])[0]

def suavizar_picos(mesh,centro_masa):
    distancias = np.linalg.norm(mesh.vertices - centro_masa, axis=1)
    promedio = np.mean(distancias)
    for i in range(0,len(mesh.vertices)):
        distancia = np.linalg.norm(mesh.vertices[i] - mesh.center_mass)
        if distancia<promedio:
            direccion = (mesh.vertices[i] - mesh.center_mass) / np.linalg.norm(mesh.vertices[i] - mesh.center_mass)
            nuevo_vertice = mesh.center_mass + direccion *promedio
            mesh.vertices[i]=nuevo_vertice
    return

def agrandar(mesh,centro_masa,dpi):
    centro_masa = mesh.center_mass
    distancias = np.linalg.norm(mesh.vertices - centro_masa, axis=1)
    promedio = np.mean(distancias)
    for i in range(0,len(mesh.vertices)):
        distancia = np.linalg.norm(mesh.vertices[i] - mesh.center_mass)
        direccion = (mesh.vertices[i] - mesh.center_mass) / np.linalg.norm(mesh.vertices[i] - mesh.center_mass)
        nuevo_vertice = mesh.center_mass + (direccion * (distancia +dpi))
        mesh.vertices[i]=nuevo_vertice
    return

def transformacion_4(mesh,dpi,light):
    centro_masa = mesh.center_mass
    direccion = np.array([
        np.cos(light[1]*np.cos(light[0])),  # Componente X
        np.sin(light[0]),  # Componente Y
        np.sin(light[1])                   # Componente Z
    ])
    new_mesh = mesh
    centro = mesh.center_mass
    theta, phi = light

    # Dirección de la luz como vector unitario
    dir_luz = np.array([
        np.cos(theta) * np.cos(phi),
        np.sin(theta) * np.cos(phi),
        np.sin(phi)
    ])

    nuevos_vertices = mesh.vertices.copy()

    for i, v in enumerate(mesh.vertices):
        v = np.array(v)
        dir_centro_a_v = v - centro
        dir_centro_a_v_unit = dir_centro_a_v / (np.linalg.norm(dir_centro_a_v) + 1e-8)

        desplazamiento = dpi * dir_luz
        proyeccion = np.dot(desplazamiento, dir_centro_a_v_unit)

        if proyeccion > 0:
            nuevos_vertices[i] = v + desplazamiento

    # Reasignar los vértices a la malla
    mesh.vertices = nuevos_vertices
    return


def visualizar(mallas,output_folder):
    centro_masa = mallas[0].center_mass

    # Obtener ejes principales de la última malla
    bbox = mallas[-1].bounding_box_oriented
    axes = bbox.primitive.transform[:3, :3]
    eje_mayor = axes[:, 0]
    eje_menor = axes[:, 2]

    cortes_largos, cortes_cortos = [], []

    for mesh in mallas:
        # Corte eje mayor
        corte_largo = mesh.section(plane_origin=centro_masa, plane_normal=eje_mayor)
        cortes_largos.append(corte_largo.to_planar()[0] if corte_largo else None)

        # Corte eje menor
        corte_corto = mesh.section(plane_origin=centro_masa, plane_normal=eje_menor)
        cortes_cortos.append(corte_corto.to_planar()[0] if corte_corto else None)

    # Limites globales para mantener la misma escala
    def calcular_limites_globales(cortes):
        min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
        for corte in cortes:
            if corte:
                for entity in corte.entities:
                    puntos = corte.vertices[entity.points]
                    min_x = min(min_x, puntos[:, 0].min())
                    max_x = max(max_x, puntos[:, 0].max())
                    min_y = min(min_y, puntos[:, 1].min())
                    max_y = max(max_y, puntos[:, 1].max())
        margin = 0.05 * max(max_x - min_x, max_y - min_y)
        return min_x - margin, max_x + margin, min_y - margin, max_y + margin

    lims_largos = calcular_limites_globales(cortes_largos)
    lims_cortos = calcular_limites_globales(cortes_cortos)

    def guardar_corte_como_imagen_y_vector(corte, nombre, limites):
        if not corte:
            return
        min_x, max_x, min_y, max_y = limites

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        for entity in corte.entities:
            puntos = corte.vertices[entity.points]
            ax.plot(puntos[:, 0], puntos[:, 1], 'k', linewidth=1)

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.axis('off')
        ax.set_aspect('equal')
        # Guardar el plot en memoria sin escribir en disco
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # Abrir como imagen PIL
        img = Image.open(buf).convert("RGB")
        # Vectorizar con base64
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        ruta = os.path.join(output_folder, f"{nombre}.txt")
        # Guardar el contenido
        with open(ruta, "w", encoding="utf-8") as archivo:
            archivo.write(img_str)
        plt.close()
        
    
    for i, (corte1, corte2) in enumerate(zip(cortes_largos, cortes_cortos)):
        guardar_corte_como_imagen_y_vector(corte1, f"iteracion_{i+1}_mayor", lims_largos)
        guardar_corte_como_imagen_y_vector(corte2, f"iteracion_{i+1}_menor", lims_cortos)

    return output_folder

def parsear_string(texto):
    # Usamos expresiones regulares para extraer la información
    patron = r"\[(\d+)\]\[(\-?\d+(?:\.\d+)?)\]\[(\-?\d+(?:\.\d+)?),\s*(\-?\d+(?:\.\d+)?)\](.*)"
    coincidencia = re.match(patron, texto)
    
    if coincidencia:
        iteraciones = int(coincidencia.group(1))
        dpi = float(coincidencia.group(2))
        light = [float(coincidencia.group(3)), float(coincidencia.group(4))]
        lsystem = coincidencia.group(5).strip()
        print(iteraciones, dpi, light, lsystem)
        return iteraciones, dpi, light, lsystem
    else:
        raise ValueError("El formato del string es incorrecto.")


def combinar_imagenes_base64(imagenes_b64):
    imagen_final = None
    for b64_string in imagenes_b64:
        image_data = base64.b64decode(b64_string)
        img = Image.open(BytesIO(image_data)).convert("L")  # escala de grises
        img = img.point(lambda p: 255 if p > 10 else 0)  # binarización ligera
        if imagen_final is None:
            imagen_final = img
        else:
            imagen_final = ImageChops.darker(imagen_final, img) # combina por superposición
    return imagen_final

def generar_imagenes_combinadas(path_base="data"):
    carpetas = [f for f in os.listdir(path_base) if os.path.isdir(os.path.join(path_base, f))]
    for carpeta in carpetas:
        path_carpeta = os.path.join(path_base, carpeta)
        archivos = os.listdir(path_carpeta)
        mayor_imgs = sorted([f for f in archivos if "_mayor.txt" in f])
        menor_imgs = sorted([f for f in archivos if "_menor.txt" in f])

        if not mayor_imgs or not menor_imgs:
            print(f"{carpeta} no tiene archivos suficientes.")
            continue

        # Leer base64
        imgs_b64_mayor = []
        for archivo in mayor_imgs:
            with open(os.path.join(path_carpeta, archivo), 'r') as f:
                imgs_b64_mayor.append(f.read())

        imgs_b64_menor = []
        for archivo in menor_imgs:
            with open(os.path.join(path_carpeta, archivo), 'r') as f:
                imgs_b64_menor.append(f.read())

        # Combinar imágenes
        img_mayor = combinar_imagenes_base64(imgs_b64_mayor)
        img_menor = combinar_imagenes_base64(imgs_b64_menor)

        # Guardar combinadas en base64
        for nombre, img in [("mayor.txt", img_mayor), ("menor.txt", img_menor)]:
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            b64_result = base64.b64encode(buffer.getvalue()).decode("utf-8")
            with open(os.path.join(path_carpeta, nombre), 'w') as f:
                f.write(b64_result)

        print(f"Generado 'mayor.txt' y 'menor.txt' en: {carpeta}")

def decode_base64_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert('L')  # escala de grises
        return image
    except Exception as e:
        print("Error al decodificar imagen:", e)
        return None

def leer_imagen_combinada(carpeta, corte="mayor", guardar_png=True):

    carpeta = Path(carpeta)
    archivos = sorted([f for f in carpeta.iterdir() if re.match(rf"iteracion_\d+_{corte}\.txt", f.name)])
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos iteracion_*_{corte}.txt en {carpeta}")

    suma_img = None

    for archivo in archivos:
        with open(archivo, "r") as f:
            base64_str = f.read()
        img = decode_base64_image(base64_str)  # Función ya existente
        if img is None:
            raise ValueError(f"No se pudo decodificar imagen en {archivo}")
        
        arr = np.array(img, dtype=np.float32)
        if suma_img is None:
            suma_img = arr
        else:
            suma_img += arr

    # Normalizar a rango 0-255
    suma_img = suma_img / suma_img.max() * 255
    suma_img = suma_img.astype(np.uint8)

    imagen_final = Image.fromarray(suma_img)

    if guardar_png:
        nombre_png = carpeta / f"{corte}.png"
        imagen_final.save(nombre_png)
        print(f"Imagen combinada guardada como: {nombre_png}")

    return imagen_final

def menu_principal_():
    while True:
        print("\n=== SISTEMA L — CONSOLA ===")
        print("1. Generar datos (aleatorio)")
        print("2. Generar datos (manual)")
        print("3. Pasar al menu de deep learning")
        print("4. Salir")

        opcion = input("Elige una opción (1-4): ")

        if opcion == '1':
            try:
                print("\n=== GENERACIÓN DE DATOS (ALEATORIA) ===")
                ejecutar_generar_datos(modo="aleatorio")
            except Exception as e:
                print(f"Error durante generación aleatoria: {e}")

        elif opcion == '2':
            try:
                print("\n=== GENERACIÓN DE DATOS (MANUAL) ===")
                ejecutar_generar_datos(modo="manual")
            except Exception as e:
                print(f"Error durante generación manual: {e}")

        elif opcion == '3':
            menu_principal()

        elif opcion == '4':
            print("Adiós.")
            break

        else:
            print("Opción no válida.")
    
    
def ejecutar_generar_datos(modo="aleatorio"):

    numero_lsystems = int(input("Cantidad de L-systems a generar: "))
    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)

    for i_lsystem in range(numero_lsystems):
        output_folder = os.path.join(base_dir, f"cortes_iterativos_{i_lsystem}")
        os.makedirs(output_folder, exist_ok=True)
        if modo == "manual":
            print("\n--- Parámetros manuales ---")
            iteraciones = int(input("Iteraciones: "))
            dpi = float(input("dpi: "))
            angulo_1 = float(input("Ángulo 1 (rad): "))
            angulo_2 = float(input("Ángulo 2 (rad): "))
            lsystem = input("L-system (ej: ABCDF): ").strip().upper()

        else:  # modo aleatorio
            iteraciones = random.randint(2, 10)
            dpi = round(random.uniform(0.1, 2), 2)
            angulo_1 = round(random.uniform(0, 2 * math.pi), 3)
            angulo_2 = round(random.uniform(0, 2 * math.pi), 3)
            cantidad_letras = random.randint(1, 10)
            lsystem = "".join(
                chr(65 + random.randint(0, 5)) for _ in range(cantidad_letras)
            )

        light = [angulo_1, angulo_2]

        string_final = f"[{iteraciones}][{dpi}][{angulo_1},{angulo_2}]{lsystem}"
        print(f"\nL-System generado: {string_final}")

        # =========================
        # Malla base
        # =========================
        mesh = trimesh.Trimesh(
            vertices=[
                [0, 0, 1], [0.8944, 0, 0.4472], [0.2764, 0.8506, 0.4472],
                [-0.7236, 0.5257, 0.4472], [-0.7236, -0.5257, 0.4472],
                [0.2764, -0.8506, 0.4472], [0.7236, 0.5257, -0.4472],
                [-0.2764, 0.8506, -0.4472], [-0.8944, 0, -0.4472],
                [-0.2764, -0.8506, -0.4472], [0.7236, -0.5257, -0.4472],
                [0, 0, -1]
            ],
            faces=[
                [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
                [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 8], [3, 8, 4],
                [4, 8, 9], [4, 9, 5], [5, 9, 10], [5, 10, 1], [1, 10, 6],
                [11, 7, 6], [11, 8, 7], [11, 9, 8], [11, 10, 9], [11, 6, 10]
            ],
            process=False
        )

        dlight = light.copy()
        mallas = [mesh.copy()]
        
        for _ in range(iteraciones):
            for simbolo in lsystem:
                centro_masa = mesh.center_mass

                if simbolo == "A":
                    transformacion_1(mesh, dpi, light)
                elif simbolo == "B":
                    transformacion_2(mesh, dpi, light)
                elif simbolo == "C":
                    transformacion_3(mesh, dpi, light)
                elif simbolo == "D":
                    transformacion_4(mesh, dpi, light)
                elif simbolo == "E":
                    agrandar(mesh, centro_masa, dpi)
                elif simbolo == "F":
                    suavizar_picos(mesh, centro_masa)

            # Actualizar luz
            light[0] += dlight[0]
            light[1] += dlight[1]

            # Guardar estado
            mallas.append(
                trimesh.Trimesh(
                    vertices=mesh.vertices.copy(),
                    faces=mesh.faces.copy(),
                    process=False
                )
            )

        # =========================
        # Visualización y guardado
        # =========================
        visualizar(mallas, output_folder)
        leer_imagen_combinada(output_folder, "mayor", guardar_png=True)
        leer_imagen_combinada(output_folder, "menor", guardar_png=True)


        ruta = os.path.join(output_folder, f"l_system.txt")
        with open(ruta, "w", encoding="utf-8") as archivo:
            archivo.write(string_final)

        print(f"✔ L-System {i_lsystem + 1} finalizado\n")
    return

if __name__ == "__main__":
    menu_principal_()


