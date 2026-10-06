"""
Trabajo Práctico N° 1 - Procesamiento de Imágenes I (TUIA - FCEIA - UNR)
Año 2026 - 2° Semestre

Problema 1: Ecualización Local de Histograma
=============================================
a. Función para implementar la ecualización local de histograma con ventana MxN.
b. Análisis de la imagen 'Imagen_con_detalles_escondidos.tif' e identificación de objetos ocultos.
c. Estudio de la influencia del tamaño de la ventana (MxN).
"""

import os
import sys

# Auto-detección del entorno virtual local (.venv) si no está activo
for p in [
    os.path.join(os.path.dirname(__file__), ".venv/lib/python3.9/site-packages"),
    os.path.join(os.path.dirname(__file__), "../.venv/lib/python3.9/site-packages"),
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, os.path.abspath(p))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def ecualizacion_local_histograma(img, ksize=(31, 31), border_type=cv2.BORDER_REPLICATE):
    """
    Aplica ecualización local de histograma sobre una imagen en escala de grises.
    
    Parámetros:
    -----------
    img : np.ndarray (uint8)
        Imagen bidimensional de entrada en escala de grises.
    ksize : tuple (int, int)
        Tamaño de la ventana local de procesamiento (M, N).
    border_type : int
        Tipo de extrapolación de bordes para cv2.copyMakeBorder (por defecto cv2.BORDER_REPLICATE).
        
    Retorna:
    --------
    img_out : np.ndarray (uint8)
        Imagen con el histograma ecualizado localmente.
    """
    assert len(img.shape) == 2, "La imagen debe ser monocromática (2D)."
    M, N = ksize
    pad_y = M // 2
    pad_x = N // 2
    
    # 1. Padding con replicación de borde
    padded = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, border_type)
    
    H, W = img.shape
    total_pixels = float(M * N)
    img_out = np.zeros((H, W), dtype=np.uint8)
    
    # 2. Desplazamiento de la ventana píxel a píxel
    for i in range(H):
        for j in range(W):
            patch = padded[i:i+M, j:j+N]
            # Histograma local
            hist, _ = np.histogram(patch, bins=256, range=[0, 256])
            # Función de distribución acumulada (CDF)
            cdf = hist.cumsum()
            val_centro = patch[pad_y, pad_x]
            # Mapeo ecualizado al rango [0, 255]
            img_out[i, j] = np.uint8(np.round((cdf[val_centro] / total_pixels) * 255.0))
            
    return img_out


def ejecutar_analisis_problema1(custom_image_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if custom_image_path and os.path.exists(custom_image_path):
        img_path = custom_image_path
    else:
        img_path = os.path.join(base_dir, "Imagen_con_detalles_escondidos.tif")
        if not os.path.exists(img_path):
            img_path = os.path.join(base_dir, "Imagen_con_objetos_ocultos.tiff")
        if not os.path.exists(img_path) and custom_image_path:
            img_path = custom_image_path
        
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo encontrar la imagen en {img_path}")
        
    print("=" * 60)
    print("PROBLEMA 1: ECUALIZACIÓN LOCAL DE HISTOGRAMA")
    print("=" * 60)
    print(f"Dimensiones de la imagen: {img.shape}")
    print(f"Rango de intensidades original: Min={img.min()}, Max={img.max()}")
    
    # 1. Ecualización Global para comparación
    eq_global = cv2.equalizeHist(img)
    
    # 2. Ecualización Local con diferentes tamaños de ventana
    ventanas = [(7, 7), (15, 15), (31, 31), (65, 65), (101, 101)]
    resultados_locales = {}
    
    print("\nProcesando ecualizaciones locales para diferentes ventanas...")
    for k in ventanas:
        t0 = time.time()
        res = ecualizacion_local_histograma(img, ksize=k)
        dt = time.time() - t0
        resultados_locales[k] = res
        print(f" - Ventana {k[0]}x{k[1]}: tiempo transcurrido = {dt:.2f} s")
        
    # Guardar resultados
    out_dir = os.path.join(base_dir, "resultados_problema1")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "original.png"), img)
    cv2.imwrite(os.path.join(out_dir, "ecualizacion_global.png"), eq_global)
    for k, res in resultados_locales.items():
        cv2.imwrite(os.path.join(out_dir, f"ecualizacion_local_{k[0]}x{k[1]}.png"), res)
        
    # Graficar comparativa
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("a) Imagen Original")
    plt.axis("off")
    
    plt.subplot(2, 3, 2)
    plt.imshow(eq_global, cmap="gray")
    plt.title("b) Ecualización Global (cv2.equalizeHist)")
    plt.axis("off")
    
    for idx, k in enumerate([(7, 7), (15, 15), (31, 31), (65, 65)]):
        plt.subplot(2, 3, idx + 3)
        plt.imshow(resultados_locales[k], cmap="gray")
        plt.title(f"Local Ventana {k[0]}x{k[1]}")
        plt.axis("off")
        
    plt.tight_layout()
    comparativa_path = os.path.join(out_dir, "comparativa_ventanas.png")
    plt.savefig(comparativa_path, dpi=200)
    plt.close()
    print(f"\nGráfica comparativa guardada en: {comparativa_path}")
    
    # 3. Extracción de los 5 ROIs con zoom exacto sobre cada recuadro (62x62 px)
    eq_optima = resultados_locales.get((31, 31), list(resultados_locales.values())[0])
    m = 2  # Margen estético de 2 píxeles
    rois = {
        "roi_1_cuadrado_concentrico": (eq_optima[6-m:68+m, 6-m:68+m], "1. Cuadrado Concéntrico (Sup. Izq.)"),
        "roi_2_linea_diagonal_45":    (eq_optima[6-m:68+m, 187-m:249+m], "2. Línea Diagonal a 45° (Sup. Der.)"),
        "roi_3_letra_a":              (eq_optima[97-m:159+m, 97-m:159+m], "3. Letra 'a' (Centro)"),
        "roi_4_lineas_paralelas":     (eq_optima[188-m:250+m, 6-m:68+m], "4. Líneas Paralelas (Inf. Izq.)"),
        "roi_5_circulo_relleno":      (eq_optima[188-m:250+m, 187-m:249+m], "5. Círculo Relleno (Inf. Der.)"),
    }
    
    # Guardar cada ROI individualmente
    for fname, (crop, _) in rois.items():
        cv2.imwrite(os.path.join(out_dir, f"{fname}.png"), crop)
        
    # Guardar panel con los 5 ROIs ampliados
    plt.figure(figsize=(18, 4))
    for idx, (_, (crop, titulo)) in enumerate(rois.items()):
        plt.subplot(1, 5, idx + 1)
        plt.imshow(crop, cmap="gray")
        plt.title(titulo, fontsize=10, weight="bold")
        plt.axis("off")
    plt.tight_layout()
    zoom_path = os.path.join(out_dir, "zoom_5_objetos_ocultos.png")
    plt.savefig(zoom_path, dpi=200)
    plt.close()
    print(f"Panel con zoom de los 5 objetos guardado en: {zoom_path}")
    
    # Detalle de los objetos descubiertos
    print("\n" + "=" * 60)
    print("DETALLES OCULTOS IDENTIFICADOS EN CADA ZONA:")
    print("=" * 60)
    print("1. Zona Superior Izquierda : Cuadrado concéntrico más claro.")
    print("2. Zona Superior Derecha   : Línea diagonal a 45°.")
    print("3. Zona Central            : Letra 'a'.")
    print("4. Zona Inferior Izquierda : Cuatro líneas horizontales paralelas.")
    print("5. Zona Inferior Derecha   : Círculo (disco relleno).")
    print("6. Fondo de la imagen      : Ruido granular / impulsivo (sal y pimienta) amplificado.")
    print("=" * 60)
    print("\nCONCLUSIONES SOBRE LA INFLUENCIA DEL TAMAÑO DE LA VENTANA (MxN):")
    print("- Ventana muy pequeña (7x7):")
    print("  Sobre-amplifica el ruido en áreas casi uniformes y genera halos y artefactos excesivos.")
    print("- Ventana intermedia (31x31):")
    print("  Tamaño óptimo. El contraste local se expande adecuadamente destacando con nitidez")
    print("  cada objeto oculto y minimizando la distorsión de los bordes del recuadro.")
    print("- Ventana grande (65x65 o mayor):")
    print("  Al aumentar demasiado, el comportamiento converge gradualmente al de la ecualización")
    print("  global, perdiendo capacidad de revelar detalles que ocupan áreas reducidas frente al fondo.")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Problema 1: Ecualización Local de Histograma")
    parser.add_argument("--image", type=str, default=None, help="Ruta a la imagen de entrada (por defecto Imagen_con_detalles_escondidos.tif)")
    args = parser.parse_args()
    ejecutar_analisis_problema1(args.image)
