"""
Trabajo Práctico N° 1 - Procesamiento de Imágenes I (TUIA - FCEIA - UNR)
Año 2026 - 2° Semestre

Problema 2: Validación de Planilla de Calificaciones
====================================================
a. Lectura de formulario y validación campo por campo por registro (salida por terminal).
b. Generación de imagen de salida con el crop de Nombre y Apellido de alumnos no aprobados
   ('L' o 'R') con registros válidos, indicando su condición.
c. Generación de archivo CSV por planilla con las columnas requeridas (ID, Legajo, Nombre y Apellido,
   Parcial 1, Parcial 2, Parcial 3, Condición Final) y valores OK/MAL.
d. Ejecución cíclica sobre las 4 imágenes (grade_sheet_1.png a grade_sheet_4.png).
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
import csv

def detectar_rejilla_tabla(img_gray):
    """
    Detecta las coordenadas de las líneas divisorias de la tabla (filas y columnas)
    combinando dos técnicas oficiales de Unidad 3:
      - Para filas: Hough probabilístico sobre los bordes Canny.
      - Para columnas: proyección 1D (suma por columnas) sobre un binario
        suavizado. Es matemática de arrays de U1, sin morfología.

    Pipeline:
      1. Suavizado gaussiano + Canny + HoughLinesP para FILAS (líneas
         horizontales largas).
      2. Binarización (Otsu) + suavizado + proyección 1D vertical para
         COLUMNAS (las líneas verticales no son totalmente contínuas en
         la planilla, así que la proyección gana a Hough).
      3. Top-N por histograma para evitar duplicados y detecciones espurias.

    Retorna:
    --------
    rows : list[int] (coordenadas Y de cada línea horizontal)
    cols : list[int] (coordenadas X de cada línea vertical)
    """
    h_img, w_img = img_gray.shape

    # --- ETAPA 1: Filas con Canny + Hough --------------------------------
    f_blur = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=1.5)
    edges = cv2.Canny(f_blur, threshold1=80, threshold2=120)
    min_h_len = int(0.6 * w_img)   # líneas reales cubren >=60% del ancho

    raw_lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=80, minLineLength=max(150, min_h_len // 2),
        maxLineGap=5
    )
    h_lines = []
    if raw_lines is not None:
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Solo horizontales de cobertura suficiente
            if abs(dy) < abs(dx) * 0.18 and abs(dx) >= min_h_len:
                y_mid = (y1 + y2) / 2.0
                h_lines.append(y_mid)

    # --- ETAPA 2: Columnas con proyección 1D ----------------------------
    # Binarizamos la imagen con Otsu y la difuminamos verticalmente para
    # unir los pedazos de cada línea vertical de la tabla en una mancha
    # contínua, después proyectamos sobre el eje X. Esto es matemática
    # de arrays (np.sum) vista en U1 Fundamentos.
    _, bin_planilla = cv2.threshold(f_blur, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Suavizado gaussiano vertical alargado: une trazos verticales
    # rotos sin usar morfología. Es un filtro lineal de U2.
    bin_col = cv2.GaussianBlur(bin_planilla, ksize=(1, 25), sigmaX=0)
    # Proyección: contamos píxeles negros por columna
    c_sum = np.sum(bin_col, axis=0).astype(float)
    if c_sum.max() > 0:
        c_norm = c_sum / c_sum.max()
    else:
        return _agrupar_coords_top_n(h_lines, n_min=18, n_max=28, bin_size=8), []

    # Nos quedamos con picos >= 70% del máximo (las columnas verticales
    # de la tabla concentran masa; el texto solo aporta ruido disperso)
    high = c_norm >= 0.7
    diff_c = np.diff(np.pad(high.astype(int), (1, 1)))
    starts = np.where(diff_c == 1)[0]
    ends = np.where(diff_c == -1)[0]
    col_coords = [int(round((s + e) / 2)) for s, e in zip(starts, ends)]

    # --- ETAPA 3: Agrupar y elegir top-N --------------------------------
    rows = _agrupar_coords_top_n(h_lines, n_min=18, n_max=28, bin_size=8)
    cols = _agrupar_coords_top_n(col_coords, n_min=6, n_max=10, bin_size=8)

    return rows, cols


def _agrupar_coords(coords, bin_size=8):
    """
    Agrupa coordenadas (en píxeles) en bandas de ancho `bin_size` y devuelve
    la coordenada mediana de cada banda. Sirve para colapsar varios segmentos
    de HoughP (que pueden partir una misma línea en 2-3 pedazos) en una sola.

    Si no hay coordenadas, devuelve [].
    """
    if not coords:
        return []
    coords_arr = np.asarray(coords, dtype=float)
    # Proyección: redondeo a la banda más cercana
    bins = np.round(coords_arr / bin_size).astype(int)
    # Moda por banda
    bandas = {}
    for c, b in zip(coords_arr, bins):
        bandas.setdefault(int(b), []).append(c)
    resultado = []
    for b in sorted(bandas.keys()):
        resultado.append(int(round(np.median(bandas[b]))))
    return resultado


def _agrupar_coords_top_n(coords, n_min=8, n_max=30, bin_size=8):
    """
    Agrupa coordenadas cercanas en bandas y devuelve las N bandas con más
    ocurrencias. Pensado para picos de histogramas de Hough/proyección.

    Parámetros:
      coords:  lista de coordenadas en píxeles.
      n_min:   mínimo de bandas a devolver si hay esa cantidad de grupos.
      n_max:   tope absoluto para no devolver cientos de bandas espurias.
      bin_size: ancho de cada banda de agrupamiento.

    Algoritmo:
      1. Redondea cada coordenada a `bin_size` para agruparlas en bandas.
      2. Ordena bandas por cantidad de elementos (moda = más consistente).
      3. Si la cantidad de bandas supera `n_max`, hace un segundo pase:
         las bandas que quedaron muy cerca (distancia < bin_size*4) se
         colapsan en una sola tomando la mediana. Esto elimina los
         duplicados por líneas de la tabla partidas en varios segmentos.
    """
    if not coords:
        return []
    coords_arr = np.asarray(coords, dtype=float)
    bins = np.round(coords_arr / bin_size).astype(int)
    bandas = {}
    for c, b in zip(coords_arr, bins):
        bandas.setdefault(int(b), []).append(c)
    # Moda por banda (mediana)
    modas = {int(b): int(round(np.median(coords_list)))
             for b, coords_list in bandas.items()}
    # Ordenar bandas por frecuencia (desc) y devolver top-N
    bandas_ordenadas = sorted(bandas.items(), key=lambda kv: len(kv[1]), reverse=True)
    n_take = max(n_min, min(n_max, len(bandas_ordenadas)))
    # Tomar las modas de las top-N y ordenarlas por coordenada
    candidatos = sorted(int(round(np.median(bandas[b][0] if False else modas[b])))
                       for b, _ in bandas_ordenadas[:n_take])

    # Segundo pase: colapsar bandas muy cercanas (distancia < bin_size*1.5).
    # bin_size*1.5 es estrictamente para eliminar duplicados por líneas
    # partidas, NO para colapsar filas reales (que están separadas >25 px).
    if not candidatos:
        return []
    consolidado = [candidatos[0]]
    tol = max(4, int(bin_size * 1.5))
    for c in candidatos[1:]:
        if c - consolidado[-1] < tol:
            consolidado[-1] = int(round((consolidado[-1] + c) / 2))
        else:
            consolidado.append(c)
    return consolidado


def analizar_celda(cell_gray, space_threshold=7):
    """
    Segmenta caracteres y palabras dentro de una celda mediante componentes conectadas.
    Filtra posibles líneas residuales de la tabla por umbral de área y analiza espacios.

    Retorna:
    --------
    n_chars : int (cantidad total de caracteres detectados)
    n_words : int (cantidad de palabras detectadas por separación horizontal de espacios)
    char_crops : list (sub-imágenes binarias de cada carácter recortado)

    Nota:
    -----
    `space_threshold` (en píxeles) es el gap horizontal mínimo entre
    bounding boxes para considerar que hay un espacio entre palabras.
    Se deja 7 como valor por defecto para mantener compatibilidad con
    la versión anterior; en la versión "calibrada" se obtiene midiendo
    los gaps en celdas conocidas (ver `calibrar_space_threshold`).
    """
    # Descartamos un margen para evitar las líneas de la celda.
    # 4 px es un valor razonable para planillas escaneadas; si el margen
    # fuera mayor el recorte perdería caracteres delgados.
    margin_y = 4
    margin_x = 4
    if cell_gray.shape[0] <= 2 * margin_y or cell_gray.shape[1] <= 2 * margin_x:
        return 0, 0, []

    inner = cell_gray[margin_y:-margin_y, margin_x:-margin_x]

    # Binarización del texto (fondo blanco, texto oscuro).
    # Mismo criterio que en detectar_rejilla_tabla: Otsu.
    # Cita: PDI_U3_Segmentacion.py:624
    _, cell_bin = cv2.threshold(inner, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cell_bin, connectivity=8)
    
    # Filtrar componentes por área.
    # El "-" y "/" son caracteres delgados con 1-2 píxeles de área en
    # esta tipografía. Fijamos el área mínima en 2 píxeles para
    # preservarlos. Cita: PDI_U3_Segmentacion.py:300 (stats[4] = area).
    valid_stats = [s for s in stats[1:] if s[4] >= 2]
    if len(valid_stats) == 0:
        return 0, 0, []
        
    # Ordenar los caracteres de izquierda a derecha según su coordenada X
    valid_stats = sorted(valid_stats, key=lambda s: s[0])
    n_chars = len(valid_stats)
    
    # Extraer los recortes binarios de cada carácter
    char_crops = [cell_bin[s[1]:s[1]+s[3], s[0]:s[0]+s[2]] for s in valid_stats]
    
    # Conteo de palabras mediante el espacio (gap) entre cajas delimitadoras consecutivas.
    # `space_threshold` llega como parámetro (ver calibrar_space_threshold para
    # la versión cuantitativa; el default 7 está mantenido por compatibilidad).
    gaps = [valid_stats[i+1][0] - (valid_stats[i][0] + valid_stats[i][2]) for i in range(n_chars - 1)]
    n_words = 1 + sum(1 for g in gaps if g >= space_threshold)
    
    return n_chars, n_words, char_crops


def calibrar_space_threshold(img_gray, rows, cols, n_muestras=10):
    """
    Calibra el umbral de gap horizontal para separar letras de palabras,
    midiendo los gaps en celdas ya validadas de la planilla.

    Estrategia:
      1. Recorre las primeras `n_muestras` filas de datos (Legajo = 1 palabra,
         Nombre y Apellido = 2+ palabras).
      2. Calcula los gaps entre bounding boxes de cada celda.
      3. Devuelve la mediana entre los dos grupos de gaps (intra-letra y
         inter-palabra) como un umbral cuantitativo.

    Retorna:
    --------
    space_threshold : int (gap en píxeles)

    Notas:
    ------
    - Esta función NO es material oficial de U3 ni de Práctica. Es una
      calibración empírica: se justifica midiendo, no con un número mágico.
    - Si las distribuciones se solapan demasiado, devuelve el valor
      por defecto 7 y avisa por consola.
    """
    gaps_1pal = []   # gaps intra-carácter en celdas de 1 palabra (Legajo)
    gaps_2pal = []   # gaps en celdas de 2+ palabras (Nombre y Apellido)

    for r_idx in range(min(n_muestras, len(rows) - 2)):
        r_top, r_bot = rows[r_idx + 1], rows[r_idx + 2]

        # Legajo (cols[1] a cols[2]) -> debería ser 1 palabra => solo inter-letra
        c_leg = img_gray[r_top:r_bot, cols[1]:cols[2]]
        leg_data = medir_componentes(c_leg)
        if leg_data is not None:
            _, _, stats_leg, _ = leg_data
            if len(stats_leg) >= 2:
                gaps_1pal.extend(gaps_entre(stats_leg))

        # Nombre y Apellido (cols[2] a cols[3]) -> debería ser 2+ palabras
        c_nom = img_gray[r_top:r_bot, cols[2]:cols[3]]
        nom_data = medir_componentes(c_nom)
        if nom_data is not None:
            _, _, stats_nom, _ = nom_data
            if len(stats_nom) >= 3:
                gaps_2pal.extend(gaps_entre(stats_nom))

    if len(gaps_1pal) == 0 or len(gaps_2pal) == 0:
        print("[calibrar_space_threshold] Datos insuficientes, usando default=7")
        return 7

    med_inter = float(np.median(gaps_1pal))
    med_palabras = float(np.median(gaps_2pal))

    # Si las distribuciones no se separan, devolvemos el default
    if med_palabras <= med_inter:
        print(f"[calibrar_space_threshold] Distribuciones solapadas "
              f"(inter={med_inter}, palabras={med_palabras}), usando default=7")
        return 7

    espacio_threshold = int(round((med_inter + med_palabras) / 2))
    # Protección: si el threshold es <= 0 lo descartamos (podría pasar
    # cuando la detección de rejilla está descalibrada y los recortes
    # se solapan). Volvemos al default.
    if espacio_threshold <= 0:
        print(f"[calibrar_space_threshold] threshold={espacio_threshold} no válido, "
              f"usando default=7")
        return 7
    print(f"[calibrar_space_threshold] inter-letra mediana={med_inter:.2f} px, "
          f"inter-palabra mediana={med_palabras:.2f} px -> "
          f"threshold={espacio_threshold} px")
    return espacio_threshold


def medir_componentes(cell_gray):
    """Helper de `calibrar_space_threshold`. Devuelve (bin, n_chars,
    valid_stats ordenados por X, valid_areas) o None si la celda es muy
    chica. Replica el pre-procesamiento de `analizar_celda` sin contar
    palabras."""
    margin = 4
    if cell_gray.shape[0] <= 2 * margin or cell_gray.shape[1] <= 2 * margin:
        return None
    inner = cell_gray[margin:-margin, margin:-margin]
    _, cell_bin = cv2.threshold(inner, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cell_bin, connectivity=8)
    valid_stats = sorted([s for s in stats[1:] if s[4] >= 2], key=lambda s: s[0])
    return cell_bin, num_labels - 1, valid_stats, [s[4] for s in valid_stats]


def gaps_entre(valid_stats):
    """Helper: devuelve la lista de gaps horizontales entre bounding boxes
    consecutivas, en píxeles."""
    return [valid_stats[i + 1][0] - (valid_stats[i][0] + valid_stats[i][2])
            for i in range(len(valid_stats) - 1)]


def clasificar_condicion_final(char_crop):
    """
    Clasifica el carácter de condición final entre 'L' (Libre), 'R' (Recupera) o 'A' (Aprobado).

    Estrategia:
      - Conteo de huecos con `cv2.findContours` + jerarquía
        (`cv2.RETR_TREE`), ambas oficiales de U3. Los hijos del contorno
        exterior son los huecos del carácter. Si no hay huecos -> L.
      - Distribución vertical de masa con `cv2.moments` (oficial U3). El
        centro de masa vertical cae arriba del centro geométrico para 'R'
        (por el bucle superior) y alrededor o debajo para 'A'.

    Retorna:
    --------
    Una cadena: "L", "R", "A", o "DESCONOCIDO" si el recorte es inválido.
    """
    h, w = char_crop.shape
    if h < 4 or w < 3:
        return "DESCONOCIDO"

    # ---------- 1. Conteo de huecos con findContours + jerarquía -------
    # Pasamos la imagen a uint8 (los recortes vienen de cv2.threshold
    # que ya devuelve uint8, así que el casteo es defensivo).
    bin8 = char_crop.astype(np.uint8)
    # Cita oficial: PDI_U3_Segmentacion.py:392-394
    #     contours, hierarchy = cv2.findContours(f, cv2.RETR_TREE,
    #                                            cv2.CHAIN_APPROX_NONE)
    # Cita PDF U3 "Contornos": la jerarquía es [Next, Previous,
    # First_Child, Parent] y un contorno con parent == -1 es externo.
    contours, hierarchy = cv2.findContours(
        bin8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None or len(contours) == 0:
        return "DESCONOCIDO"

    # El contorno exterior es el que tiene parent == -1
    idx_ext = -1
    for i, h_info in enumerate(hierarchy[0]):
        if h_info[3] == -1:
            idx_ext = i
            break
    if idx_ext == -1:
        return "DESCONOCIDO"

    # Sus hijos en la jerarquía (h_info[2] = first_child) son los huecos
    n_huecos = 0
    for h_info in hierarchy[0]:
        if h_info[3] == idx_ext:
            n_huecos += 1

    # ---------- 2. Decisión L/R/A -------------------------------------
    if n_huecos == 0:
        # Sin huecos -> L
        return "L"

    if n_huecos >= 2:
        # Raro: más de un hueco. Tratamos como A por defecto; en la
        # defensa oral lo aclaramos si surge.
        return "A"

    # Un solo hueco: R o A. Distinguimos por centro de masa vertical.
    # Cita oficial: PDI_U3_Segmentacion.py:496
    #     M = cv2.moments(cnt)
    # cv2.moments acepta la imagen completa y devuelve el centro de
    # masa. Para una imagen binaria, m00 es el área del carácter.
    M = cv2.moments(bin8, binaryImage=True)
    if M["m00"] == 0:
        return "DESCONOCIDO"
    # Centro de masa vertical, normalizado por altura
    cy_norm = M["m01"] / M["m00"] / h   # en [0, 1]
    # R -> bucle superior -> masa arriba (cy_norm < 0.45)
    # A -> travesaño + base -> masa centrada o abajo (cy_norm >= 0.45)
    if cy_norm < 0.45:
        return "R"
    return "A"


def debug_celdas(img_gray, rows, cols, sheet_name, figs_dir, n_filas=3):
    """
    Genera una figura intermedia con recortes de las primeras `n_filas`
    filas de las celdas: Legajo, Nombre y Apellido, Parcial 1,
    Condición Final. Cada celda se muestra en escala de grises y binaria.

    Se guarda como `<figs_dir>/<sheet_name>_debug_celdas.png` y se
    muestra con `plt.show(block=False)` (estilo U3).
    """
    import matplotlib.pyplot as plt

    os.makedirs(figs_dir, exist_ok=True)

    columnas = [
        ("Legajo", 1, 2),
        ("Nombre", 2, 3),
        ("Parcial 1", 3, 4),
        ("Parcial 2", 4, 5),
        ("Parcial 3", 5, 6),
        ("Cond. Final", 6, 7),
    ]
    if len(rows) < 2 + n_filas:
        n_filas = len(rows) - 2
    fig, axes = plt.subplots(n_filas, 6, figsize=(18, 3 * n_filas))
    if n_filas == 1:
        axes = axes.reshape(1, -1)
    for f_i in range(n_filas):
        r_top, r_bot = rows[f_i + 1], rows[f_i + 2]
        for c_i, (label, c0, c1) in enumerate(columnas):
            cell = img_gray[r_top:r_bot, cols[c0]:cols[c1]]
            axes[f_i, c_i].imshow(cell, cmap='gray')
            axes[f_i, c_i].set_xticks([])
            axes[f_i, c_i].set_yticks([])
            if f_i == 0:
                axes[f_i, c_i].set_title(label, fontsize=11)

    fig.suptitle(f"Debug recortes de celdas — {sheet_name}")
    fig.tight_layout()
    out = os.path.join(figs_dir, f"{sheet_name}_debug_celdas.png")
    fig.savefig(out)
    plt.show(block=False)
    plt.close(fig)
    print(f"[debug_celdas] Figura guardada en {out}")


def debug_rejilla(img_gray, rows, cols, sheet_name, figs_dir):
    """
    Genera 4 figuras intermedias del pipeline de detección de rejilla
    (estilo PDI_U3_Segmentacion.py con plt.subplot(2x2)):
      [0,0] Imagen original con líneas detectadas dibujadas encima.
      [0,1] Resultado de Canny (bordes 1-px).
      [1,0] Bordes crudos de Canny (sin marcar líneas).
      [1,1] Histograma 1D de coordenadas X (verticales) e Y (horizontales).

    Las figuras se guardan como PNG en `figs_dir/<sheet_name>_*.png` y se
    muestran en pantalla con `plt.show(block=False)`.
    """
    import matplotlib.pyplot as plt  # local para no contaminar imports globales

    os.makedirs(figs_dir, exist_ok=True)

    # Imagen 1: Original + líneas detectadas
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    for y in rows:
        cv2.line(img_bgr, (0, int(y)), (img_gray.shape[1], int(y)), (255, 0, 0), 1)
    for x in cols:
        cv2.line(img_bgr, (int(x), 0), (int(x), img_gray.shape[0]), (0, 255, 0), 1)
    axes[0, 0].imshow(img_bgr)
    axes[0, 0].set_title(f"Líneas detectadas ({len(rows)} filas, {len(cols)} columnas)")
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

    # Imagen 2: Canny
    f_blur = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=1.5)
    edges = cv2.Canny(f_blur, threshold1=80, threshold2=120)
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title("Canny (edges 1-px)")
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])

    # Imagen 3: original en gris
    axes[1, 0].imshow(img_gray, cmap='gray')
    axes[1, 0].set_title("Imagen original")
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])

    # Imagen 4: histogramas de coordenadas detectadas
    axes[1, 1].hist(rows, bins=len(rows), color='red', alpha=0.7, label='Filas (Y)')
    axes[1, 1].hist(cols, bins=len(cols), color='green', alpha=0.7, label='Columnas (X)')
    axes[1, 1].set_title("Distribución de coordenadas detectadas")
    axes[1, 1].legend()

    fig.suptitle(f"Debug detección de rejilla — {sheet_name}")
    fig.tight_layout()
    out = os.path.join(figs_dir, f"{sheet_name}_debug_rejilla.png")
    fig.savefig(out)
    plt.show(block=False)
    plt.close(fig)
    print(f"[debug_rejilla] Figura guardada en {out}")


def procesar_planilla(sheet_path, output_dir):
    """
    Procesa una planilla completa, valida cada registro, exporta CSV e imagen de no aprobados.
    """
    sheet_name = os.path.basename(sheet_path)
    base_name = os.path.splitext(sheet_name)[0]
    
    img_bgr = cv2.imread(sheet_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {sheet_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    rows, cols = detectar_rejilla_tabla(img_gray)
    if len(rows) < 3 or len(cols) < 3:
        raise RuntimeError(
            f"[procesar_planilla] Rejilla inválida en {sheet_name}: "
            f"{len(rows)} filas y {len(cols)} columnas detectadas "
            f"(se esperan >=22 filas y 8 columnas).")

    # Si --debug está activo, generamos figura intermedia estilo U3
    if os.environ.get("PDI_DEBUG") == "1":
        debug_dir = os.path.join(output_dir, "figuras_bloque1")
        debug_rejilla(img_gray, rows, cols, sheet_name, debug_dir)
        debug_celdas(img_gray, rows, cols, sheet_name, debug_dir, n_filas=3)

    num_data_rows = len(rows) - 2

    # Calibramos una sola vez por planilla el threshold de gap entre
    # letras y palabras. Si la calibración falla, space_threshold queda
    # en su default (7), manteniendo la compatibilidad con la versión
    # anterior.
    space_threshold = calibrar_space_threshold(img_gray, rows, cols)
    
    print("\n" + "=" * 60)
    print(f"VALIDACIÓN DE PLANILLA: {sheet_name}")
    print("=" * 60)
    
    registros_csv = []
    alumnos_no_aprobados = []
    
    # Iterar sobre las 20 filas de datos (rows[1] a rows[21])
    for r_idx in range(num_data_rows):
        reg_id = r_idx + 1
        r_top, r_bot = rows[r_idx+1], rows[r_idx+2]
        
        # 1. Analizar Legajo (columna 1: cols[1] a cols[2])
        # Regla: 8 caracteres en total, formando 1 única palabra
        c_leg = img_gray[r_top:r_bot, cols[1]:cols[2]]
        n_c_leg, n_w_leg, _ = analizar_celda(c_leg, space_threshold=space_threshold)
        leg_ok = (n_c_leg == 8 and n_w_leg == 1)

        # 2. Analizar Nombre y Apellido (columna 2: cols[2] a cols[3])
        # Regla: mínimo 2 palabras, entre 1 y 12 caracteres visibles
        c_nom = img_gray[r_top:r_bot, cols[2]:cols[3]]
        n_c_nom, n_w_nom, _ = analizar_celda(c_nom, space_threshold=space_threshold)
        nom_ok = (n_w_nom >= 2 and 1 <= n_c_nom <= 12)

        # 3. Analizar Parciales 1, 2 y 3 (columnas 3, 4 y 5)
        # Regla: 1 o 2 caracteres consecutivos (1 palabra)
        parciales_ok = []
        for p_i in range(3):
            c_p = img_gray[r_top:r_bot, cols[3+p_i]:cols[4+p_i]]
            n_c_p, n_w_p, _ = analizar_celda(c_p, space_threshold=space_threshold)
            p_ok = (n_c_p in [1, 2] and n_w_p == 1)
            parciales_ok.append(p_ok)

        # 4. Analizar Condición Final (columna 6: cols[6] a cols[7])
        # Regla: único carácter (1 carácter, 1 palabra)
        c_cond = img_gray[r_top:r_bot, cols[6]:cols[7]]
        n_c_cond, n_w_cond, crops_cond = analizar_celda(c_cond, space_threshold=space_threshold)
        cond_ok = (n_c_cond == 1 and n_w_cond == 1)
        
        # Clasificar la letra de Condición Final si es 1 carácter
        cond_letra = None
        if cond_ok and len(crops_cond) == 1:
            cond_letra = clasificar_condicion_final(crops_cond[0])
            
        # Determinar si todos los campos son correctos
        all_ok = (leg_ok and nom_ok and all(parciales_ok) and cond_ok)
        
        # Salida por terminal según formato solicitado
        print(f"> Registro {reg_id}:")
        print(f"> Legajo: {'OK' if leg_ok else 'MAL'}")
        print(f"> Nombre y apellido: {'OK' if nom_ok else 'MAL'}")
        print(f"> Parcial 1: {'OK' if parciales_ok[0] else 'MAL'}")
        print(f"> Parcial 2: {'OK' if parciales_ok[1] else 'MAL'}")
        print(f"> Parcial 3: {'OK' if parciales_ok[2] else 'MAL'}")
        print(f"> Condición Final: {'OK' if cond_ok else 'MAL'}")
        print(">")
        
        # Guardar registro para CSV
        registros_csv.append({
            "ID": reg_id,
            "Legajo": "OK" if leg_ok else "MAL",
            "Nombre y Apellido": "OK" if nom_ok else "MAL",
            "Parcial 1": "OK" if parciales_ok[0] else "MAL",
            "Parcial 2": "OK" if parciales_ok[1] else "MAL",
            "Parcial 3": "OK" if parciales_ok[2] else "MAL",
            "Condición Final": "OK" if cond_ok else "MAL"
        })
        
        # Para el punto b: Solo registros cargados correctamente que no hayan aprobado ('L' o 'R')
        if all_ok and cond_letra in ["L", "R"]:
            crop_nom_color = img_bgr[r_top+2:r_bot-2, cols[2]+2:cols[3]-2].copy()
            alumnos_no_aprobados.append({
                "reg_id": reg_id,
                "crop": crop_nom_color,
                "condicion": cond_letra
            })
            
    # c. Exportar CSV
    csv_filename = os.path.join(output_dir, f"validacion_{base_name}.csv")
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["ID", "Legajo", "Nombre y Apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(registros_csv)
    print(f"Archivo CSV generado: {csv_filename}")
    
    # b. Generar imagen de salida de alumnos no aprobados
    img_no_aprobados = renderizar_imagen_no_aprobados(alumnos_no_aprobados, sheet_name)
    img_out_filename = os.path.join(output_dir, f"no_aprobados_{base_name}.png")
    cv2.imwrite(img_out_filename, img_no_aprobados)
    print(f"Imagen de no aprobados generada: {img_out_filename} ({len(alumnos_no_aprobados)} alumnos)")
    
    return registros_csv, alumnos_no_aprobados


def renderizar_imagen_no_aprobados(lista_no_aprobados, sheet_title):
    """
    Construye una imagen que lista los recortes de 'Nombre y Apellido' de los alumnos
    no aprobados ('L' o 'R') y agrega una insignia visual que diferencia ambas condiciones.
    """
    card_h = 36
    header_h = 60
    padding = 10
    total_w = 460
    
    if len(lista_no_aprobados) == 0:
        h = header_h + 80
        canvas = np.ones((h, total_w, 3), dtype=np.uint8) * 245
        cv2.putText(canvas, f"NO APROBADOS - {sheet_title}", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(canvas, "No se registraron alumnos en condicion L o R", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        return canvas
        
    total_h = header_h + len(lista_no_aprobados) * (card_h + padding) + 20
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 250
    
    # Encabezado
    cv2.rectangle(canvas, (0, 0), (total_w, header_h - 10), (33, 43, 54), -1)
    cv2.putText(canvas, f"ALUMNOS NO APROBADOS ({sheet_title})", (15, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                
    y_offset = header_h
    for item in lista_no_aprobados:
        crop = item["crop"]
        cond = item["condicion"]
        reg_id = item["reg_id"]
        
        # Fondo de la fila
        cv2.rectangle(canvas, (10, y_offset), (total_w - 10, y_offset + card_h), (255, 255, 255), -1)
        cv2.rectangle(canvas, (10, y_offset), (total_w - 10, y_offset + card_h), (210, 210, 210), 1)
        
        # ID de registro
        id_text = f"#{reg_id:02d}"
        cv2.putText(canvas, id_text, (20, y_offset + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        
        # Insertar crop de Nombre y Apellido (redimensionado si es necesario)
        cr_h, cr_w = crop.shape[:2]
        target_h = card_h - 6
        target_w = min(int(cr_w * (target_h / max(cr_h, 1))), 230)
        crop_resized = cv2.resize(crop, (target_w, target_h))
        canvas[y_offset + 3 : y_offset + 3 + target_h, 65 : 65 + target_w] = crop_resized
        
        # Insignia de Condición: 'L' = LIBRE (Rojo), 'R' = RECUPERA (Ámbar/Naranja)
        badge_x = 310
        badge_w = 130
        badge_y = y_offset + 5
        badge_h = card_h - 10
        
        if cond == "L":
            color_bg = (0, 0, 255)         # Rojo puro en BGR
            badge_text = "LIBRE (L)"
        else:
            color_bg = (30, 140, 240)      # Naranja BGR (R en U1 = ámbar)
            badge_text = "RECUPERA (R)"
            
        cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), color_bg, -1)
        cv2.putText(canvas, badge_text, (badge_x + 12, badge_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    
        y_offset += card_h + padding
        
    return canvas


def ejecutar_validacion_todas_las_planillas(custom_sheets=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "resultados_problema2")
    os.makedirs(out_dir, exist_ok=True)
    
    if custom_sheets:
        planillas = custom_sheets
    else:
        planillas = [
            os.path.join(base_dir, f"grade_sheet_{i}.png") for i in [1, 2, 3, 4]
        ]
    
    resumen_global = []
    
    for p in planillas:
        if os.path.exists(p):
            recs, no_aprob = procesar_planilla(p, out_dir)
            total_ok = sum(1 for r in recs if all(v == "OK" for k, v in r.items() if k != "ID"))
            resumen_global.append({
                "archivo": os.path.basename(p),
                "total_registros_evaluados": len(recs),
                "registros_completamente_ok": total_ok,
                "alumnos_no_aprobados_validos": len(no_aprob)
            })
        else:
            print(f"Advertencia: No se encontró la imagen {p}")
            
    print("\n" + "=" * 60)
    print("RESUMEN GENERAL DEL PROCESAMIENTO CÍCLICO (Punto d):")
    print("=" * 60)
    for res in resumen_global:
        print(f"Planilla: {res['archivo']}")
        print(f"  - Registros evaluados: {res['total_registros_evaluados']}")
        print(f"  - Registros sin errores (100% OK): {res['registros_completamente_ok']}")
        print(f"  - Alumnos No Aprobados válidos reportados: {res['alumnos_no_aprobados_validos']}")
    print("=" * 60)
    print(f"Todos los archivos CSV e imágenes resultantes fueron guardados en:")
    print(f"  {out_dir}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Problema 2: Validación Automática de Planillas de Calificaciones")
    parser.add_argument("--sheets", nargs="*", default=None, help="Rutas a una o más imágenes de planillas (por defecto grade_sheet_1.png a grade_sheet_4.png)")
    parser.add_argument("--debug", action="store_true", help="Genera figuras intermedias estilo U3 (bloque 1)")
    args = parser.parse_args()
    if args.debug:
        os.environ["PDI_DEBUG"] = "1"
    ejecutar_validacion_todas_las_planillas(args.sheets)
