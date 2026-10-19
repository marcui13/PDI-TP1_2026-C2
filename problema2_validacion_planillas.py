"""
Trabajo Práctico N° 1 - Procesamiento de Imágenes I (TUIA - FCEIA - UNR)
Año 2026 - 2° Semestre

Problema 2: Validación de Planilla de Calificaciones
====================================================
a. Lectura de formulario y validación campo por campo por registro (salida por terminal).
b. Generación de imagen de salida con el crop de Nombre y Apellido de alumnos no aprobados
   ('L' o 'R') con registros válidos, indicando su condición en un panel visual estilizado.
c. Generación de archivo CSV por planilla con las columnas requeridas (ID, Legajo, Nombre y Apellido,
   Parcial 1, Parcial 2, Parcial 3, Condición Final) y valores OK/MAL.
d. Ejecución cíclica sobre las 4 imágenes (grade_sheet_1.png a grade_sheet_4.png).

Metodología (Alineada estrictamente a Unidades 1 a 3 - Sin Morfología Matemática):
- Binarización por umbral global de tinta (pixel < 150).
- Detección estructural de grilla mediante perfiles de proyección 1D (sumas horizontales y verticales).
- Extracción de componentes conexas ordenadas horizontalmente y análisis de gaps para palabras.
- Clasificador topológico de condición ('L', 'R', 'A') mediante jerarquía de contornos (Euler) y trazo lateral.
- Diagnóstico visual paso a paso de la detección de rejilla y procesamiento de celdas.
"""

import os
import sys
import csv
import cv2
import numpy as np

# Configurar directorio temporal para caché de Matplotlib si es necesario
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib.pyplot as plt

# Auto-detección del entorno virtual local (.venv) si no está activo
for p in [
    os.path.join(os.path.dirname(__file__), ".venv/lib/python3.9/site-packages"),
    os.path.join(os.path.dirname(__file__), "../.venv/lib/python3.9/site-packages"),
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, os.path.abspath(p))

CAMPOS = ["Legajo", "Nombre y apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]
UMBRAL_BINARIO = 150       # Píxel < 150 se considera tinta oscura
UMBRAL_ESPACIO = 6         # Hueco horizontal (px) a partir del cual empieza otra palabra
UMBRAL_AREA = 3            # Manchas con área <= 3 px se descartan (ruido / restos)


def detectar_lineas(proyeccion, umbral):
    """
    Devuelve lista de (inicio, fin) de cada línea de la grilla.
    Agrupa píxeles contiguos por si la línea física tiene grosor > 1 px.
    """
    idx = np.where(proyeccion > umbral)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > 1)[0]
    inicios = np.concatenate(([idx[0]], idx[cortes + 1]))
    fines = np.concatenate((idx[cortes], [idx[-1]]))
    return list(zip(inicios, fines))


def detectar_celdas(img_th):
    """
    Detecta las celdas de la tabla mediante proyecciones 1D horizontales y verticales.
    Retorna:
      - filas: lista de tuplas (y_ini, y_fin) de líneas horizontales
      - cols: lista de tuplas (x_ini, x_fin) de líneas verticales
      - registros_coords: lista por registro de 6 tuplas (y0, y1, x0, x1) para cada campo.
    """
    alto, ancho = img_th.shape
    filas = detectar_lineas(np.sum(img_th, 1), 0.5 * ancho)
    cols = detectar_lineas(np.sum(img_th, 0), 0.4 * alto)

    # filas[0] = borde superior, filas[1] = fin encabezado, luego 20 registros
    # cols[0] = borde izquierdo, cols[1] = fin de columna 'Nro.', luego 6 campos
    registros_coords = []
    for r in range(1, len(filas) - 1):
        y0, y1 = filas[r][1] + 1, filas[r + 1][0]
        celdas_coords = []
        for c in range(1, len(cols) - 1):
            x0 = cols[c][1] + 1      # 1 px después de la línea divisoria izquierda
            x1 = cols[c + 1][0]      # donde empieza la línea divisoria derecha
            celdas_coords.append((y0, y1, x0, x1))
        registros_coords.append(celdas_coords)
    return filas, cols, registros_coords


def componentes(celda):
    """
    Calcula las estadísticas de las componentes conexas (sin fondo ni ruido),
    ordenadas horizontalmente de izquierda a derecha.
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(celda.astype(np.uint8), 8, cv2.CV_32S)
    stats = stats[1:]   # Descartar fondo
    stats = stats[stats[:, cv2.CC_STAT_AREA] > UMBRAL_AREA]
    if len(stats) == 0:
        return stats
    return stats[np.argsort(stats[:, cv2.CC_STAT_LEFT])]


def contar(celda):
    """Devuelve (cantidad de caracteres, cantidad de palabras) de una celda."""
    stats = componentes(celda)
    if len(stats) == 0:
        return 0, 0
    fin_anterior = stats[:-1, cv2.CC_STAT_LEFT] + stats[:-1, cv2.CC_STAT_WIDTH]
    huecos = stats[1:, cv2.CC_STAT_LEFT] - fin_anterior
    return len(stats), int(np.sum(huecos > UMBRAL_ESPACIO)) + 1


def validar(campo, cant_caracteres, cant_palabras):
    """Valida la consistencia sintáctica de cada campo según especificaciones."""
    if campo == "Legajo":
        ok = (cant_caracteres == 8 and cant_palabras == 1)
    elif campo == "Nombre y apellido":
        ok = (cant_palabras >= 2 and cant_caracteres <= 12)
    elif campo == "Condición Final":
        ok = (cant_caracteres == 1)
    else:   # Parcial 1, 2 y 3
        ok = (cant_palabras == 1 and 1 <= cant_caracteres <= 2)
    return "OK" if ok else "MAL"


def clasificar_condicion(celda):
    """
    Distingue determinísticamente 'L' / 'R' / 'A' por forma topológica:
     - 'L': 0 agujeros internos (Euler).
     - 'R' y 'A': 1 agujero interno. 'R' posee un trazo vertical continuo en su borde izquierdo (> 0.90), 'A' no.
    """
    stats = componentes(celda)
    if len(stats) == 0:
        return "DESCONOCIDO"
    x, y, w, h = stats[0, :4]
    letra = celda[y:y + h, x:x + w].astype(np.uint8)
    contornos, jerarquia = cv2.findContours(letra, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if jerarquia is None or len(jerarquia) == 0:
        return "DESCONOCIDO"
    agujeros = sum(1 for j in jerarquia[0] if j[3] != -1)
    if agujeros == 0:
        return "L"
    columna_izquierda = letra[:, :max(1, w // 5)].any(axis=1).mean()
    return "R" if columna_izquierda > 0.9 else "A"


def renderizar_imagen_no_aprobados(lista_no_aprobados, sheet_title):
    """
    Construye una imagen con estilo estilizado que lista los recortes de 'Nombre y Apellido'
    de los alumnos no aprobados ('L' o 'R') con insignias visuales distintivas.
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

    # Encabezado superior
    cv2.rectangle(canvas, (0, 0), (total_w, header_h - 10), (33, 43, 54), -1)
    cv2.putText(canvas, f"ALUMNOS NO APROBADOS ({sheet_title})", (15, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    y_offset = header_h
    for item in lista_no_aprobados:
        crop = item["crop"]
        cond = item["condicion"]
        reg_id = item["reg_id"]

        # Fondo de la tarjeta individual
        cv2.rectangle(canvas, (10, y_offset), (total_w - 10, y_offset + card_h), (255, 255, 255), -1)
        cv2.rectangle(canvas, (10, y_offset), (total_w - 10, y_offset + card_h), (210, 210, 210), 1)

        # ID de registro
        id_text = f"#{reg_id:02d}"
        cv2.putText(canvas, id_text, (20, y_offset + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)

        # Insertar recorte de Nombre y Apellido
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
            color_bg = (50, 50, 220)       # Rojo BGR
            badge_text = "LIBRE (L)"
        else:
            color_bg = (30, 140, 240)      # Naranja BGR
            badge_text = "RECUPERA (R)"

        cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), color_bg, -1)
        cv2.putText(canvas, badge_text, (badge_x + 12, badge_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        y_offset += card_h + padding

    return canvas


def generar_graficos_paso_a_paso(sheet_path, output_dir, mostrar=False):
    """
    Genera y guarda gráficos detallados de diagnóstico que demuestran paso a paso:
      1. Detección de rejilla: Binarización, perfiles de proyección 1D y grilla superpuesta.
      2. Detección de celdas: Extracción limpia, análisis de componentes conexas, gaps y Euler.
    """
    img_bgr = cv2.imread(sheet_path)
    if img_bgr is None:
        return
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_th = img_gray < UMBRAL_BINARIO
    alto, ancho = img_th.shape

    # Proyecciones 1D
    proy_h = np.sum(img_th, axis=1)
    proy_v = np.sum(img_th, axis=0)
    filas = detectar_lineas(proy_h, 0.5 * ancho)
    cols = detectar_lineas(proy_v, 0.4 * alto)

    # -------------------------------------------------------------------------
    # FIGURA 1: PASO A PASO DETECCIÓN DE REJILLA
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))

    # (1) Imagen binarizada
    axes1[0, 0].imshow(img_th, cmap='gray')
    axes1[0, 0].set_title("(1) Binarización Global (Píxeles oscuros < 150)", fontsize=12, fontweight='bold')
    axes1[0, 0].axis('off')

    # (2) Proyección Horizontal y Umbral de Filas
    axes1[0, 1].plot(proy_h, range(alto), color='#1f77b4', lw=1.2, label='Perfil horizontal')
    axes1[0, 1].axvline(0.5 * ancho, color='r', linestyle='--', lw=1.5, label=f'Umbral fila = 50% ({0.5 * ancho:.0f} px)')
    for f_ini, f_fin in filas:
        axes1[0, 1].axhline((f_ini + f_fin) / 2, color='green', alpha=0.35, lw=1)
    axes1[0, 1].invert_yaxis()
    axes1[0, 1].set_title(f"(2) Proyección Horizontal: {len(filas)} líneas detectadas", fontsize=12, fontweight='bold')
    axes1[0, 1].set_xlabel("Píxeles de tinta por fila (suma horizontal)")
    axes1[0, 1].set_ylabel("Fila (Y)")
    axes1[0, 1].legend(loc='lower right')
    axes1[0, 1].grid(True, alpha=0.3)

    # (3) Proyección Vertical y Umbral de Columnas
    axes1[1, 0].plot(proy_v, color='#1f77b4', lw=1.2, label='Perfil vertical')
    axes1[1, 0].axhline(0.4 * alto, color='r', linestyle='--', lw=1.5, label=f'Umbral col = 40% ({0.4 * alto:.0f} px)')
    for c_ini, c_fin in cols:
        axes1[1, 0].axvline((c_ini + c_fin) / 2, color='green', alpha=0.35, lw=1)
    axes1[1, 0].set_title(f"(3) Proyección Vertical: {len(cols)} líneas detectadas", fontsize=12, fontweight='bold')
    axes1[1, 0].set_xlabel("Columna (X)")
    axes1[1, 0].set_ylabel("Píxeles de tinta por columna (suma vertical)")
    axes1[1, 0].legend(loc='upper right')
    axes1[1, 0].grid(True, alpha=0.3)

    # (4) Rejilla Delimitada
    grid_vis = img_bgr.copy()
    for f_ini, f_fin in filas:
        cv2.line(grid_vis, (cols[0][0], f_ini), (cols[-1][1], f_ini), (0, 0, 255), 2)
    for c_ini, c_fin in cols:
        cv2.line(grid_vis, (c_ini, filas[0][0]), (c_ini, filas[-1][1]), (255, 0, 0), 2)
    axes1[1, 1].imshow(cv2.cvtColor(grid_vis, cv2.COLOR_BGR2RGB))
    axes1[1, 1].set_title(f"(4) Rejilla Delimitada: {len(filas)-1} filas x {len(cols)-1} columnas", fontsize=12, fontweight='bold')
    axes1[1, 1].axis('off')

    plt.tight_layout()
    ruta_fig1 = os.path.join(output_dir, "paso_a_paso_deteccion_rejilla.png")
    plt.savefig(ruta_fig1, dpi=200)
    print(f"Gráfico de rejilla generado: {ruta_fig1}")
    if mostrar:
        plt.show()
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # FIGURA 2: PASO A PASO DETECCIÓN Y ANÁLISIS DE CELDAS
    # -------------------------------------------------------------------------
    r0 = 1 # Fila del Registro 1
    y0, y1 = filas[r0][1] + 1, filas[r0 + 1][0]
    celdas_bin = [img_th[y0:y1, cols[c][1] + 1 : cols[c + 1][0]] for c in range(1, len(cols) - 1)]

    fig2 = plt.figure(figsize=(16, 12))
    gs = fig2.add_gridspec(3, 6, height_ratios=[1, 1.3, 1.3])

    # Fila 1: Celdas individuales recortadas
    for i in range(6):
        ax = fig2.add_subplot(gs[0, i])
        ax.imshow(celdas_bin[i], cmap='gray')
        ax.set_title(f"{CAMPOS[i]}\n({celdas_bin[i].shape[0]}x{celdas_bin[i].shape[1]} px)", fontsize=10, fontweight='bold')
        ax.axis('off')

    # Fila 2 Izq: Componentes Conexas y Bounding Boxes (Nombre y Apellido)
    celda_nom = celdas_bin[1]
    stats_sorted = componentes(celda_nom)

    vis_nom = cv2.cvtColor((celda_nom * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for idx, s in enumerate(stats_sorted):
        x, y, w, h = s[:4]
        cv2.rectangle(vis_nom, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(vis_nom, str(idx + 1), (x, max(8, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)

    ax_bb = fig2.add_subplot(gs[1, :3])
    ax_bb.imshow(cv2.cvtColor(vis_nom, cv2.COLOR_BGR2RGB))
    ax_bb.set_title(f"(A) Bounding Boxes y Orden Secuencial ({len(stats_sorted)} caracteres detectados)", fontsize=11, fontweight='bold')
    ax_bb.axis('off')

    # Fila 2 Der: Análisis de Gaps Horizontales
    fin_ant = stats_sorted[:-1, cv2.CC_STAT_LEFT] + stats_sorted[:-1, cv2.CC_STAT_WIDTH]
    gaps = stats_sorted[1:, cv2.CC_STAT_LEFT] - fin_ant

    ax_gap = fig2.add_subplot(gs[1, 3:])
    x_gaps = [f"{i+1}-{i+2}" for i in range(len(gaps))]
    colores_gaps = ['#d9534f' if g > 6 else '#5bc0de' for g in gaps]
    bars = ax_gap.bar(x_gaps, gaps, color=colores_gaps, edgecolor='black', alpha=0.85)
    ax_gap.axhline(6, color='r', linestyle='--', lw=1.5, label='Umbral espacio = 6 px')
    ax_gap.set_title("(B) Análisis de Gaps Horizontales: Inter-Letra (<=6 px) vs Inter-Palabra (>6 px)", fontsize=11, fontweight='bold')
    ax_gap.set_xlabel("Entre componente i e i+1")
    ax_gap.set_ylabel("Distancia horizontal (px)")
    ax_gap.legend()
    ax_gap.grid(True, alpha=0.3, axis='y')
    for bar, g in zip(bars, gaps):
        ax_gap.text(bar.get_x() + bar.get_width() / 2, g + 0.3, str(g), ha='center', fontsize=9, fontweight='bold')

    # Fila 3: Topología de Euler y Trazo Lateral
    celda_cond = celdas_bin[5]
    s_c = componentes(celda_cond)
    x_c, y_c, w_c, h_c = s_c[0, :4]
    letra_crop = celda_cond[y_c:y_c + h_c, x_c:x_c + w_c].astype(np.uint8)

    # 3.1: Carácter recortado
    ax_c1 = fig2.add_subplot(gs[2, 0:2])
    ax_c1.imshow(letra_crop, cmap='gray')
    ax_c1.set_title(f"(C) Carácter Condición Final ({h_c}x{w_c} px)", fontsize=11, fontweight='bold')
    ax_c1.axis('off')

    # 3.2: Contornos internos / externos
    contornos, jerarquia = cv2.findContours(letra_crop.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    vis_contornos = np.zeros((h_c, w_c, 3), dtype=np.uint8)
    agujeros = 0
    if jerarquia is not None and len(jerarquia) > 0:
        for idx_cont, j in enumerate(jerarquia[0]):
            color = (0, 255, 255) if j[3] == -1 else (0, 0, 255)
            if j[3] != -1:
                agujeros += 1
            cv2.drawContours(vis_contornos, contornos, idx_cont, color, 1)

    ax_c2 = fig2.add_subplot(gs[2, 2:4])
    ax_c2.imshow(vis_contornos)
    ax_c2.set_title(f"(D) Topología Euler: {agujeros} cavidad(es) cerrada(s)", fontsize=11, fontweight='bold')
    ax_c2.axis('off')

    # 3.3: Perfil vertical izquierdo
    w_strip = max(1, w_c // 5)
    franja_izq = letra_crop[:, :w_strip]
    cobertura = franja_izq.any(axis=1).mean()

    ax_c3 = fig2.add_subplot(gs[2, 4:])
    ax_c3.barh(range(h_c), [1 if franja_izq[row, :].any() else 0 for row in range(h_c)], color='#2ca02c', edgecolor='black')
    axes_inv = ax_c3.invert_yaxis()
    ax_c3.set_title(f"(E) Perfil Vertical Lateral Izquierdo (Cobertura: {cobertura*100:.1f}%)", fontsize=11, fontweight='bold')
    ax_c3.set_xlabel("Presencia de trazo")
    ax_c3.set_ylabel("Fila del carácter (Y)")
    ax_c3.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    ruta_fig2 = os.path.join(output_dir, "paso_a_paso_deteccion_celdas.png")
    plt.savefig(ruta_fig2, dpi=200)
    print(f"Gráfico de celdas generado: {ruta_fig2}")
    if mostrar:
        plt.show()
    plt.close(fig2)


def procesar_planilla(sheet_path, output_dir):
    """
    Procesa una planilla completa:
    a. Valida campo por campo cada registro y muestra por consola.
    b. Genera imagen de no aprobados en el directorio de salida.
    c. Genera archivo CSV estructurado por registro.
    """
    sheet_name = os.path.basename(sheet_path)
    base_name = os.path.splitext(sheet_name)[0]

    img_bgr = cv2.imread(sheet_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {sheet_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Binarización sin morfología
    img_th = img_gray < UMBRAL_BINARIO

    # 2. Detección de celdas mediante proyecciones 1D
    filas, cols, registros_coords = detectar_celdas(img_th)

    print("\n" + "=" * 60)
    print(f"VALIDACIÓN DE PLANILLA: {sheet_name}")
    print("=" * 60)

    registros_csv = []
    alumnos_no_aprobados = []

    # Iterar sobre los 20 registros
    for r_idx, celdas_coords in enumerate(registros_coords):
        reg_id = r_idx + 1
        print(f"> Registro {reg_id}:")
        estados_registro = []

        # Extraer subimágenes binarias para los 6 campos
        celdas_bin = [img_th[y0:y1, x0:x1] for (y0, y1, x0, x1) in celdas_coords]

        for campo, celda_bin in zip(CAMPOS, celdas_bin):
            n_chars, n_words = contar(celda_bin)
            estado = validar(campo, n_chars, n_words)
            print(f"> {campo}: {estado}")
            estados_registro.append(estado)
        print(">")

        # Guardar para CSV
        registros_csv.append({
            "ID": reg_id,
            "Legajo": estados_registro[0],
            "Nombre y Apellido": estados_registro[1],
            "Parcial 1": estados_registro[2],
            "Parcial 2": estados_registro[3],
            "Parcial 3": estados_registro[4],
            "Condición Final": estados_registro[5]
        })

        # Punto b: Solo registros 100% OK con condición 'L' o 'R'
        if all(e == "OK" for e in estados_registro):
            condicion = clasificar_condicion(celdas_bin[5])
            if condicion in ("L", "R"):
                y0, y1, x0, x1 = celdas_coords[1]
                crop_nom = img_bgr[y0:y1, x0:x1].copy()
                alumnos_no_aprobados.append({
                    "reg_id": reg_id,
                    "crop": crop_nom,
                    "condicion": condicion
                })

    # c. Exportar CSV
    csv_filename = os.path.join(output_dir, f"validacion_{base_name}.csv")
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["ID", "Legajo", "Nombre y Apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(registros_csv)
    print(f"Archivo CSV generado: {csv_filename}")

    # b. Generar imagen estilizada de no aprobados
    img_no_aprobados = renderizar_imagen_no_aprobados(alumnos_no_aprobados, sheet_name)
    img_out_filename = os.path.join(output_dir, f"no_aprobados_{base_name}.png")
    cv2.imwrite(img_out_filename, img_no_aprobados)
    print(f"Imagen de no aprobados generada: {img_out_filename} ({len(alumnos_no_aprobados)} alumnos)")

    return registros_csv, alumnos_no_aprobados


def ejecutar_validacion_todas_las_planillas(custom_sheets=None, graficos=True, mostrar=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "resultados_problema2")
    os.makedirs(out_dir, exist_ok=True)

    if custom_sheets:
        planillas = custom_sheets
    else:
        planillas = [
            os.path.join(base_dir, f"grade_sheet_{i}.png") for i in [1, 2, 3, 4]
        ]

    # Generación de gráficos paso a paso sobre la primera planilla
    if graficos and len(planillas) > 0 and os.path.exists(planillas[0]):
        print("\n" + "=" * 60)
        print(f"GENERANDO DIAGNÓSTICOS VISUALES PASO A PASO ({os.path.basename(planillas[0])})")
        print("=" * 60)
        generar_graficos_paso_a_paso(planillas[0], out_dir, mostrar=mostrar)

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
    parser.add_argument("--plot", "--mostrar", action="store_true", help="Muestra interactivamente las figuras paso a paso en pantalla")
    parser.add_argument("--no-graficos", action="store_true", help="Desactiva la generación de imágenes de diagnóstico paso a paso")
    args = parser.parse_args()
    ejecutar_validacion_todas_las_planillas(args.sheets, graficos=not args.no_graficos, mostrar=args.plot)
