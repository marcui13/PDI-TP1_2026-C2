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
    Detecta las coordenadas de las líneas divisorias de la tabla (filas y columnas).
    Utiliza operaciones morfológicas y proyecciones unidimensionales.
    """
    # 1. Binarización de líneas oscuras
    bin_grid = (img_gray < 80).astype(np.uint8)
    
    # 2. Aperturas morfológicas para aislar componentes horizontales y verticales
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    lines_h = cv2.morphologyEx(bin_grid, cv2.MORPH_OPEN, k_h)
    lines_v = cv2.morphologyEx(bin_grid, cv2.MORPH_OPEN, k_v)
    
    # 3. Detección de columnas mediante proyección vertical
    c_sum = np.sum(lines_v, axis=0)
    c_thresh = c_sum > (0.3 * c_sum.max())
    diff_c = np.diff(np.pad(c_thresh.astype(int), (1, 1)))
    col_starts = np.where(diff_c == 1)[0]
    col_ends = np.where(diff_c == -1)[0]
    cols = [int(round((s + e) / 2)) for s, e in zip(col_starts, col_ends)]
    
    # Se esperan 8 líneas verticales que delimitan 7 columnas
    table_left, table_right = cols[0], cols[-1]
    table_w = table_right - table_left
    
    # 4. Detección de filas completas que cubren al menos el 70% del ancho de la tabla
    r_sum = np.sum(lines_h[:, table_left:table_right], axis=1)
    r_thresh = r_sum > (0.7 * table_w)
    diff_r = np.diff(np.pad(r_thresh.astype(int), (1, 1)))
    row_starts = np.where(diff_r == 1)[0]
    row_ends = np.where(diff_r == -1)[0]
    rows = [int(round((s + e) / 2)) for s, e in zip(row_starts, row_ends)]
    
    return rows, cols


def analizar_celda(cell_gray):
    """
    Segmenta caracteres y palabras dentro de una celda mediante componentes conectadas.
    Filtra posibles líneas residuales de la tabla por umbral de área y analiza espacios.
    
    Retorna:
    --------
    n_chars : int (cantidad total de caracteres detectados)
    n_words : int (cantidad de palabras detectadas por separación horizontal de espacios)
    char_crops : list (sub-imágenes binarias de cada carácter recortado)
    """
    # Descartamos un margen de 2 píxeles para evitar las líneas de la celda
    margin_y = 2
    margin_x = 2
    if cell_gray.shape[0] <= 2 * margin_y or cell_gray.shape[1] <= 2 * margin_x:
        return 0, 0, []
        
    inner = cell_gray[margin_y:-margin_y, margin_x:-margin_x]
    
    # Binarización del texto (fondo blanco, texto oscuro)
    cell_bin = (inner < 120).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cell_bin, connectivity=8)
    
    # Filtrar componentes por área (eliminar ruido de fondo menor a 3 píxeles)
    # stats: [x, y, width, height, area]
    valid_stats = [s for s in stats[1:] if s[4] >= 3]
    if len(valid_stats) == 0:
        return 0, 0, []
        
    # Ordenar los caracteres de izquierda a derecha según su coordenada X
    valid_stats = sorted(valid_stats, key=lambda s: s[0])
    n_chars = len(valid_stats)
    
    # Extraer los recortes binarios de cada carácter
    char_crops = [cell_bin[s[1]:s[1]+s[3], s[0]:s[0]+s[2]] for s in valid_stats]
    
    # Conteo de palabras mediante el espacio (gap) entre cajas delimitadoras consecutivas
    # En texto tipográfico de estas planillas, el inter-letra es <= 4 px y el espacio entre palabras >= 7 px
    gaps = [valid_stats[i+1][0] - (valid_stats[i][0] + valid_stats[i][2]) for i in range(n_chars - 1)]
    space_threshold = 7
    n_words = 1 + sum(1 for g in gaps if g >= space_threshold)
    
    return n_chars, n_words, char_crops


def clasificar_condicion_final(char_crop):
    """
    Clasifica el carácter de condición final entre 'L' (Libre), 'R' (Recupera) o 'A' (Aprobado).
    Utiliza invariantes topológicas (número de huecos de fondo / característica de Euler)
    y la distribución vertical de masa.
    """
    h, w = char_crop.shape
    if h < 4 or w < 3:
        return "DESCONOCIDO"
        
    # Invertir el carácter con borde para contar componentes de fondo (huecos internos)
    inv = np.pad(1 - char_crop, 1, constant_values=1)
    num_bg, _, _, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    # num_bg incluye el fondo exterior y los huecos interiores
    huecos = num_bg - 2
    
    if huecos <= 0:
        # La 'L' no tiene ningún hueco cerrado
        return "L"
    else:
        # 'A' y 'R' tienen 1 hueco interior
        # La 'R' concentra mayor masa en la mitad superior (el bucle superior)
        # La 'A' es triangular apuntando hacia arriba, concentrando mayor masa en la base
        mid_y = h // 2
        top_mass = float(char_crop[:mid_y, :].sum())
        bot_mass = float(char_crop[mid_y:, :].sum())
        ratio = top_mass / (bot_mass + 1e-5)
        if ratio > 0.9:
            return "R"
        else:
            return "A"


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
    num_data_rows = len(rows) - 2
    
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
        n_c_leg, n_w_leg, _ = analizar_celda(c_leg)
        leg_ok = (n_c_leg == 8 and n_w_leg == 1)
        
        # 2. Analizar Nombre y Apellido (columna 2: cols[2] a cols[3])
        # Regla: mínimo 2 palabras, entre 1 y 12 caracteres visibles
        c_nom = img_gray[r_top:r_bot, cols[2]:cols[3]]
        n_c_nom, n_w_nom, _ = analizar_celda(c_nom)
        nom_ok = (n_w_nom >= 2 and 1 <= n_c_nom <= 12)
        
        # 3. Analizar Parciales 1, 2 y 3 (columnas 3, 4 y 5)
        # Regla: 1 o 2 caracteres consecutivos (1 palabra)
        parciales_ok = []
        for p_i in range(3):
            c_p = img_gray[r_top:r_bot, cols[3+p_i]:cols[4+p_i]]
            n_c_p, n_w_p, _ = analizar_celda(c_p)
            p_ok = (n_c_p in [1, 2] and n_w_p == 1)
            parciales_ok.append(p_ok)
            
        # 4. Analizar Condición Final (columna 6: cols[6] a cols[7])
        # Regla: único carácter (1 carácter, 1 palabra)
        c_cond = img_gray[r_top:r_bot, cols[6]:cols[7]]
        n_c_cond, n_w_cond, crops_cond = analizar_celda(c_cond)
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
    args = parser.parse_args()
    ejecutar_validacion_todas_las_planillas(args.sheets)
