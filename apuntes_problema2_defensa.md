# Apuntes para la defensa oral — Problema 2 (Validación de Planillas)

> Material personal de estudio. NO va en la entrega. Se va acumulando por
> bloque a medida que repasamos el script `problema2_validacion_planillas.py`.

## Convención de figuras intermedias (para el informe)

Decidido junto con el repaso:

- **Cada función** arma su propia figura con `plt.subplots(...)` estilo
  `PDI_U3_Segmentacion.py` (e.g. `plt.subplot(2x2)` con `imshow` +
  `cmap='gray'` + títulos). Llama a `plt.show(block=False)` al final.
- **Cada figura se guarda como PNG** además de mostrarse, así no hay que
  regenerar nada para el informe.
- **Una figura global del pipeline entero** al final del script
  (`figura_pipeline`) para mostrar de un vistazo todo el proceso.
- **Layout de carpetas**:
  ```
  resultados_problema2/
    figuras_bloque1/   # detección de rejilla (por planilla)
    figuras_bloque2/   # análisis de celda
    figuras_bloque3/   # clasificación topológica
    figuras_bloque4/   # orquestación + reporte
    figura_pipeline/   # pipeline entero
    validacion_grade_sheet_<id>.csv
    no_aprobados_grade_sheet_<id>.png
  ```
- **Total esperado**: ~5 figuras por planilla × 4 planillas = ~20 PNGs.
  Es justo el material que el profe pidió para el informe.

## Contexto

- TP1 Problema 2 entregado en `PDI-TP1_2026-C2/`.
- Defensa anterior: nos fue mal porque partes del código no estaban en el
  material visto en clase (morfología, sobre todo).
- Restricción: **código respaldado por lo que se vio en Unidad 1, 2, 3 y
  Práctica**. Si una técnica no está en ese material, se reemplaza.

## Fuentes oficiales cacheadas (citamos de acá)

- `Unidad1/PDI_TUIA_U1_Fundamentos.pdf` + código `Unidad1/Codigo/`
- `Unidad1/PDI_TUIA_U1_Introduccion.pdf`
- `Unidad2/PDI_TUIA_U2_Transformacion_y_Filtrado.pdf` + código `Unidad2/Código-20250910/`
- `Unidad2/PDI_TUIA_U2_Filtrado_Frecuencial.pdf`
- `Unidad3/PDI_TUIA_U3_Segmentacion.pdf` + código `Unidad3/Codigo/`
  (el más usado: `PDI_U3_Segmentacion.py`, `canny.py`, `canny_example.py`)
- `Practica/Letras A y B-20260205/` y `Practica/Reemplazo de fondo verde-20260205/`

> Nota: el único lugar donde aparece morfología en el material
> oficialmente visto es `Practica/Letras A y B-20260205/clasifico_Letras_AB.py`
> (con `cv2.getStructuringElement` y `cv2.morphologyEx`). El profe dijo
> en la defensa anterior que **no se puede usar morfología porque es
> contenido de unidades posteriores**, así que en este Problema 2
> la evitamos por las dudas.

---

## Bloque 1 — `detectar_rejilla_tabla(img_gray)`

**Qué hace la función**: encontrar automáticamente las coordenadas de las
líneas horizontales y verticales que delimitan filas y columnas de la
planilla, para poder recortar celdas después.

### Línea por línea

| Lín | Código | Estado | Razón |
|----:|--------|:------:|-------|
| 36 | `bin_grid = (img_gray < 80).astype(np.uint8)` | 🟡 reemplazar | Binarización "casera". Existe oficial en U3. |
| 39-40 | `cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))` y vertical `(1, 40)` | 🔴 fuera | Morfología no se dio en clase (profesor confirmó en defensa). |
| 41-42 | `cv2.morphologyEx(bin_grid, cv2.MORPH_OPEN, k_h/v)` | 🔴 fuera | Ídem. Reemplazo: Sobel + Canny + Hough. |
| 45 | `c_sum = np.sum(lines_v, axis=0)` | 🟡 | Array math de U1 — válido, defender como tal. |
| 46 | `c_thresh = c_sum > (0.3 * c_sum.max())` | 🟡 | Umbral al 30% del máximo — convencional, no oficial. |
| 47-50 | `np.pad + np.diff + np.where` para detectar picos | 🟡 | Matemática pura, defendible. |
| 53 | `table_left, table_right = cols[0], cols[-1]` | 🔴 asumir 8 líneas | Asunción rígida: si hay más/menos columnas revienta. |

### Decisiones de rediseño (con cita textual)

1. **Binarización → `cv2.threshold` + `THRESH_OTSU`**.
   - Cita PDF U3 ("Umbralado Global"): *"Este método calcula el umbral que
     minimiza la varianza intraclase"* (ese es Otsu).
   - Cita oficial `PDI_U3_Segmentacion.py:624`:
     ```python
     T_otsu, img_th_otsu = cv2.threshold(img, thresh=127, maxval=255,
                                         type=cv2.THRESH_OTSU)
     ```
   - Mi versión: `_, bin_grid = cv2.threshold(img_gray, 0, 255,
     cv2.THRESH_BINARY + cv2.THRESH_OTSU)` (U3 hace `thresh=127`,
     pero con `+THRESH_OTSU` el flag se ignora).

2. **Bordes → Canny con suavizado previo**.
   - Cita `PDI_U3_Segmentacion.py:335-342`:
     ```python
     f_blur = cv2.GaussianBlur(f, ksize=(3, 3), sigmaX=1.5)
     gcan = cv2.Canny(f_blur, threshold1=80, threshold2=120)
     ```
   - Cita `canny.py:5-9` (versión "a mano" pero usamos la oficial de cv2
     que es la misma idea en 4 pasos):
     `gradiente → supresión de no-máximos → doble umbralado → histéresis`.

3. **Detección de líneas → Transformada de Hough**.
   - Cita PDF U3 pág "Transformada de Hough": *"La transformada de Hough
     es una estrategia para encontrar y unir segmentos de [línea]"*.
   - Cita oficial `PDI_U3_Segmentacion.py:509-538`:
     ```python
     edges = cv2.Canny(f, 100, 170, apertureSize=3)
     lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=250)
     ```
   - Mi versión: `cv2.HoughLinesP` (variante probabilística) devuelve
     directamente los segmentos `(x1,y1,x2,y2)`, más útil acá.
     Mismo `cv2.HoughLinesP` es API estándar de OpenCV.

### Qué estudiarse para defender en oral

- **Concepto de umbralado global** y por qué Otsu lo elige solo.
- **Canny en 4 pasos** (gradiente, NMS, doble umbral, histéresis) — sin
  entrar en fórmulas, pero poder decir "suavizo, hallo bordes con Canny,
  y aplico Hough".
- **Hough intuitivo**: cada pixel de la imagen vota en el espacio (ρ, θ);
  picos en ese espacio = líneas. Eso lo podés explicar en 30 segundos.
- **Por qué no se usa morfología**: no se dio en clase. Si te preguntan
  más, decís "es contenido de unidades posteriores".

---

## Bloque 2 — `analizar_celda(cell_gray)`

**Qué hace**: segmenta caracteres dentro de una celda (Legajo, Nombre,
Parcial) y cuenta cuántas palabras hay usando los gaps horizontales
entre bounding boxes.

### Línea por línea

| Lín | Código | Estado | Cita oficial |
|----:|--------|:------:|---|
| 78-84 | Margen interno `inner = cell_gray[2:-2, 2:-2]` | 🟢 | U1 numpy slicing |
| 87 | `cell_bin = (inner < 120).astype(np.uint8)` | 🟡 | Reemplazar por `cv2.threshold`+`THRESH_OTSU` (consistencia con Bloque 1) |
| 89 | `cv2.connectedComponentsWithStats(cell_bin, connectivity=8)` | 🟢 | `PDI_U3_Segmentacion.py:300` + PDF U3 "Etiquetado de Componentes Conectadas" |
| 93 | `valid_stats = [s for s in stats[1:] if s[4] >= 3]` | 🟢 | Descarta fondo + ruido por área mínima |
| 98 | `sorted(valid_stats, key=lambda s: s[0])` | 🟢 | Python básico |
| 102 | `char_crops = [cell_bin[s[1]:s[1]+s[3], s[0]:s[0]+s[2]] for s in valid_stats]` | 🟢 | U1 slicing, alimenta al Bloque 3 |
| 105-108 | Conteo de palabras por gap, `space_threshold = 7` | 🟡 | Número mágico — defender como constante calibrada experimentalmente (mediana de inter-letra vs inter-palabra) |

### Decisiones de rediseño (con cita textual)

1. **Binarización → `cv2.threshold`+`THRESH_OTSU`**.
   - Cita `PDI_U3_Segmentacion.py:624`:
     ```python
     _, img_th = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
     ```
   - Para celdas con texto muy fino, igual se puede aplicar
     `cv2.THRESH_BINARY_INV` si se invierte el resultado; lo dejamos
     igual al Bloque 1 por consistencia.

2. **Constante `7` → calibración empírica**.
   - Confirmado contra los 5 PDFs y los `.py` de U1/U2/U3/Practica:
     **no hay material oficial** sobre segmentación de texto por gaps.
   - Defendible: medir la distribución de gaps en celdas conocidas de la
     planilla 1 (Legajos de 1 palabra, Nombres de 2+ palabras) y poner
     el threshold en la mediana entre los dos clusters. Esto es K-medias
     1D implícito + estadística elemental.
   - Implementación: `calibrar_space_threshold(img_gray, rows, cols,
     n_samples=N)` que devuelve el threshold. Se aplica igual al resto.

### Qué estudiarse para defender en oral

- **Conectividad 4 vs 8**: 4 son los vecinos cardinales (arriba, abajo,
  izq, der); 8 incluye las diagonales. Para texto impreso conectividad 8
  suele ser la correcta. Cita: PDF U3 "Etiquetado de Componentes
  Conectadas" — *"Un camino desde p1 a pn podrá ser 4-conectado u
  8-conectado"*.
- **Por qué filtrar por área**: una mota de polvo o un pixel suelto se
  vuelve una "componente" con área 1-2 y rompe el conteo.
- **Por qué ordenar por X**: la lectura del texto es de izquierda a
  derecha; si no ordenás, los caracteres vienen en orden de detección
  (que es impredecible).
- **Por qué un threshold de 7 px para el gap**: lo calibramos
  empíricamente. Un pixel arriba o un pixel abajo puede romper palabras
  en dos.

---

## Bloque 3 — Clasificador topológico L / R / A (`clasificar_condicion_final`)

**Qué hace**: dado el recorte binario del carácter de Condición Final,
clasifica L (Libre, sin huecos), R (Recupera, 1 hueco con masa arriba) o
A (Aprobado, 1 hueco con masa más centrada).

### Línea por línea

| Lín | Código | Estado | Cita oficial |
|----:|--------|:------:|---|
| 119-121 | `if h < 4 or w < 3: return "DESCONOCIDO"` | 🟢 | Sentido común |
| 124-127 | `inv = np.pad(1 - char_crop, 1, ...)` + `connectedComponentsWithStats(..., connectivity=4)` | 🔴 | Truco clásico de topología. No aparece en el material oficial. Reemplazar por `findContours(..., RETR_TREE)` + jerarquía. |
| 129-131 | `if huecos <= 0: return "L"` | 🟢 | Conceptual OK |
| 133-143 | Ratio `top_mass / bot_mass > 0.9` → R, si no A | 🟡 | Número mágico 0.9. Reemplazar por `cv2.moments` para obtener centro de masa vertical oficial. |

### Decisiones de rediseño (con cita textual)

1. **Conteo de huecos → `cv2.findContours` con `RETR_TREE`** (reemplaza el truco del invertido).
   - Cita oficial `PDI_U3_Segmentacion.py:392-394`:
     ```python
     contours, hierarchy = cv2.findContours(f, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
     ```
   - Cita PDF U3 "Contornos": la jerarquía es `[Next, Previous, First_Child, Parent]`. Un contorno con `parent == -1` es **externo**; sus `first_child` son los **huecos**.
   - Mi versión: `idx_ext = next(i for i, h in enumerate(hierarchy[0]) if h[3] == -1); n_huecos = sum(1 for h in hierarchy[0] if h[3] == idx_ext)`.

2. **Distribución vertical → `cv2.moments`** (reemplaza el ratio manual).
   - Cita oficial `PDI_U3_Segmentacion.py:496-497`:
     ```python
     M = cv2.moments(cnt)
     huMoments = cv2.HuMoments(M)
     ```
   - Mi versión: `cx = M["m10"]/M["m00"]` (centro de masa X) y `cy = M["m01"]/M["m00"]` (centro de masa Y). Si `cy` cae arriba de la mitad → R, si no → A. Sin umbral 0.9.

### Qué estudiarse para defender en oral

- **Idea conceptual** de L/R/A: L no tiene huecos cerrados, R tiene 1
  hueco arriba, A tiene 1 hueco centrado.
- **`findContours` con `RETR_TREE`**: te da la **jerarquía** de objetos.
  El exterior es el contorno con `parent == -1`. Sus hijos son huecos.
- **`cv2.moments`**: oficial de U3, da centro de masa y momentos
  centrales. Reemplaza al corte a mano.
- **Por qué `connectivity=4` y no 8**: más estricto; en contornos
  delgados (texto) suele alcanzar. Pero en este script **se usa 4**, lo
  que podría romper con tipografías pixel-art. Pasar a 8 es más
  consistente con `PDI_U3_Segmentacion.py:300`.

### ⚠️ Cuestión de fondo

Contar huecos para clasificar letras **es conceptualmente correcto**,
pero **no es material oficial** de U3 ni de Práctica. Si el profe te
pregunta "¿de dónde sale contar huecos para distinguir letras?",
respondés: "Es la propiedad topológica principal. En visión por
computadora es estándar usar el **número de Euler** que cuenta
componentes - huecos. Como en este curso vimos
`connectedComponentsWithStats` y `findContours` con jerarquía, usamos
esos como base."

---

## Bloque 4 — Orquestación + I/O (`procesar_planilla` + `renderizar_imagen_no_aprobados` + `ejecutar_validacion_todas_las_planillas`)

**Qué hace**: el orquestador. Aplica Bloques 1+2+3 a cada registro,
acumula para el CSV y arma la imagen final con las tarjetas de los
alumnos L/R.

### Sub-bloques

#### 4a — `procesar_planilla` (líneas 146-253)

| Elemento | Líneas | Estado | Cita |
|----------|-------:|:------:|------|
| `cv2.imread` + `cvtColor(... BGR2GRAY)` | 153-156 | 🟢 | U1/U3 patrón estándar |
| Asume `num_data_rows = len(rows) - 2` | 159 | 🟡 | Rompe si no son 22 filas |
| Reglas: `leg_ok`, `nom_ok`, `p_ok`, `cond_ok` | 169-198 | 🟢 | Vienen de la consigna del TP |
| Encadenamiento a `clasificar_condicion_final` | 200-203 | 🟢 | Solo si `cond_ok` y `crops_cond` único |
| Acumulación CSV siempre (OK o MAL) | 218-227 | 🟢 | Coherente con consigna |
| Selección no aprobados con `all_ok AND L/R` | 229-236 | 🟢 | Coincide con README |
| `csv.DictWriter` con cabecera | 239-245 | 🟢 | API estándar Python |

#### 4b — `renderizar_imagen_no_aprobados` (líneas 256-324)

| Elemento | Líneas | Estado | Cita |
|----------|-------:|:------:|------|
| `np.ones((h,w,3), uint8) * color` | 268, 276 | 🟢 | U1 creación canvas |
| `cv2.putText` con `FONT_HERSHEY_SIMPLEX` | 269-272, 280-281 | 🟢 | `Unidad2/Código-20250910/PI_U2_ej_video_face_detection_and_blur.py:19` |
| `cv2.rectangle` para encabezado y tarjetas | 279, 290-291, 318 | 🟢 | OpenCV estándar |
| `cv2.resize` para encajar crop | 302 | 🟢 | OpenCV estándar |
| Insignia color L/R | 311-316 | 🟢 | 🟡 comentario BGR confuso |
| Constantes UI (`card_h`, `header_h`, etc.) | 261-264 | 🟡 | "Valores razonables", defendible |

#### 4c — `ejecutar_validacion_todas_las_planillas` (líneas 327-366)

| Elemento | Líneas | Estado | Cita |
|----------|-------:|:------:|------|
| `os.path.dirname(__file__)` + `os.makedirs(exist_ok=True)` | 328-330 | 🟢 | Python estándar |
| Argparse con `--sheets` opcional | 367-372 | 🟢 | API estándar |
| Loop con guard `os.path.exists` | 341-352 | 🟢 | Patrón defensivo |

### Qué mejorarían la defensa

1. **`num_data_rows = len(rows) - 2`** → medir y warning si no coincide.
2. **`cols[1..-1]`** y `rows[1..-2]` hardcodeados → medir `len(cols) == 8`.
3. **Comentario BGR vs RGB** en línea 312 → reemplazar `(50, 50, 220)`
   por `(0, 0, 255)` (rojo puro BGR).

### Qué estudiarse para defender en oral

- **Pipeline en cascada**: imagen → rejilla (Bloque 1) → celdas
  (Bloque 2) → clasificación (Bloque 3) → reporte. Saberlo de memoria.
- **`csv.DictWriter`**: mapea claves del dict a columnas del CSV.
- **`cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)`**: por qué pasar a gris
  para detectar la rejilla — la luminancia distingue líneas blancas
  de oscuras.
- **`with open(...)`**: garantiza cierre del archivo aunque haya error.
- **`os.makedirs(..., exist_ok=True)`**: idempotente.

---

## Resumen del estado de los 4 bloques

- **Bloque 1**: 2 puntos 🔴 (morfología, asumir 8 líneas), 2 puntos 🟡
  (binarizado manual, proyección 1D).
- **Bloque 2**: 1 punto 🟡 (umbral `7` mágico → calibración empírica).
- **Bloque 3**: 1 punto 🔴 (invertido+componentes → `findContours
  RETR_TREE`), 1 punto 🟡 (ratio 0.9 → `cv2.moments`).
- **Bloque 4**: 3 puntos 🟡 menores (num_data_rows=-2, BGR confuso,
  asumir 22 filas).

**Próximo paso**: codear las reescrituras chicas (Bloque 4) y la
calibración del Bloque 2, después pasar a los reemplazos grandes
(Bloques 1 y 3).

## Bloque 3 — Clasificador topológico L / R / A

*(pendiente de repaso)*

## Bloque 4 — Orquestación + I/O (CSV, terminal, imagen resumen)

*(pendiente de repaso)*
