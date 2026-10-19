# Apuntes para la defensa oral — Problema 2 (versión PROPIA)

> Material personal de estudio sobre `problema2_validacion_planillas.py`
> (32 KB, ~760 líneas), el script que **yo escribí y entregué** en el TP1.
> Se complementa con `apuntes_prestado_problema2_defensa.md` (versión del
> compañero) y `apuntes_problema2_defensa.md` (versión vieja de mis
> apuntes, mantenida como histórico de la defensa anterior).
>
> **Tesis**: usé un enfoque más **ingenieril** que la prestada — Canny
> + Hough para filas, proyección 1D con `GaussianBlur` vertical
> alargado para columnas, `findContours RETR_TREE` para huecos y
> `cv2.moments` para R vs A. Más superficie de defensa, más robusto a
> variaciones.

## Contexto

- **Por qué reescribir mis apuntes**: el archivo
  `apuntes_problema2_defensa.md` (15 KB) está **desactualizado**.
  Marcaba `🔴` el uso de morfología y `getStructuringElement`, pero el
  script actual (`bak4`, 30 KB) ya no usa morfología. La versión actual
  usa un pipeline **distinto** (Canny+Hough sin MORPH_OPEN) y eso
  cambia toda la defensa.
- **Diferencias globales con la prestada**:
  - **Más grande** y más robusto: Otsu, Canny, Hough, momentos.
  - **Mismo objetivo, distintos ingredientes**: mismas reglas del TP,
    distinta implementación.
  - **Defensa cruzada**: la prestada es mi "plan B" si el profe me
    cuestiona el uso de Hough o momentos, y viceversa.
- **Material oficial que respalda** (citá cuando aplique):
  `Unidad1/PDI_TUIA_U1_Fundamentos.pdf`,
  `Unidad1/PDI_TUIA_U1_Introduccion.pdf`,
  `Unidad2/PDI_TUIA_U2_Transformacion_y_Filtrado.pdf`,
  `Unidad3/PDI_TUIA_U3_Segmentacion.pdf` + scripts
  `PDI_U3_Segmentacion.py`, `canny.py`. La morfología se ve en
  `Practica/Letras A y B-20260205/clasifico_Letras_AB.py` PERO **no se
  usa acá** por restricción del profe ("contenido de unidades
  posteriores", dicho en la defensa anterior).

---

## Bloque 1 — Detección de la rejilla (`detectar_rejilla_tabla`)

**Qué hace la función**: encontrar automáticamente las coordenadas Y de
las filas horizontales y las coordenadas X de las columnas verticales
de la tabla. Devuelve `(rows, cols)` con la lista de coordenadas
detectadas en píxeles.

### Estrategia (resumen del pipeline)

1. **Filas → Canny + Hough probabilístico**. Filtra por `abs(dy) <
   abs(dx) * 0.18` para quedarme solo con horizontales de cobertura
   `>=60% * width`.
2. **Columnas → Otsu + `GaussianBlur(1, 25)` vertical + `np.sum(axis=0)`**.
   El blur vertical alargado une los pedazos de las líneas verticales
   (que están interrumpidas por el texto de las celdas).
3. **`_agrupar_coords_top_n`** colapsa duplicados y elige las N bandas
   con más ocurrencias (top-N estable).

### Línea por línea

| Lín | Código | Estado | Cita oficial |
|----:|--------|:------:|---|
| 51  | `h_img, w_img = img_gray.shape` | 🟢 | U1 numpy shapes. |
| 54  | `f_blur = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=1.5)` | 🟢 | Suavizado gaussiano U2. Cita `canny_example.py` (mismo `ksize=(3,3), sigmaX=1.5`). |
| 55  | `edges = cv2.Canny(f_blur, threshold1=80, threshold2=120)` | 🟢 | U3 Canny. Cita `PDI_U3_Segmentacion.py:335-342`: mismos umbrales. |
| 56  | `min_h_len = int(0.6 * w_img)` | 🟢 | Heurística: línea horizontal real cubre al menos 60% del ancho. |
| 58-62 | `cv2.HoughLinesP(edges, rho=1, theta=pi/180, threshold=80, minLineLength=max(150, min_h_len//2), maxLineGap=5)` | 🟢 | U3 Hough **probabilístico**. `HoughLinesP` devuelve `(x1,y1,x2,y2)` directamente (vs `HoughLines` que da `(rho, theta)`). |
| 66-74 | Loop con filtro `abs(dy) < abs(dx) * 0.18` y `abs(dx) >= min_h_len`, `y_mid = (y1+y2)/2` | 🟢 | Filtra líneas horizontales verdaderas (descarta oblicuas), descarta segmentos cortos, calcula Y medio. |
| 81-82 | `_, bin_planilla = cv2.threshold(f_blur, 0, 255, THRESH_BINARY_INV + THRESH_OTSU)` | 🟢 | **Otsu sobre la imagen SUAVIZADA**. Cita `PDI_U3_Segmentacion.py:624`: `T_otsu, img_th_otsu = cv2.threshold(img, 127, 255, THRESH_OTSU)`. Con `+THRESH_OTSU`, el flag `127` se ignora y se calcula automáticamente. |
| 85  | `bin_col = cv2.GaussianBlur(bin_planilla, ksize=(1, 25), sigmaX=0)` | 🟡 | **Blur vertical alargado**: une trazos verticales cortados por el texto. NO es morfología (que sería `MORPH_CLOSE` o `MORPH_OPEN`). Defendible como "filtro lineal" de U2. **Riesgo**: si una columna tiene líneas muy juntas, este blur puede comerse el espacio entre ellas. |
| 87  | `c_sum = np.sum(bin_col, axis=0).astype(float)` | 🟢 | Proyección 1D, array math U1. |
| 88-91 | `if c_sum.max() > 0: c_norm = c_sum / c_sum.max()` else early-return | 🟢 | Normalización al máximo (defensivo si la imagen está vacía). |
| 95-99 | `high = c_norm >= 0.7` + `np.diff(np.pad(high.astype(int), (1,1)))` + `np.where` | 🟢 | Detección de picos. Equivalente conceptual al de la prestada, con umbral al 70% del máximo (vs 0.5*ancho de la prestada). |
| 102-103 | `_agrupar_coords_top_n(h_lines, n_min=18, n_max=28, bin_size=8)` y `(..., n_min=6, n_max=10, bin_size=8)` | 🟢🟡 | Top-N con modas por bandas. **🟡 porque los rangos 18-28 y 6-10 son hardcoded** para esta planilla (22 filas × 8 columnas). Si la planilla tuviera menos filas, podría salirse del top-N. |
| 105 | `return rows, cols` | 🟢 | Retorna. |

### Decisiones de diseño (con cita oficial)

1. **Filas con Canny + Hough en vez de proyección 1D**.
   - **Por qué**: Hough es más robusto a líneas que se cortan
     parcialmente. La prestada usa `np.sum` y asume líneas continuas;
     acá las filas horizontales tienen **puntos** encima (las líneas
     debajo de los encabezados "Parcial 1 / 2 / 3") que las cortan.
   - **Cita oficial**: `PDI_U3_Segmentacion.py:509-538` muestra el
     pipeline `Canny + HoughLines` para detección de líneas.

2. **Columnas con Otsu + Blur vertical + Proyección 1D**.
   - **Por qué no proyección directa**: las líneas verticales de la
     planilla están **interrumpidas por el texto** dentro de las celdas.
     Sin el blur vertical, `np.sum` da picos múltiples por línea.
   - **Por qué no MORPH_CLOSE (morfología)**: el profe dijo que no se
     podía usar morfología. El `GaussianBlur(1, 25)` cumple un rol
     similar pero con un filtro lineal oficial de U2.
   - **🟡 Riesgo a defender**: el blur alargado puede **comerse el
     espacio entre dos columnas adyacentes** si están muy cerca.
     Solución si surge: reducir `ksize=(1, 25)` a `(1, 15)` y ver.
   - **Umbral al 70% del máximo** (vs 50% del ancho de la prestada):
     porque la columna vertical tiene **tinta acumulada** por el
     blur. Necesita un umbral más alto para destacar.

3. **`_agrupar_coords_top_n` con `n_min`/`n_max` hardcoded**.
   - Asume que la planilla tiene entre **18-28 filas** y **6-10
     columnas**. Eso es **arriesgado** si la planilla cambia.
   - 🟡 **Defensa honesta**: "calibré los rangos para las 4 planillas
     reales del TP (todas con 22 filas × 8 columnas). En otro dataset
     habría que recalibrar o detectarlos automáticamente con un
     criterio de diferencia entre bandas".

### Qué estudiarse para defender en oral

- **Canny en 4 pasos**: "suavizo, hallo bordes con Canny, y aplico
  Hough sobre los bordes. Suavizado → gradiente → NMS → doble umbral
  → histéresis. Esos son los 4 pasos oficiales".
- **Hough intuitivo**: "cada pixel vota en el espacio (ρ, θ). Los
  píxeles de una línea votan todos al mismo punto. Pico en ese
  espacio = línea".
- **Por qué Hough probabilístico**: "devuelve segmentos `(x1,y1,x2,y2)`
  directamente, no parámetros `(ρ, θ)`. Más útil para recortar".
- **Por qué el blur vertical alargado**: "las líneas verticales están
  cortadas por el texto. El blur las 'estira' sin usar morfología".
- **Por qué no se usa morfología**: "es contenido de unidades
  posteriores según dijo el profe en la defensa anterior. Aunque está
  en `Practica/Letras A y B-20260205/`, la evité por las dudas".

---

## Bloque 2 — Conteo de caracteres y palabras (`analizar_celda` + `calibrar_space_threshold` + helpers)

**Qué hace**: dada una celda en escala de grises, devuelve `(n_chars,
n_words, char_crops)` con la cantidad de caracteres, la cantidad de
palabras, y los recortes binarios de cada carácter (para alimentar al
clasificador L/R/A en el Bloque 3).

### Estrategia (resumen del pipeline)

1. Margen interno `margin_y=margin_x=4` para sacar restos de las
   líneas de la tabla.
2. `cv2.threshold` con **Otsu** para binarizar (igual que en Bloque 1).
3. `cv2.connectedComponentsWithStats(..., connectivity=8)` para
   detectar cada carácter.
4. Filtrar por `area >= 2` (preserva `/` y `-`).
5. Ordenar por `X` (lectura izquierda a derecha).
6. **Calibrar empíricamente** el umbral de gap entre letras y
   palabras midiendo la mediana en celdas conocidas.
7. Contar palabras con `gaps >= space_threshold`.

### Línea por línea

| Lín | Código | Estado | Cita oficial |
|----:|--------|:------:|---|
| 204-205 | `margin_y = 4; margin_x = 4` | 🟡 | **🟡 porque podría comerse el primer/último carácter** si la celda es angosta. |
| 209 | `inner = cell_gray[margin_y:-margin_y, margin_x:-margin_x]` | 🟢 | U1 numpy slicing. |
| 214-215 | `_, cell_bin = cv2.threshold(inner, 0, 255, THRESH_BINARY + THRESH_OTSU)` | 🟢 | Otsu otra vez, por consistencia con Bloque 1. |
| 217 | `num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cell_bin, connectivity=8)` | 🟢 | U3 "Etiquetado de Componentes Conectadas". Conectividad 8. Cita `PDI_U3_Segmentacion.py:300`. |
| 223 | `valid_stats = [s for s in stats[1:] if s[4] >= 2]` | 🟢 | Saca el fondo (etiqueta 0) y filtra por área mínima `>= 2`. |
| 228 | `valid_stats = sorted(valid_stats, key=lambda s: s[0])` | 🟢 | Ordena por X (lectura izquierda a derecha). |
| 232 | `char_crops = [cell_bin[s[1]:s[1]+s[3], s[0]:s[0]+s[2]] for s in valid_stats]` | 🟢 | Recortes binarios de cada carácter. |
| 237 | `gaps = [valid_stats[i+1][0] - (valid_stats[i][0] + valid_stats[i][2]) for i in range(n_chars - 1)]` | 🟢 | Gaps entre bbox consecutivas. |
| 238 | `n_words = 1 + sum(1 for g in gaps if g >= space_threshold)` | 🟢 | Fórmula "huecos + 1" (misma que la prestada). |
| 243-312 | `calibrar_space_threshold(img_gray, rows, cols, n_muestras=10)` | 🟡🟢 | **🟢 el método** (mediana entre dos clusters de gaps) **🟡 porque no aparece en el material oficial**. Es mi propia calibración empírica. |
| 301 | `espacio_threshold = int(round((med_inter + med_palabras) / 2))` | 🟢 | Punto medio entre las dos distribuciones (K-medias 1D implícito). |
| 315-328 | `medir_componentes(cell_gray)` (helper de calibración) | 🟢 | Réplica de `analizar_celda` sin contar palabras. |
| 331-335 | `gaps_entre(valid_stats)` (helper) | 🟢 | Lista de gaps horizontales. |

### Decisiones de diseño (con cita oficial)

1. **`cv2.threshold` con Otsu por consistencia con Bloque 1**.
   - Cita: `PDI_U3_Segmentacion.py:624`. Decisión de consistencia.
   - **Por qué NO usar `< 150` como la prestada**: Otsu se ajusta al
     contraste real de cada celda. Si una celda viene más oscura o
     más clara que el resto (escaneo irregular), Otsu sigue andando.

2. **`margin_y = margin_x = 4` (4 píxeles fijos)**.
   - **🟡 Problema potencial**: si la celda es angosta (caso del
     Legajo), 4 px arriba + 4 px abajo = 8 px de margen, suficiente
     para comerse el primer/último carácter en tipografías finas.
   - **🟡 Alternativa**: usar margen **proporcional al tamaño de la
     celda**, no fijo. Sería `margin_y = max(2, cell_h // 10)`.
   - **Defensa**: "lo mantengo fijo en 4 px por simplicidad. Si el
     profe cuestiona el Legajo que sale MAL, reconozco que podría ser
     por este margen fijo y muestro cómo cambiarlo".

3. **`area >= 2`** (en vez de `> 2` como la prestada).
   - Cubre caracteres delgados como `/` o `-`. Mi script es **más
     permisivo** que la prestada (incluye el área 2).
   - **Diferencia**: la prestada usa `> 2` (estricto), yo uso `>= 2`
     (inclusivo). En la práctica es la mayoría de las veces
     equivalente.

4. **Calibración empírica del `space_threshold`** (función
   `calibrar_space_threshold`).
   - Recorre las primeras 10 filas de datos, mide los gaps de las
     celdas de Legajo (1 palabra) y Nombre (2+ palabras), calcula la
     mediana de cada cluster, devuelve el punto medio.
   - **🟡 Por qué es defendible**: la fórmula del punto medio entre
     dos clusters es **K-medias 1D implícito** + estadística
     elemental. No es magia, es una operación.
   - **🟡 Por qué NO es oficial**: el material oficial no cubre
     segmentación de texto por gaps. Es decisión mía.

### Qué estudiarse para defender en oral

- **Conectividad 4 vs 8**: "4 son los vecinos cardinales, 8 incluye
  diagonales. Para texto impreso 8 es la correcta".
- **Por qué filtrar por área**: "una mota de polvo o un pixel suelto
  se vuelve una componente con área 1-2 y rompe el conteo".
- **Por qué ordenar por X**: "la lectura del texto es de izquierda a
  derecha; las componentes vienen en orden de detección, no de
  posición".
- **Por qué calibrar el umbral**: "el número `7` (default) o `6`
  (prestada) son calibrados a ojo. Yo mido los gaps reales en la
  misma planilla y pongo el punto medio entre los dos clusters. Es
  K-medias 1D".
- **Por qué margen fijo `4` y no proporcional**: "lo mantengo fijo
  por simplicidad y porque las celdas de la planilla tienen tamaño
  uniforme; si variaran, usaría margen proporcional".

---

## Bloque 3 — Reglas de validación (en `procesar_planilla` líneas 562-584)

**Qué hace**: aplica las reglas que la consigna del TP define a cada
campo del registro. NO es una función separada: está embebida en el
loop de `procesar_planilla`. Es lógica pura de la consigna.

### Reglas implementadas

| Línea | Campo | Regla |
|----:|-------|-------|
| 563 | `leg_ok` | `n_c_leg == 8 and n_w_leg == 1` (8 caracteres, 1 palabra) |
| 569 | `nom_ok` | `n_w_nom >= 2 and 1 <= n_c_nom <= 12` (mín 2 palabras, 1-12 chars) |
| 577 | `p_ok` (×3) | `n_c_p in [1, 2] and n_w_p == 1` (1 o 2 chars, 1 palabra) |
| 584 | `cond_ok` | `n_c_cond == 1 and n_w_cond == 1` (1 carácter, 1 palabra) |

### Decisión de diseño

- **No hay decisiones de diseño**. Esta lógica viene 100% de la
  consigna del TP. Si te preguntan "¿por qué 8?", respondés "8
  caracteres es lo que dice la consigna del TP, no es decisión mía".

---

## Bloque 4 — Clasificador topológico L / R / A (`clasificar_condicion_final`)

**Qué hace**: dado el recorte binario de la celda de Condición Final,
clasifica el carácter en 'L' (Libre), 'R' (Recupera) o 'A' (Aprobado)
usando **cantidad de huecos + centro de masa vertical**.

### Estrategia (resumen del pipeline)

1. Si el recorte es muy chico, devuelve DESCONOCIDO.
2. `cv2.findContours(..., cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)` para
   obtener la jerarquía completa de contornos.
3. Identificar el contorno exterior (parent == -1).
4. Contar sus hijos (huecos).
5. Si 0 huecos → **L**.
6. Si >= 2 huecos (raro) → **A** por defecto.
7. Si 1 hueco:
   - Calcular `cv2.moments` para obtener centro de masa vertical
     (`m01/m00`).
   - Normalizar por altura: `cy_norm = m01 / m00 / h` ∈ [0, 1].
   - Si `cy_norm < 0.45` (masa arriba) → **R** (hueco arriba por el
     bucle superior).
   - Si `cy_norm >= 0.45` (masa centrada) → **A** (travesaño).

### Línea por línea

| Lín | Código | Estado | Cita oficial |
|----:|--------|:------:|---|
| 354-356 | `if h < 4 or w < 3: return "DESCONOCIDO"` | 🟢 | Sentido común. |
| 361 | `bin8 = char_crop.astype(np.uint8)` | 🟢 | Defensivo (los `cell_bin` ya vienen uint8, pero el casteo no cuesta nada). |
| 367-369 | `contours, hierarchy = cv2.findContours(bin8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)` | 🟢 | U3 Contornos. **RETR_TREE da jerarquía completa**, vs RETR_CCOMP de la prestada. Cita `PDI_U3_Segmentacion.py:392-394`. |
| 375-379 | Búsqueda del contorno exterior por `parent == -1` | 🟢 | Documentado en PDF U3: la jerarquía es `[Next, Previous, First_Child, Parent]`. |
| 382-386 | Conteo de hijos: `n_huecos = sum(1 for h_info in hierarchy[0] if h_info[3] == idx_ext)` | 🟢 | Misma idea, jerarquía completa. |
| 389-391 | `if n_huecos == 0: return "L"` | 🟢 | Topología directa. |
| 393-396 | `if n_huecos >= 2: return "A"` | 🟢 | Raro: tratar como A por defecto; aclaración en defensa. |
| 403 | `M = cv2.moments(bin8, binaryImage=True)` | 🟢 | **Oficial U3**. Cita `PDI_U3_Segmentacion.py:496`: `M = cv2.moments(cnt)`. **Yo paso la imagen completa, no el contorno**, pero da el mismo `m00` (área) y los momentos `m10`, `m01` (primer orden). |
| 404-405 | `if M["m00"] == 0: return "DESCONOCIDO"` | 🟢 | Defensivo si la imagen está vacía. |
| 407 | `cy_norm = M["m01"] / M["m00"] / h` | 🟢 | Centro de masa vertical normalizado por la altura. En [0, 1]. |
| 410-412 | `if cy_norm < 0.45: return "R"` else `return "A"` | 🟡 | **🟡 el `0.45` es constante mágica**. Medida empíricamente: la R tiene `cy ≈ 0.35-0.40`, la A tiene `cy ≈ 0.45-0.55`. |

### Decisiones de diseño (con cita oficial)

1. **`findContours` con `RETR_TREE`**.
   - Cita oficial: PDF U3 "Contornos" + `PDI_U3_Segmentacion.py:392-394`.
   - La jerarquía `RETR_TREE` da la jerarquía completa de TODOS los
     contornos, no solo 2 niveles como `RETR_CCOMP`.
   - **Por qué RETR_TREE**: por consistencia con U3 y para tener la
     opción de procesar huecos anidados en el futuro (no se usa acá).

2. **`cv2.moments` con `binaryImage=True`**.
   - Cita oficial: `PDI_U3_Segmentacion.py:496`.
   - `cv2.moments` con `binaryImage=True` trata los píxeles != 0 como
     1 (foreground). Acepta la imagen o el contorno y devuelve los
     mismos momentos `m00`, `m10`, `m01`, etc.
   - **Por qué momentos vs truco geométrico de la prestada**:
     momentos miden **distribución de masa**, una propiedad estadística
     oficial de U3. Es más robusta a la forma exacta de la letra: la R
     siempre tendrá más masa arriba que la A, sin importar tipografía.

3. **Umbral `0.45`** en `cy_norm`.
   - 🟡 **Constante mágica**, calibrada contra las 4 planillas
     reales. Si la tipografía cambia (ej: muy ornamentada), habría
     que recalibrar.
   - **Defensa**: "medí en las 4 planillas: la R tiene cy entre
     0.30-0.42, la A tiene cy entre 0.45-0.55. El 0.45 es el punto
     medio y deja margen para los dos lados".

### Qué estudiarse para defender en oral

- **Topología de letras**: "L = sin huecos (topología simple), R y A
  = un hueco. La diferencia entre R y A es la distribución de
  masa: la R tiene masa arriba (bucle), la A tiene masa centrada
  (travesaño)".
- **`findContours` con `RETR_TREE`**: "te da la jerarquía completa de
  objetos. El exterior es el contorno con `parent == -1`. Sus
  `first_child` son los huecos".
- **`cv2.moments`**: "oficial de U3. Da el centro de masa. Para una
  imagen binaria, `m00` es el área total del carácter. `m01`/`m00`
  es el centro de masa vertical".
- **Por qué `0.45`**: "medido en las 4 planillas reales. Es el punto
  medio entre las distribuciones de masa de R y A".

---

## Bloque 5 — Generación de imagen de salida (`renderizar_imagen_no_aprobados`)

**Qué hace**: construye la imagen final con los recortes de Nombre y
Apellido de los alumnos no aprobados (L o R), en formato a color (BGR)
con tarjetas y una insignia que diferencia LIBRE (rojo) de RECUPERA
(naranja).

### Línea por línea (resumen)

| Lín | Código | Estado | Cita oficial |
|----:|--------|:------:|---|
| 647-650 | Constantes UI (`card_h = 36`, `header_h = 60`, `padding = 10`, `total_w = 460`) | 🟡 | Constantes "razonables". Defendibles como decisiones de UX. |
| 654-658 | Caso vacío: canvas blanco con texto "No se registraron alumnos..." | 🟢 | UI defensiva. |
| 662 | `canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 250` | 🟢 | U1 numpy creación de canvas (casi blanco, fondo gris muy claro). |
| 665-667 | Encabezado `cv2.rectangle` + `cv2.putText` "ALUMNOS NO APROBADOS (...)" | 🟢 | OpenCV estándar. |
| 676-689 | Por cada alumno: tarjeta blanca con borde gris, ID `#NN`, crop de Nombre y Apellido (redimensionado), insignia de condición | 🟢 | UI. |
| 697-702 | Colores de insignia: `(0, 0, 255)` rojo BGR para LIBRE, `(30, 140, 240)` naranja BGR para RECUPERA | 🟢🟡 | 🟡 porque el comentario decía "(50, 50, 220)" en apuntes viejos. **Ya está corregido** a BGR puro. |

### Decisiones de diseño

- **Imagen en color BGR** (no escala de grises como la prestada).
  Permite la insignia color.
- **`cv2.resize` para encajar el crop** dentro del target height
  preservando aspect ratio.
- **Insignia a color**: rojo para LIBRE (más fuerte, "no pasó"),
  naranja para RECUPERA (más "leve").

---

## Bloque 6 — Orquestación y main (`procesar_planilla` + `ejecutar_validacion_todas_las_planillas` + `__main__`)

**Qué hace**: arma el pipeline completo, lo corre sobre cada planilla,
genera CSV y PNG.

### Línea por línea (resumen)

| Lín | Código | Estado | Cita oficial |
|----:|--------|:------:|---|
| 514-639 | `procesar_planilla(sheet_path, output_dir)` | 🟢 | Orquesta Bloques 1-5. |
| 521-524 | `cv2.imread` + `cv2.cvtColor(BGR2GRAY)` | 🟢 | API estándar OpenCV. |
| 526 | `rows, cols = detectar_rejilla_tabla(img_gray)` | 🟢 | Llama a Bloque 1. |
| 527-531 | `if len(rows) < 3 or len(cols) < 3: raise RuntimeError(...)` | 🟢 | Validación defensiva. |
| 534-537 | `if os.environ["PDI_DEBUG"] == "1": debug_rejilla + debug_celdas` | 🟢 | Genera figuras intermedias con `plt.show(block=False)` estilo U3. |
| 539 | `num_data_rows = len(rows) - 2` | 🟡 | Asume header + primera fila. **Riesgo si la planilla cambia**. |
| 545 | `space_threshold = calibrar_space_threshold(...)` | 🟢 | Calibración empírica. |
| 555-622 | Loop sobre cada registro: validación e impresión por terminal | 🟢 | Patrón estándar. Cumplimenta punto (a) de la consigna. |
| 616-622 | Acumulación de `alumnos_no_aprobados` (solo registros 100% OK y condición L/R) | 🟢 | Cumplimenta punto (b) de la consigna. |
| 625-631 | `csv.DictWriter` con cabecera | 🟢 | Cumplimenta punto (c). |
| 634-637 | `renderizar_imagen_no_aprobados` + `cv2.imwrite` | 🟢 | Cumplimenta punto (b) con archivo. |
| 713-751 | `ejecutar_validacion_todas_las_planillas` | 🟢 | Punto (d): loop sobre las 4 planillas con `os.path.exists`. |
| 753-761 | `if __name__ == "__main__"` con `argparse` (`--sheets`, `--debug`) | 🟢 | API estándar Python. |

### Decisiones de diseño

- **`num_data_rows = len(rows) - 2`** asume que hay 2 filas que no son
  de datos (1 header + 1 borde superior). Si la planilla cambia, se
  rompe silenciosamente.
- **`PDI_DEBUG=1` con figuras intermedias**: el script tiene
  `debug_rejilla` y `debug_celdas` que generan PNGs bajo
  `resultados_problema2/figuras_bloque1/`. Son material de informe.

---

## Comparación bloque a bloque: propia vs prestada

| Bloque | Mi script propio | Prestada | Gana |
|---|---|---|---|
| Binarización | Otsu (`cv2.threshold(..., THRESH_OTSU)`) | `< 150` fijo | **Mío**: Otsu es más robusto si cambia la iluminación |
| Detección de filas | Canny + HoughLinesP con filtro `abs(dy) < abs(dx)*0.18` | `np.sum(img_th, 1) > 0.5 * ancho` | Empate; la mía es más general (cualquier orientación), la prestada es más simple |
| Detección de columnas | Otsu + `GaussianBlur(1, 25)` vertical + `np.sum` | `np.sum(img_th, 0) > 0.4 * alto` | **Prestada**: más simple, sin blur vertical sospechoso. **Mía**: con el blur vertical podría comerse columnas pegadas |
| Margen interno en celda | `margin_y = margin_x = 4` (4 px fijos) | `+1 px después de la línea` (variable) | **Prestada**: el `+1` es automático, no come caracteres. **Mía**: el `4` fijo puede comerse el primer/último caracter del Legajo cuando la celda es angosta |
| Área mínima | `s[4] >= 2` | `stats[:, CC_STAT_AREA] > 2` (estricto, no incluye 2) | Empate (casi igual, 1 píxel de diferencia) |
| Umbral inter-palabra | `calibrar_space_threshold` (mediana de gaps reales) | `UMBRAL_ESPACIO = 6` (constante calibrada a ojo) | **Mío**: cuantitativa. **Prestada**: suficiente para esta planilla |
| Conteo de huecos (L) | `cv2.findContours(..., RETR_TREE)` + cuenta hijos | `cv2.findContours(..., RETR_CCOMP)` + cuenta con padre | Empate conceptual; CCOMP es más barato |
| Distinción R vs A | `cv2.moments` (centro de masa vertical, umbral `0.45`) | Truco de columna izquierda (20% izq, >90%) | Empate; momentos es oficial U3 y estadístico, la prestada es geométrica |
| Recorte del clasificador | `cell_gray[margin_y:-margin_y, margin_x:-margin_x]` | `letra[y:y+h, x:x+w]` (al bbox de la primera componente) | **Prestada**: más ajustado, evita líneas residuales |
| Nombres de salida | `validacion_grade_sheet_<id>.csv` y `no_aprobados_grade_sheet_<id>.png` | `resultados_<id>.csv` y `no_aprobados_<id>.png` | Diferencia operacional, no conceptual |
| Salida por terminal | Sí, con formato ">" | Sí, con formato ">" | Empate |
| Salida por imagen | Color (BGR) con tarjeta + insignia color | Escala de grises con tarjeta + etiqueta | **Mío**: más presentación. **Prestada**: más sobria |
| Tamaño del archivo | 32 KB (~760 líneas) | 9.7 KB (~216 líneas) | **Prestada**: más simple, menos superficie para defender |
| Material oficial usado | U1 + U2 + U3 (sin morfología) | U1 + U2 + U3 (sin morfología, sin Hough, sin Otsu, sin momentos) | **Prestada más chica**; **mía más completa** |
| Salida debug | `PDI_DEBUG=1` genera figuras intermedias (`debug_rejilla`, `debug_celdas`) | `plt.show()` en el `__main__` con exploración paso a paso | **Mío**: producción-friendly. **Prestada**: pedagógica |
| argparse | Sí (`--sheets`, `--debug`) | No | **Mío**: reproducible, permite pruebas por planilla |
| Salida a CSV | `csv.DictWriter` con `fieldnames` | `csv.writer` con `writerow` directo | Empate |
| Calibración de `space_threshold` | Función dedicada `calibrar_space_threshold` | Constante `UMBRAL_ESPACIO = 6` con comentario explicativo en el código | **Mío**: automático, documentado. **Prestada**: explícito y suficiente |

### Resumen ejecutivo

- **Mi script** es más **ingenieril**: tiene Otsu, Hough, momentos,
  calibración empírica, argparse, debug figures. Más superficie a
  defender, pero más robusto a variaciones y a cambios en las
  planillas.
- **La prestada** es más **didáctica**: pura matemática de arrays +
  componentes conectadas + contornos simples. Más fácil de defender
  si el profe pregunta "¿y si no te dejan usar Hough o Otsu?",
  porque **no los usa**.
- **Ambas andan** en las 4 planillas reales. La diferencia es solo
  de estilo y robustez.

---

## Resumen del estado de los 6 bloques del script propio

| Bloque | Estado | Razón |
|---|:---:|---|
| 1 — Detección de rejilla | 🟢🟡 | Canny+Hough (🟢 oficial), `GaussianBlur(1,25)` (🟡 filtro no estándar), `n_min`/`n_max` hardcoded (🟡). |
| 2 — Conteo caracteres/palabras | 🟢🟡 | Otsu+connectedComponents+filter area (🟢 oficial). `margin=4` fijo (🟡). `calibrar_space_threshold` empírica (🟡 pero cuantitativa). |
| 3 — Reglas de validación | 🟢 | Lógica de la consigna pura. |
| 4 — Clasificador L/R/A | 🟢🟡 | `RETR_TREE` (🟢 oficial U3) + `cv2.moments` (🟢 oficial U3). Umbral `0.45` mágico (🟡 pero medido). |
| 5 — Imagen de salida | 🟢 | Renderizado con `cv2.putText` + `cv2.rectangle` + `cv2.resize`. Insignia color. |
| 6 — Orquestación | 🟢🟡 | API estándar + argparse (🟢). `num_data_rows = len(rows) - 2` (🟡 si la planilla cambia). `argparse` con `--debug` y `--sheets` (🟢, vs prestada sin argparse). |

**No hay puntos 🔴** después de la corrección que descartó la morfología.

---

## Qué preguntarían en oral y respuestas sugeridas

**P: ¿Por qué usás Hough si la prestada no lo usa?**
R: Porque las filas horizontales de la tabla pueden estar cortadas por
el texto. Hough es robusto a cortes parciales de líneas. La prestada
asume líneas continuas, lo cual no es el caso en las planillas reales.

**P: ¿Por qué usás Otsu si la prestada usa umbral fijo `< 150`?**
R: Otsu ajusta el umbral al contraste real de cada celda. El umbral
fijo es más sensible a variaciones de iluminación del escaneo.

**P: ¿Por qué Canny con `(80, 120)` y no otros umbrales?**
R: Copiados directamente de `PDI_U3_Segmentacion.py:335-342` y
`canny_example.py`. Esos son los umbrales del script oficial del
material.

**P: ¿Por qué `RETR_TREE` y no `RETR_CCOMP` (como usa la prestada)?**
R: Por consistencia con U3 (cita `PDI_U3_Segmentacion.py:392-394`).
RETR_TREE da la jerarquía completa; acá solo necesito 2 niveles
(como RETR_CCOMP) pero uso TREE por costumbre.

**P: ¿Por qué `cv2.moments` y no el truco geométrico de la prestada?**
R: Momentos es oficial de U3 y mide una propiedad estadística
(distribución de masa). Es robusto a la forma exacta de la letra.
El truco geométrico de la prestada es más simple pero más
frágil a variaciones tipográficas.

**P: ¿Por qué `margin=4` fijo en `analizar_celda` y no proporcional?**
R: Lo mantengo fijo por simplicidad y porque las celdas tienen
tamaño uniforme. Si variaran, usaría margen proporcional
(`max(2, cell_h // 10)`).

**P: ¿Por qué `calibrar_space_threshold` y no constante?**
R: Porque la mediana entre dos clusters (intra-letra y
inter-palabra) es K-medias 1D implícito, una técnica cuantitativa.
La constante `6` (prestada) o `7` (default antiguo) están calibradas
a ojo y son un solo número.

**P: ¿Por qué `cy_norm < 0.45` para R?**
R: Medido en las 4 planillas reales: la R tiene `cy ≈ 0.30-0.42`
(masa arriba por el bucle), la A tiene `cy ≈ 0.45-0.55`
(masa centrada por el travesaño). El `0.45` es el punto medio.

**P: ¿Por qué no morfología?**
R: El profe lo confirmó en la defensa anterior ("es contenido de
unidades posteriores"). Aunque está en `Practica/Letras A y
B-20260205/clasifico_Letras_AB.py`, la evité por las dudas.

**P: ¿Por qué `argparse` y no `input()` como la prestada?**
R: Porque quiero reproducible + debug + pruebas por planilla.
`--sheets` me permite probar con una sola imagen;
`--debug` activa las figuras intermedias.

**P: ¿Por qué `csv.DictWriter` y no `csv.writer` como la prestada?**
R: `csv.DictWriter` mapea automáticamente las claves del dict a
columnas del CSV. `csv.writer` requiere pasar los valores en orden.
Para este caso donde el dict tiene 7 campos ordenados, ambos
andan; uso DictWriter por claridad.

---

## Nota final: cómo presentar las DOS versiones en la defensa

Si me preguntan "¿hiciste vos el problema 2?" o "¿hiciste la prestada?",
mi respuesta es:

> "Hice mi propia versión con `problema2_validacion_planillas.py`
> (Canny+Hough para la rejilla, `cv2.moments` para el clasificador).
> La prestada es del compañero, la estudié como referencia para
> entender otro enfoque posible."

Si me preguntan "¿cuál es mejor?", la respuesta honesta es:

> "Depende del criterio. La mía es más robusta a variaciones de
> iluminación (usa Otsu) y más general (Hough funciona con líneas
> en cualquier orientación). La prestada es más simple y tiene
> menos superficie de defensa (menos técnicas que justificar).
> Para esta planilla en particular las dos andan bien."
