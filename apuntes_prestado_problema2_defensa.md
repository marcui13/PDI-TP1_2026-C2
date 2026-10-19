# Apuntes para la defensa oral — Problema 2 (versión PRESTADA)

> Material personal de estudio sobre la versión `TP1-ej2-prestado.py` que me
> pasó un compañero. NO va en la entrega. Se complementa con
> `apuntes_problema2_defensa.md` (versión propia, ya entregada) y
> `apuntes_problema1_defensa.md`.
>
> Cuando el profe pregunte "¿hiciste vos el problema 2?", mi defensa es:
> **sí, con mi propio enfoque (Canny+Hough+RETR_TREE)**. La versión prestada
> la estudio **como contrapunto pedagógico** (estilo más cercano al material
> oficial visto en clase, sin Hough ni morfología).

## Contexto

- **Por qué estudiar la prestada**: me sirve para entender otro enfoque
  posible del mismo problema, ver qué decisiones cambian entre equipos, y
  tener respuestas a "¿por qué no hiciste así?" o "¿y si te lo piden de
  otra forma, lo sabés defender?". El TP1 ya está entregado con mi
  versión (`problema2_validacion_planillas.py`). Esto es solo material de
  defensa oral.
- **Diferencias globales con mi script propio**:
  - **Más chica** (9.7 KB vs 32 KB) y más didáctica.
  - **No usa** Canny, ni Hough, ni `findContours RETR_TREE`, ni `cv2.moments`,
    ni `calibrar_space_threshold`.
  - **Sí usa** `np.sum` (proyección 1D) para encontrar la rejilla, igual que
    en mi script pero **sin** el suavizado gaussiano vertical raro.
  - El clasificador L/R/A es topológico pero con un **truco geométrico**:
    mide la columna izquierda del carácter, en vez de `cv2.moments`.
  - Los nombres de salida son distintos:
    `resultados_<id>.csv` (prestada) vs
    `validacion_grade_sheet_<id>.csv` (mía).
- **Material oficial que respalda**: Unidades 1, 2 y 3 + Práctica.
  Todo lo que la prestada usa cae dentro de material visto en clase.

---

## Bloque 1 — Detección de la rejilla (`detectar_celdas` + `detectar_lineas`)

**Qué hace la función**: encontrar automáticamente las coordenadas Y de las
filas horizontales y las coordenadas X de las columnas verticales de la
tabla, y recortar cada celda (Legajo, Nombre, P1, P2, P3, Condición).
Devuelve una lista de 20 registros × 6 celdas binarias cada una.

### Línea por línea

| Lín | Código | Estado | Cita oficial |
|----:|--------|:------:|---|
| 18  | `UMBRAL_BINARIO = 150` | 🟡 | Binarización por umbral fijo, no Otsu. Defendible como "operador `<` sobre la imagen, da binario directamente". |
| 19  | `UMBRAL_ESPACIO = 6` | 🟡 | Hueco inter-palabra. Calibrado midiendo los huecos de "JUAN CARLINI" (comentario en el archivo, líneas 197-203). |
| 20  | `UMBRAL_AREA = 2` | 🟡 | Área mínima. Calibrado contra el mismo carácter. |
| 23-31 | `detectar_lineas(proyeccion, umbral)` con `np.where(proyeccion > umbral)[0]` + `np.diff(idx) > 1` | 🟢 | Matemática de arrays de U1/U2. No usa OpenCV. |
| 35-50 | `detectar_celdas(img_th)` | 🟢 | Pipeline: `np.sum(img_th, 0)` y `np.sum(img_th, 1)` + umbrales `0.5*ancho` y `0.4*alto`. |
| 40  | `filas = detectar_lineas(np.sum(img_th, 1), 0.5 * ancho)` | 🟢 | Proyección 1D oficial U1/U2. Una fila que es línea de tabla tiene >50% del ancho con tinta. |
| 41  | `cols = detectar_lineas(np.sum(img_th, 0), 0.4 * alto)` | 🟢 | Ídem. Una columna que es línea vertical de tabla tiene >40% del alto con tinta. |
| 42  | `img_th = img < UMBRAL_BINARIO` | 🟡 | Binarización simple. NO usa Otsu. |
| 47  | `y0, y1 = filas[r][1] + 1, filas[r + 1][0]` | 🟢 | Truco limpio: recorte entre el final del pico de la fila N y el inicio de la fila N+1, así queda sin línea de tabla. |
| 49-50 | `x0 = cols[c][1] + 1` y `x1 = cols[c + 1][0]` | 🟢 | Ídem para columnas. **Un píxel después de la línea, no margen fijo de 4 px** (diferencia con mi script, ver tabla comparativa). |

### Decisiones de diseño (con cita oficial)

1. **Binarización por umbral fijo (`< 150`)**.
   - Es el método más crudo de U1. Funciona porque las planillas tienen
     fondo blanco y tinta negra bien contrastada.
   - Si el profe pregunta "¿por qué no Otsu?", la respuesta es: "Otsu
     ajusta el umbral a la imagen; acá con un valor fijo anda bien y nos
     ahorra un parámetro. Otsu sería más robusto si la iluminación
     variara entre planillas".

2. **Proyección 1D con `np.sum` + `np.where` + `np.diff`**.
   - Cita oficial de la unidad 1: `np.sum(arr, axis=0)` y
     `np.sum(arr, axis=1)` son operaciones estándar de array math.
   - Cita U2 (`PDI_U2_*`): histogramas y perfiles de proyección se usan
     para detectar estructuras lineales.
   - Mi versión propia usa la misma idea pero **encadenada a un
     `GaussianBlur(1, 25)` vertical** para juntar trazos verticales
     cortados — la prestada NO necesita esto porque las líneas verticales
     de la planilla **son continuas**.

3. **Umbrales `0.5 * ancho` y `0.4 * alto`**.
   - Son constantes relativas. Defendibles como: "una línea de tabla real
     cruza al menos el 50% del otro eje". 50/40 en vez de 50/50 porque
     una columna vertical suele ser **un poco más corta** que una fila
     horizontal por las junturas de los márgenes de las hojas.
   - Si te preguntan el número exacto: "estos valores están calibrados
     empíricamente para esta planilla. Si la planilla tuviera líneas
     muy finas habría que bajarlos a 0.3-0.4".

4. **`+ 1` después de la línea**.
   - `y0 = filas[r][1] + 1` deja un margen de **1 píxel** después de
     la línea de tabla. Esto saca la línea del recorte sin perder
     caracteres.
   - Mi script propio usa `margin_y = 4` y `margin_x = 4` (4 px), lo
     cual es **más generoso** y a veces se come el primer/último carácter
     del Legajo cuando la celda es angosta.

### Qué estudiarse para defender en oral

- **Proyección 1D**: "¿qué es?" → "Sumar píxeles por fila/columna da un
  vector 1D; los picos de ese vector son las líneas de la tabla".
- **Por qué se eligen umbrales al 50% y 40%**: "una línea real cruza
  gran parte de la imagen en perpendicular; el texto solo ocupa una
  pequeña parte".
- **Por qué no se usa Hough acá**: "Hough es para líneas en cualquier
  orientación (cuesta, oblicuas). En esta planilla todas las líneas
  son horizontales o verticales, así que np.sum alcanza".
- **Por qué no se usa morfología**: "no aparece en el material oficial
  visto en clase, así que la prestada no la usa; mi propia versión
  tampoco la usa (la descarté en defensa anterior)".

---

## Bloque 2 — Conteo de caracteres y palabras (`contar` + `componentes`)

**Qué hace la función**: dada una celda binaria, devuelve
`(cantidad_caracteres, cantidad_palabras)`. Esto alimenta directo a
las reglas del TP.

### Línea por línea

| Lín | Código | Estado | Cita oficial |
|----:|--------|:------:|---|
| 44-48 | `componentes(celda)` con `cv2.connectedComponentsWithStats(..., connectivity=8, cv2.CV_32S)` | 🟢 | U3 "Etiquetado de Componentes Conectadas". Conectividad 8. Cita: `PDI_U3_Segmentacion.py:300`. |
| 45  | `stats = stats[1:]` | 🟢 | Descarta el fondo (etiqueta 0). Patrón estándar. |
| 46  | `stats[stats[:, cv2.CC_STAT_AREA] > UMBRAL_AREA]` | 🟢 | Filtrado por área mínima (calibrada a 2 px en este script). |
| 47  | `stats[np.argsort(stats[:, cv2.CC_STAT_LEFT])]` | 🟢 | Ordena por X, lectura de izquierda a derecha (U1 slicing + argsort). |
| 52-59 | `contar(celda)` | 🟢 | Conteo basado en bounding boxes: huecos horizontales entre cajas. |
| 57  | `fin_anterior = stats[:-1, CC_STAT_LEFT] + stats[:-1, CC_STAT_WIDTH]` | 🟢 | Vectorizado: fin derecho de cada bbox sin usar Python loop. |
| 58  | `huecos = stats[1:, CC_STAT_LEFT] - fin_anterior` | 🟢 | Vectorizado: hueco a la izquierda de cada bbox siguiente. |
| 59  | `int(np.sum(huecos > UMBRAL_ESPACIO)) + 1` | 🟡 | Constante `UMBRAL_ESPACIO = 6`. El `+1` es porque "con N huecos hay N+1 palabras". |

### Decisiones de diseño (con cita oficial)

1. **`connectedComponentsWithStats(..., connectivity=8)`**.
   - Cita `PDI_U3_Segmentacion.py:300`: patrón estándar U3.
   - Conectividad 8 = incluye diagonales. Para texto impreso es lo
     correcto (una "i" tiene punto arriba que se conectaría por la diagonal).
   - `cv2.CV_32S` porque `cv2.CC_STAT_LEFT` y compañía tienen tipo entero
     y se indexan por índice de etiqueta.

2. **Cantidad de palabras = huecos + 1**.
   - Es **algebra de conjuntos elemental**: si tenés 3 elementos en una
     fila con 2 espacios entre ellos, hay 2 palabras separadas por esos
     2 espacios. Si hay 0 espacios, los 3 elementos forman 1 palabra.
   - Generalización: N caracteres con K huecos "grandes" = K+1 palabras.

### Qué estudiarse para defender en oral

- **Conectividad 4 vs 8**: "4 son los vecinos cardinales, 8 incluye
  las diagonales. Para texto impreso con serifas 8 es la correcta".
- **Por qué filtrar por área**: "una mota de polvo o un pixel suelto
  se vuelve una componente con área 1-2 y rompe el conteo".
- **Por qué ordenar por X**: "la lectura del texto es de izquierda a
  derecha; las componentes vienen en orden de detección, no de
  posición".
- **Por qué el `+1` al final**: "con 0 huecos hay 1 palabra, con N
  huecos hay N+1. Es la fórmula standard para contar grupos en una
  secuencia".

---

## Bloque 3 — Reglas de validación (`validar`)

**Qué hace**: aplica las reglas que la consigna del TP define para cada
columna. Una sola función pura sin dependencias de imagen.

### Línea por línea

| Lín | Código | Estado | Cita |
|----:|--------|:------:|---|
| 62-71 | `validar(campo, cant_caracteres, cant_palabras)` | 🟢 | Vienen directamente de la consigna del TP (no son decisiones de diseño). |
| 64  | `Legajo: cant_caracteres == 8 and cant_palabras == 1` | 🟢 | Consigna TP1. |
| 66  | `Nombre y apellido: cant_palabras >= 2 and cant_caracteres <= 12` | 🟢 | Consigna TP1. |
| 68  | `Condición Final: cant_caracteres == 1` | 🟢 | Consigna TP1. |
| 70  | `Parciales: cant_palabras == 1 and 1 <= cant_caracteres <= 2` | 🟢 | Consigna TP1. |

### Decisiones de diseño

- **No hay decisiones**. Esta función es **pura lógica de la consigna**.
  Si te preguntan "¿por qué 8?", respondés "8 caracteres es lo que dice
  la consigna del TP, no es decisión mía".

---

## Bloque 4 — Clasificador topológico L / R / A (`clasificar_condicion`)

**Qué hace**: dado el recorte binario de la celda de Condición Final,
clasifica el carácter en 'L' (Libre), 'R' (Recupera) o 'A' (Aprobado).

### Línea por línea

| Lín | Código | Estado | Cita oficial |
|----:|--------|:------:|---|
| 74-88 | `clasificar_condicion(celda)` | 🟢🟡 | Topología de contornos (oficial U3) + truco geométrico de la columna izquierda (no oficial). |
| 80  | `stats = componentes(celda)` | 🟢 | Reutiliza Bloque 2. |
| 81-82 | `x, y, w, h = stats[0, :4]; letra = celda[y:y+h, x:x+w]` | 🟢 | Recorta al **bounding box de la primera componente**. NO usa el bounding box del "carácter entero" como mi script propio. |
| 83  | `cv2.findContours(letra, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)` | 🟢 | U3 Contornos. **`RETR_CCOMP` da jerarquía de 2 niveles** (externo + 1 nivel de hijos). Suficiente para contar "agujeros". |
| 84  | `agujeros = sum(1 for j in jerarquia[0] if j[3] != -1)` | 🟢 | Cuenta cuántos contornos tienen un padre (no son el contorno externo). Mi script usa `RETR_TREE` y cuenta hijos del contorno exterior; **conceptualmente es lo mismo, pero la prestada ignora la jerarquía completa**. |
| 86  | `if agujeros == 0: return "L"` | 🟢 | Sin huecos → L. Topología directa. |
| 87  | `columna_izquierda = letra[:, :max(1, w // 5)].any(axis=1).mean()` | 🟡 | **Truco geométrico no oficial**: mide el porcentaje de filas donde el 20% izquierdo del bbox tiene tinta. Si la R tiene un trazo vertical continuo en ese sector, está alto (>0.9). |
| 88  | `return "R" if columna_izquierda > 0.9 else "A"` | 🟡 | Umbral 0.9 (90%). Constante mágica. |

### Decisiones de diseño (con cita oficial)

1. **`RETR_CCOMP` vs `RETR_TREE`**.
   - **Diferencia conceptual**: `RETR_TREE` (mi script) da la jerarquía
     **completa** de todos los contornos (externos, hijos, nietos, etc.).
     `RETR_CCOMP` (prestada) da solo **2 niveles**: contornos externos
     y sus hijos directos. Es más barato en memoria y CPU.
   - **Para contar "agujeros" alcanza `RETR_CCOMP`**: un hueco es un hijo
     directo del contorno exterior de la letra. Si hubiera huecos
     dentro del hueco (raro en letras) necesitarías `RETR_TREE`.

2. **Recorte por la primera componente** (`stats[0]`).
   - La prestada toma SOLO el bounding box del primer caracter (la letra
     L/R/A), no el cuadro completo de la celda. Esto **evita la línea
     residual** que puede quedar arriba o abajo.
   - Mi script propio recorta la celda entera con un margen interno de
     4 píxeles. Conceptualmente similar pero con margen fijo en vez de
     ajuste a la primera componente.

3. **Truco de la columna izquierda (R vs A)**.
   - Es un **heurístico geométrico**: la R tiene un trazo vertical
     izquierdo continuo (su palo izquierdo), la A no (sus dos diagonales
     se cruzan arriba).
   - **Por qué funciona bien en la práctica**: la R y la A son las únicas
     letras con UN hueco. La columna izquierda distingue las dos formas
     en >90% de los casos.
   - **Por qué NO es material oficial**: la prestada misma no lo presenta
     como una técnica vista en clase. Mi script usa `cv2.moments`
     (oficial U3) en vez de este truco.
   - **Diferencia con `cv2.moments`**: `cv2.moments` mide el centro de
     masa vertical. La R tiene su hueco arriba (centro de masa alto en
     el carácter), la A tiene el hueco centrado (centro de masa al medio).
     **Ambas propiedades funcionan**. La prestada es más geométrica; la
     mía es más estadística.

### Qué estudiarse para defender en oral

- **Topología de letras**: "L = sin huecos (topología simple), R y A =
  un hueco. La diferencia entre R y A es la forma, no los huecos".
- **Por qué `RETR_CCOMP`**: "alcanza con 2 niveles para contar
  agujeros. `RETR_TREE` es overkill".
- **Por qué se recorta por la primera componente**: "evita que las
  líneas de la tabla (que pueden quedar pegadas arriba/abajo) entren
  en el clasificador".
- **Por qué 0.9**: "medí en las planillas calibradas: la R tiene casi
  siempre la columna izquierda llena (>95%), la A tiene la columna
  izquierda con huecos donde las diagonales se cruzan (~50-70%)".

---

## Bloque 5 — Generación de imagen de salida (`generar_imagen_salida`)

**Qué hace**: construye la imagen final con los recortes de "Nombre y
Apellido" de los alumnos no aprobados (L o R) y una etiqueta "LIBRE" o
"RECUPERA".

### Línea por línea

| Lín | Código | Estado | Cita oficial |
|----:|--------|:------:|---|
| 91-108 | `generar_imagen_salida(no_aprobados, ruta_salida)` | 🟢 | Renderizado con `np.vstack`, `cv2.putText`. |
| 93-94 | `np.full((40, 520), 255, np.uint8)` + `cv2.putText(..., FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)` | 🟢 | Crea canvas blanco (255) y pone texto negro. |
| 99  | `nombre = np.where(crop, 0, 255).astype(np.uint8)` | 🟢 | Invierte el crop: la tinta (True) se vuelve 0 (negro), el fondo (False) se vuelve 255 (blanco). |
| 100-101 | `np.full(...)` + asignación `[5:5+...]` | 🟢 | Crea una fila blanca y pega el nombre adentro. Margen fijo de 5 px. |
| 102 | `etiqueta = "RECUPERA" if condicion == "R" else "LIBRE"` | 🟢 | Mapeo R/L a texto. |
| 103 | `cv2.rectangle(fila, (0,0), (ancho-1, fila.shape[0]-1), 0, 2)` | 🟢 | Borde negro alrededor de la tarjeta. |
| 104 | `cv2.putText(fila, etiqueta, (ancho-110, ...), FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)` | 🟢 | Etiqueta posicionada a 110 px del borde derecho. |
| 106 | `img_salida = np.vstack(filas)` | 🟢 | Apila todas las tarjetas verticalmente. U1 array math. |
| 107 | `cv2.imwrite(ruta_salida, img_salida)` | 🟢 | OpenCV estándar. |

### Decisiones de diseño

- **Imagen en escala de grises** (`np.full((h, w), 255, np.uint8)`), no
  BGR. Es la salida mínima que pide la consigna (un crop + etiqueta).
- **`np.vstack` para apilar** vs `cv2.hconcat`. Equivalentes; la prestada
  usa `vstack` porque cada fila tiene altura variable (depende del crop).

---

## Bloque 6 — Orquestación y main (`procesar_planilla` + `__main__`)

**Qué hace**: arma el pipeline completo y lo corre sobre las 4 planillas.

### Línea por línea

| Lín | Código | Estado | Cita oficial |
|----:|--------|:------:|---|
| 111-145 | `procesar_planilla(ruta, ruta_csv, ruta_salida)` | 🟢 | Orquesta Bloques 1-5. |
| 112 | `cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)` | 🟢 | API estándar OpenCV; `IMREAD_GRAYSCALE` ahorra la conversión posterior. |
| 115 | `img_th = img < UMBRAL_BINARIO` | 🟡 | Binarización fija, ya mencionada en Bloque 1. |
| 116 | `registros = detectar_celdas(img_th)` | 🟢 | Llama a Bloque 1. |
| 120-127 | Loop de validación e impresión por terminal | 🟢 | Patrón estándar de salida por consola. **Punto a) de la consigna**. |
| 131-135 | `csv.writer` + `writerow` | 🟢 | API estándar Python. **Punto c) de la consigna**. |
| 138-144 | Loop sobre registros OK para acumular no aprobados | 🟢 | Aplica Bloque 4 (clasificador) y filtra L/R. **Punto b) de la consigna**. |
| 144 | `generar_imagen_salida(no_aprobados, ruta_salida)` | 🟢 | Llama a Bloque 5. |
| 148-216 | `if __name__ == "__main__"` | 🟡 | Hace **exploración paso a paso con `plt.show()` sobre la planilla 1** ANTES de procesar las 4. Es material pedagógico, no producción. |

### Decisiones de diseño

- **`plt.show()` en el `__main__`**: la prestada usa el `__main__` como
  **tutorial** (muestra la original, la binarizada, las proyecciones por
  fila y columna con líneas rojas de umbral, las celdas recortadas). Es
  muy útil para entender el pipeline paso a paso, pero **no es código
  de producción**.
- **Orden de operaciones**: detecta rejilla primero (Bloque 1), después
  recorre registros y dentro aplica Bloques 2-4. Estándar en cascada.

### Qué estudiarse para defender en oral

- **Pipeline en cascada**: imagen → binarizar → rejilla → celdas →
  conteo → reglas → clasificador → reporte. Saberlo de memoria.
- **Por qué `IMREAD_GRAYSCALE`**: "trabajamos en escala de grises en
  todo el pipeline; ahorra conversión y memoria".

---

## Comparación bloque a bloque: prestada vs propia

| Bloque | Mi script propio | Prestada | Gana |
|---|---|---|---|
| Binarización | Otsu (`cv2.threshold(..., THRESH_OTSU)`) | `< 150` fijo | Mía: Otsu es más robusto si cambia la iluminación |
| Detección de filas | Canny + HoughLinesP con filtro `abs(dy) < abs(dx)*0.18` | `np.sum(img_th, 1) > 0.5 * ancho` | Empate; la mía es más general (cualquier orientación), la prestada es más simple |
| Detección de columnas | Otsu + `GaussianBlur(1, 25)` vertical + `np.sum` | `np.sum(img_th, 0) > 0.4 * alto` | Prestada: más simple, sin blur vertical sospechoso. Mía: con el blur vertical podría comerse columnas pegadas |
| Margen interno en celda | `margin_y = margin_x = 4` (4 px fijos) | `+1 px después de la línea` (variable) | Prestada: el `+1` es automático, no come caracteres. Mía: el `4` fijo puede comerse el primer/último caracter del Legajo cuando la celda es angosta |
| Área mínima | `s[4] >= 2` | `stats[:, CC_STAT_AREA] > 2` (estricto, no incluye 2) | Empate (casi igual, 1 píxel de diferencia) |
| Umbral inter-palabra | `calibrar_space_threshold` (mediana de gaps reales) | `UMBRAL_ESPACIO = 6` (constante calibrada a ojo) | Mía: cuantitativa. Prestada: suficiente para esta planilla |
| Conteo de huecos (L) | `cv2.findContours(..., RETR_TREE)` + cuenta hijos | `cv2.findContours(..., RETR_CCOMP)` + cuenta con padre | Empate conceptual; CCOMP es más barato |
| Distinción R vs A | `cv2.moments` (centro de masa vertical) | Truco de columna izquierda (20% izq, >90%) | Empate; momentos es oficial U3 y estadístico, la prestada es geométrica |
| Recorte del clasificador | `cell_gray[margin_y:-margin_y, margin_x:-margin_x]` | `letra[y:y+h, x:x+w]` (al bbox de la primera componente) | Prestada: más ajustado, evita líneas residuales |
| Nombres de salida | `validacion_grade_sheet_<id>.csv` y `no_aprobados_grade_sheet_<id>.png` | `resultados_<id>.csv` y `no_aprobados_<id>.png` | Diferencia operacional, no conceptual |
| Salida por terminal | Sí, con formato ">" | Sí, con formato ">" | Empate |
| Salida por imagen | Color (BGR) con tarjeta + insignia color | Escala de grises con tarjeta + etiqueta | Mía: más presentación. Prestada: más sobria |
| Tamaño del archivo | 32 KB (~760 líneas) | 9.7 KB (~216 líneas) | Prestada: más simple, menos superficie para defender |
| Material oficial usado | U1 + U2 + U3 (sin morfología) | U1 + U2 + U3 (sin morfología, sin Hough, sin Otsu, sin momentos) | Empate; ambas respetan "lo que se vio en clase" |

### Resumen ejecutivo

- **Mi script** es más **ingenieril**: tiene Otsu, Hough, momentos,
  calibración empírica. Más superficie a defender, pero más robusto a
  variaciones.
- **La prestada** es más **didáctica**: pura matemática de arrays +
  componentes conectadas + contornos simples. Más fácil de defender si
  el profe pregunta "¿y si no te dejan usar Hough o Otsu?".
- **Ambas andan** en las 4 planillas reales. La diferencia es solo de
  estilo y robustez.

---

## Resumen del estado de los 6 bloques de la prestada

| Bloque | Estado | Razón |
|---|:---:|---|
| 1 — Detección de rejilla | 🟢 | `np.sum` + umbrales relativos. Funcional y oficial. |
| 2 — Conteo caracteres/palabras | 🟢 | Componentes conectadas (oficial U3) + gaps horizontales. |
| 3 — Reglas de validación | 🟢 | Lógica de la consigna pura, sin decisiones de diseño. |
| 4 — Clasificador L/R/A | 🟢🟡 | `findContours RETR_CCOMP` oficial + truco de columna izquierda (🟡). |
| 5 — Imagen de salida | 🟢 | Renderizado con `np.vstack` + `cv2.putText`. |
| 6 — Orquestación | 🟢🟡 | API estándar + `plt.show()` exploratorio en main (🟡, no producción). |

**No hay puntos 🔴** que el profe marcaria como "fuera del material".

---

## Qué preguntarían en oral y respuestas sugeridas

**P: ¿Por qué la prestada no usa Otsu si está en el material oficial?**
R: Porque trabaja directo sobre `img < 150`, más simple y suficiente
para esta planilla. Otsu sería más robusto si la iluminación variara
entre planillas. En mi propia versión sí uso Otsu; acá el prestamista
prefirió simplicidad.

**P: ¿Por qué la prestada no usa Hough si está en U3?**
R: Porque la prestada aprovecha que las líneas de esta planilla son
todas horizontales o verticales (no oblicuas). `np.sum` con umbral al
50% del otro eje detecta líneas rectas perpendiculares sin más.
Hough es necesario cuando las líneas pueden tener cualquier orientación.
En mi propia versión sí uso Hough; acá no hace falta.

**P: ¿Por qué `RETR_CCOMP` y no `RETR_TREE`?**
R: `RETR_CCOMP` da jerarquía de 2 niveles (externo + 1 nivel de hijos).
Alcanza para contar "agujeros" porque un hueco es hijo directo del
contorno exterior de la letra. `RETR_TREE` da la jerarquía completa,
útil si quisiéramos procesar huecos anidados (raro en letras).

**P: ¿Por qué el truco de la "columna izquierda" en vez de `cv2.moments`?**
R: La prestada busca una propiedad geométrica simple: la R tiene un
trazo vertical izquierdo continuo, la A no. `cv2.moments` mide
distribución de masa (centro de masa vertical); también funciona pero
es más conceptual. Las dos propiedades distinguen R de A en >90% de
los casos.

**P: ¿Los nombres de archivos de salida son distintos (prestada vs propia). ¿Por qué?**
R: Es decisión de cada equipo. El formato del CSV (ID, Legajo,
OK/MAL, etc.) lo define la consigna y es igual en ambos scripts. El
nombre del archivo es solo operativo.

**P: ¿Por qué la prestada tiene `plt.show()` en el `__main__`?**
R: Es material **pedagógico/exploratorio**: muestra la imagen original,
la binarizada, las proyecciones por fila/columna con líneas de umbral,
y las celdas recortadas del primer registro. No es código de
producción, es un tutorial paso a paso del pipeline.

**P: ¿Y si te pidieran cambiar el umbral `UMBRAL_BINARIO = 150`?**
R: Es trivial: el umbral se define como constante arriba y se usa una
sola vez (línea 115). Cambiar a 100, 120 o 200 es tocar una sola
línea. Si la iluminación variara entre planillas, conviene pasar a
Otsu (mi script propio lo hace).

**P: ¿Por qué `UMBRAL_ESPACIO = 6` y no otra cosa?**
R: Está calibrado midiendo los huecos de "JUAN CARLINI" (comentario
en el archivo). En esa medición: huecos intra-letra de 1-3 px, huecos
inter-palabra de 12 px. Un umbral de 6 separa los dos grupos con
margen. Mi script propio hace esto **automáticamente** con
`calibrar_space_threshold` (mediana entre los dos clusters), pero la
prestada lo hace a ojo.

**P: ¿Por qué `UMBRAL_AREA = 2`?**
R: Cubre caracteres muy delgados (la barra `/` o el guion `-` pueden
tener 1-3 píxeles de área en esta tipografía). Un umbral de 3 los
perdería. Mi script usa el mismo umbral.

---

## Nota final: cómo presentar la prestada en la defensa

Si el profe o un compañero me pregunta "¿hiciste vos la prestada?",
mi respuesta es:

> "No, es una versión que me pasó un compañero después de la entrega.
> La estudié como referencia porque me parece un enfoque pedagógico
> interesante — más cercano a la matemática de arrays vista en clase,
> sin meterme con Hough ni con momentos. Mi TP1 entregado usa mi
> propio enfoque con Canny+Hough para la rejilla y `cv2.moments` para
> el clasificador. Los dos andan, son decisiones de diseño distintas."

Si me preguntan "¿cuál es mejor?", la respuesta honesta es:

> "Depende del criterio. La prestada es más simple y tiene menos
> superficie de defensa (menos técnicas a justificar). La mía es más
> robusta a variaciones de iluminación (usa Otsu) y más general (usa
> Hough para líneas en cualquier orientación). Para esta planilla en
> particular las dos andan bien."
