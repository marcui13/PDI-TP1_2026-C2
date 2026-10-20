# Guía narrativa para la defensa oral del Problema 2
## Validación Automática de Planillas de Calificaciones mediante Procesamiento Digital de Imágenes Clásico
**Cátedra:** Procesamiento de Imágenes I (IA 4.4) — Tecnicatura Universitaria en Inteligencia Artificial (TUIA)  
**Facultad de Ciencias Exactas, Ingeniería y Agrimensura (FCEIA) — Universidad Nacional de Rosario (UNR)**  
**Año lectivo:** 2026 — 2° Semestre  

---

## 2. Introducción general: propósito del procesamiento y recorrido global de los datos

El propósito central del **Problema 2** es diseñar, implementar y validar un sistema de auditoría óptica completamente autónomo y determinístico para procesar planillas de calificaciones escaneadas. El sistema debe examinar registros académicos manuscritos o tipografiados, verificar campo por campo el cumplimiento estricto de restricciones sintácticas, clasificar la condición final de los alumnos y emitir reportes estructurados tanto en formato tabular como en paneles visuales de diagnóstico.

Todo este desarrollo se enmarca estrictamente en el paradigma del **Procesamiento Digital de Imágenes (PDI) clásico**, apoyado en los fundamentos teóricos y prácticos de las **Unidades 1, 2 y 3** de la materia:
* **Unidad 1:** Geometría discreta, proyecciones unidimensionales (perfiles integrales de proyección / transformada discreta de Radon) y muestreo espacial.
* **Unidad 2:** Transformaciones de intensidad, relaciones espaciales entre píxeles y conectividad.
* **Unidad 3:** Binarización global adaptativa por el método de Otsu, análisis de componentes conexas en conectividad 8, extracción de características morfológicas y topología discreta basada en la característica de Euler.

A diferencia de los enfoques comerciales contemporáneos, esta solución **prescinde intencionalmente de motores de Reconocimiento Óptico de Caracteres (OCR) externos (como Tesseract)** y de modelos de aprendizaje automático supervisado o redes neuronales convolucionales. La justificación de esta decisión de diseño es doble: por un lado, pedagógica, demostrando cómo las propiedades geométricas, estadísticas y topológicas del dato visual permiten resolver problemas complejos de inspección documental con predictibilidad absoluta y mínimo costo computacional; por otro lado, de robustez, eliminando cajas negras y garantizando que cada decisión de aceptación o rechazo responda a una regla matemática verificable.

```
+---------------------------------------------------------------------------------------------------+
|                               RECORRIDO GLOBAL DEL FLUJO DE DATOS                                |
+---------------------------------------------------------------------------------------------------+
|  [ grade_sheet_X.png ]  (Imagen BGR de alta resolución con 20 filas x 7 columnas)                |
|           |                                                                                       |
|           v                                                                                       |
|  [ Escala de Grises ]   (cv2.COLOR_BGR2GRAY: reducción a matriz de luminancia uint8)              |
|           |                                                                                       |
|           v                                                                                       |
|  [ Binarización Otsu ]  (Cálculo de T dinámico maximizando varianza inter-clase sigma_b^2;        |
|           |              polaridad: tinta=1 [True], papel=0 [False])                              |
|           v                                                                                       |
|  [ Proyecciones 1D ]    (Sumas np.sum en axis 1 y axis 0; umbrales 50% ancho y 40% alto;           |
|           |              agrupamiento de píxeles en detectar_lineas)                              |
|           v                                                                                       |
|  [ Recorte de Celdas ]  (Extracción con margen de seguridad +1 px para evitar bordes de grilla)  |
|           |                                                                                       |
|           v                                                                                       |
|  [ Componentes Conexas] (cv2.connectedComponentsWithStats con conectividad 8; filtrado de ruido   |
|           |              con UMBRAL_AREA = 2 px; ordenamiento horizontal CC_STAT_LEFT)           |
|           v                                                                                       |
|  [ Análisis de Gaps ]   (Medición de huecos inter-caracteres; umbral UMBRAL_ESPACIO = 6 px        |
|           |              para conteo de palabras sin leer tipografía)                             |
|           v                                                                                       |
|  [ Auditoría Sintáctica](Validación booleana de reglas: Legajo [8c, 1p], Nombre [<=12c, >=2p],    |
|           |              Parciales [1-2c, 1p], Condición [1c] -> OK / MAL)                        |
|           v                                                                                       |
|  [ Topología de Euler ] (Para registros 100% OK: conteo de cavidades con RETR_CCOMP               |
|           |              0 huecos='L'; 1 hueco='R' o 'A'; desempate por trazo lateral izquierdo)  |
|           v                                                                                       |
|  [ Exportación ]        (CSV con columnas ID y campos; PNG estilizado con crops de no aprobados;  |
|                         figuras diagnósticas paso a paso)                                         |
+---------------------------------------------------------------------------------------------------+
```

A lo largo de las siguientes secciones, reconstruimos minuciosamente cada una de las ocho etapas que componen el algoritmo implementado en `problema2_validacion_planillas.py` y `TP1b.py`, fundamentando las decisiones matemáticas y empíricas que garantizan su funcionamiento.

---

## 3. Desarrollo narrativo de las ocho etapas

### Etapa 1. Imagen original

En primer lugar, partimos del archivo de imagen correspondiente a la planilla de calificaciones. El sistema recibe como entrada imágenes digitales en formato rasterizado comprimido sin pérdidas PNG (`grade_sheet_1.png`, `grade_sheet_2.png`, `grade_sheet_3.png`, `grade_sheet_4.png`, acompañadas por el modelo de plantilla vacía `grade_sheet_empty.png`).

La lectura se realiza a través de la librería OpenCV utilizando la función `cv2.imread(sheet_path)` en su modo por defecto, lo que produce una matriz tridimensional en el espacio de color BGR de tipo `numpy.uint8`. Antes de iniciar cualquier cálculo, el código implementa un control explícito de integridad:

```python
img_bgr = cv2.imread(sheet_path)
if img_bgr is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {sheet_path}")
```

Esta validación previene fallas silenciosas en la ejecución por rutas inválidas o corrupción de cabeceras, interrumpiendo el flujo con una excepción tipificada antes de comprometer memoria en operaciones posteriores.

Al analizar las características visuales y estructurales de estas imágenes, constatamos que representan un formulario académico estructurado en dos grandes zonas:
1. **Zona de encabezado superior (aprox. primeras 200 a 250 líneas de píxeles):** Contiene elementos decorativos e institucionales, tales como el logotipo institucional, una ilustración temática de robots y el título central de la materia ("Procesamiento de Imágenes 1").
2. **Zona de cuadrícula tabular:** Una tabla matricial compuesta por líneas continuas oscuras que delimita un encabezado de columnas ("Nro.", "Legajo", "Nombre y Apellido", "Notas", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final") y exactamente **20 filas de registros de alumnos**.

Desde la perspectiva del procesamiento digital, las imágenes de entrada plantean severos desafíos de heterogeneidad que descartan cualquier solución ingenua basada en máscaras de coordenadas rígidas:
* **Disparidad de resolución espacial y escalado:** Mientras que las planillas 1, 2 y 3 poseen dimensiones de aproximadamente $867 \times 1107$ píxeles, la planilla `grade_sheet_4.png` presenta una resolución diferente de $1082 \times 917$ píxeles, con una escala métrica de los caracteres y las celdas sustancialmente distinta.
* **Variaciones de iluminación y contraste:** Cada escaneo presenta niveles de fondo de papel desparejos (algunas hojas exhiben tonalidades grisáceas más oscuras o desbalance tonal entre los márgenes y el centro).
* **Trazos finos y vulnerabilidad al ruido:** El texto contiene caracteres delgados donde ciertos elementos esenciales —como el guion del legajo (`-`) o los puntos de las letras— tienen grosores de apenas 1 o 2 píxeles, coexistiendo con motas de polvo del escáner y pequeñas imperfecciones del papel.

Por estas razones, la imagen original actúa como el sustrato geométrico de referencia: de ella se extrae la información estructural y sobre ella se realizarán posteriormente los recortes finales en color verdadero (`crop`) de los nombres de los alumnos no aprobados, pero su análisis cuantitativo requiere su desacople inmediato a un espacio representacional monocromático y binario.

---

### Etapa 2. Escala de grises y binarización de Otsu

Una vez leída la imagen en color, el siguiente paso consiste en desacoplar la crominancia de la luminancia. Para ello, transformamos la matriz tridimensional BGR a una matriz bidimensional monocromática en escala de grises mediante:

```python
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
```

En Visión por Computadora, esta reducción dimensional de 3 canales a 1 es fundamental: la información cromática de la tinta negra y el papel blanco es redundante, mientras que trabajar con una sola matriz de $N \times M$ valores enteros de 8 bits ($[0, 255]$) triplica la eficiencia del procesamiento y permite modelar directamente la función de intensidad luminosa $f(x, y)$.

El objetivo medular de esta etapa es transformar la imagen continua en niveles de gris a una imagen binaria booleana donde se segregue sin ambigüedades el **objeto de interés** (la tinta oscura correspondiente a las líneas de la grilla y los caracteres alfanuméricos) del **fondo** (el papel claro de la hoja).

En versiones preliminares del desarrollo (como se observa en `TP1b.py`), se experimentó fijando un umbral escalar estático:
$$\text{img\_th} = \text{img\_gray} < 150$$

Sin embargo, someter un lote heterogéneo de documentos a un umbral constante acarrea fallas de generalización: si una hoja fue digitalizada con baja exposición luminosa, un umbral fijo puede fusionar letras vecinas con el fondo; por el contrario, si la hoja es sobreexpuesta (como ocurre en la Planilla 4), el umbral fijo puede adelgazar los trazos finos hasta quebrar caracteres continuos en fragmentos disconexos.

Para dotar al sistema de invariancia frente a las condiciones de captura, implementamos en `problema2_validacion_planillas.py` la binarización adaptativa mediante el **método de Otsu** (Unidad 3), encapsulado en la función `calcular_umbral_binario`:

```python
def calcular_umbral_binario(img_gray):
    umbral, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(umbral)
```

#### Fundamento probabilístico de Otsu
El algoritmo de Otsu es un método no paramétrico y no supervisado de selección automática de umbral basado en el histograma unidimensional normalizado de la imagen, visto como una función de densidad de probabilidad discreta $p_i = n_i / N$, para $i \in [0, L-1]$. Suponiendo que la imagen contiene dos clases de píxeles bimodales —Clase $C_0$ (tinta oscura) y Clase $C_1$ (papel claro)— separadas por un umbral variable $T$, Otsu busca aquel valor óptimo $T^*$ que **maximiza la varianza inter-clase** $\sigma_b^2(T)$, lo cual es matemáticamente equivalente a minimizar la varianza intra-clase $\sigma_w^2(T)$:

$$\sigma_b^2(T) = \omega_0(T) \cdot \omega_1(T) \cdot [\mu_0(T) - \mu_1(T)]^2$$

Donde:
* $\omega_0(T) = \sum_{i=0}^{T} p_i$ y $\omega_1(T) = \sum_{i=T+1}^{L-1} p_i$ representan las probabilidades a priori de ocurrencia de la tinta y el papel.
* $\mu_0(T) = \frac{\sum_{i=0}^{T} i \cdot p_i}{\omega_0(T)}$ y $\mu_1(T) = \frac{\sum_{i=T+1}^{L-1} i \cdot p_i}{\omega_1(T)}$ son las intensidades medias de cada clase.

Al evaluar las 4 planillas del trabajo práctico, Otsu converge dinámicamente a umbrales específicos calibrados para la reflectancia de cada hoja:
* `grade_sheet_1.png`: $T = 134.0$
* `grade_sheet_2.png`: $T = 135.0$
* `grade_sheet_3.png`: $T = 138.0$
* `grade_sheet_4.png`: $T = 143.0$

#### Polaridad de la imagen binaria
Una vez obtenido el umbral dinámico, la binarización efectiva se concreta en el código mediante la comparación relacional vectorizada:

```python
img_th = img_gray < umbral_binario
```

Esta operación aplica estrictamente la convención formal de la morfología matemática y el análisis de visión: **los píxeles con intensidad inferior al umbral (tinta oscura) toman el valor booleano `True` (o 1), mientras que los píxeles claros del fondo toman `False` (o 0)**. De este modo, la imagen resultante `img_th` representa a la tinta como masas activas sobre las cuales se podrán calcular momentos, sumatorias y componentes conexas en las etapas subsecuentes.

---

### Etapa 3. Detección estructural de la rejilla con proyecciones unidimensionales (1D)

Habiendo obtenido la máscara binaria de la planilla, el algoritmo debe resolver la localización espacial de cada una de las celdas que contienen los datos de los estudiantes. Para mantener el sistema libre de dependencias rígidas de posición absoluta, implementamos una técnica clásica de la Unidad 1: los **perfiles de proyección unidimensionales (1D)**, los cuales constituyen la aproximación discreta de la Transformada de Radon a lo largo de los ejes ortogonales cartesianos.

#### Concepto y cálculo de los perfiles de proyección
Un perfil de proyección 1D colapsa una de las dimensiones espaciales de la matriz binaria sumando la cantidad de píxeles encendidos (`True`) a lo largo de la dirección ortogonal:
1. **Proyección horizontal (eje Y):** Se calcula sumando los píxeles activos fila por fila a lo largo del eje horizontal:
   $$P_h(y) = \sum_{x=0}^{\text{ancho}-1} \text{img\_th}(y, x) \quad \longrightarrow \quad \texttt{proy\_h = np.sum(img\_th, axis=1)}$$
   Cada elemento del vector resultante $P_h(y)$ cuantifica cuántos píxeles de tinta existen en la fila $y$.
2. **Proyección vertical (eje X):** Se calcula sumando los píxeles activos columna por columna a lo largo del eje vertical:
   $$P_v(x) = \sum_{y=0}^{\text{alto}-1} \text{img\_th}(y, x) \quad \longrightarrow \quad \texttt{proy\_v = np.sum(img\_th, axis=0)}$$
   Cada elemento del vector resultante $P_v(x)$ cuantifica la acumulación vertical de tinta en la columna $x$.

```
PROYECCIÓN HORIZONTAL Ph(y)                     PROYECCIÓN VERTICAL Pv(x)
Y                                               Y ^
0 |                                             alto|
  | [Texto/Logo: picos ~100-250 px]                 |  ||  ||  ||  ||  ||  ||  ||
  |-------------------------------------            |  ||  ||  ||  ||  ||  ||  ||  <- Picos verticales
y0|====================================> 800 px     |  ||  ||  ||  ||  ||  ||  ||     (líneas de columnas)
  | [Contenido celda: picos bajos]                  |  ||  ||  ||  ||  ||  ||  ||
y1|====================================> 800 px     |------------------------------ Umbral = 40% alto
  | [Picos periódicos: líneas de filas]             |  ..  ..  ..  ..  ..  ..  ..  <- Texto de celdas
H |                                               0 +----------------------------> X
  +-------------------------------------> Masa      0                              ancho
  0                 Umbral = 50% ancho
```

#### Discriminación de líneas estructurales mediante umbrales proporcionales
En una planilla estructurada, las líneas divisorias de la tabla tienen una propiedad morfológica sobresaliente: son rectas continuas de trazo oscuro que atraviesan horizontalmente casi la totalidad del ancho de la página, y verticalmente la mayor parte de la altura de la grilla. En contraste, el texto manuscrito o impreso está compuesto por caracteres discretos separados por espacios en blanco, y el encabezado decorativo contiene zonas claras intercaladas.

Por consiguiente, en las señales de proyección $P_h(y)$ y $P_v(x)$, las líneas divisorias se manifiestan como **picos masivos de gran amplitud**, mientras que los caracteres representan ondulaciones basales de baja energía. Para discriminar automáticamente las líneas físicas sin requerir valores fijos en píxeles, establecemos dos umbrales proporcionales a las dimensiones de la imagen:
* **Umbral de líneas horizontales:** $0.5 \times \text{ancho}$ (50% de la longitud horizontal total).
* **Umbral de líneas verticales:** $0.4 \times \text{alto}$ (40% de la longitud vertical total).

#### Agrupamiento de píxeles contiguos (`detectar_lineas`)
En el mundo digital real, las líneas impresas y escaneadas rara vez miden exactamente 1 píxel de espesor; debido a la difusión de la tinta y a la función de transferencia de modulación (MTF) del escáner, las líneas físicas poseen un grosor de entre 2 y 4 píxeles. Si consideráramos cada píxel que supera el umbral como una línea independiente, una misma línea divisoria generaría múltiples cortes superpuestos.

Para resolver este problema, diseñamos la función `detectar_lineas`:

```python
def detectar_lineas(proyeccion, umbral):
    idx = np.where(proyeccion > umbral)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > 1)[0]
    inicios = np.concatenate(([idx[0]], idx[cortes + 1]))
    fines = np.concatenate((idx[cortes], [idx[-1]]))
    return list(zip(inicios, fines))
```

El algoritmo opera detectando primero el conjunto ordenado de índices discretos que superan el umbral (`idx`). Luego calcula la derivada discreta hacia adelante (`np.diff(idx)`). Cuando los píxeles pertenecen a la misma línea gruesa, la diferencia entre índices consecutivos es estrictamente igual a 1. En cambio, cuando la diferencia es mayor a 1 (`np.diff(idx) > 1`), se ha producido un salto en el espacio, marcando la frontera entre el fin de una línea y el inicio de la siguiente. De este modo, la función colapsa agrupamientos de píxeles contiguos en intervalos estructurados `(inicio, fin)`, modelando con exactitud la coordenada inicial y final del grosor de cada línea de la cuadrícula.

#### Deducción dinámica de celdas y campos (`detectar_celdas` y `determinar_campos`)
A partir de los intervalos de líneas detectados:
* La proyección horizontal aísla 22 líneas horizontales (`filas`): la línea 0 corresponde al borde superior de la tabla; la línea 1 marca el cierre del encabezado ("Nro.", "Legajo", etc.); y las restantes 20 líneas delimitan las filas de los 20 alumnos registrados.
* La proyección vertical aísla 8 líneas verticales (`cols`): la línea 0 es el margen izquierdo; la línea 1 divide la columna de índice numérico ("Nro."); y las líneas subsiguientes separan los campos de datos.

En `problema2_validacion_planillas.py`, la correspondencia de campos se calcula dinámicamente mediante `determinar_campos(cols)`, deduciendo la cantidad de parciales en función de las columnas detectadas (por defecto: `["Legajo", "Nombre y apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]`).

---

### Etapa 4. Recorte de celdas y análisis de componentes conexos

#### Construcción de las Regiones de Interés (ROIs) y manejo de fronteras
Una vez obtenidas las listas de intervalos de líneas `filas` y `cols`, el algoritmo construye la cuadrícula de celdas individuales para cada uno de los 20 registros. Un error crítico en el que incurren muchas implementaciones ingenuas es recortar la celda exactamente sobre los límites detectados de las líneas (`y0 = filas[r][0]`, `y1 = filas[r+1][0]`). Dado que las líneas tienen grosor y pueden presentar leves fluctuaciones o rebabas de digitalización, este recorte descuidado arrastra franjas oscuras en los bordes de la celda. Al binarizar, esos bordes se transformarían en falsas componentes conexas alargadas que distorsionarían por completo el conteo de letras.

Para neutralizar este fenómeno, implementamos una precaución estricta de **márgenes de seguridad perimetrales hacia adentro**:

```python
for r in range(1, len(filas) - 1):
    y0, y1 = filas[r][1] + 1, filas[r + 1][0]
    celdas_coords = []
    for c in range(1, len(cols) - 1):
        x0 = cols[c][1] + 1      # 1 px después de que finaliza la línea izquierda
        x1 = cols[c + 1][0]      # justo donde comienza la línea derecha
        celdas_coords.append((y0, y1, x0, x1))
```

Nótese que la coordenada vertical $y_0$ se posiciona exactamente en `filas[r][1] + 1` (es decir, un píxel por debajo del fin del trazo de la línea superior), y termina en `filas[r+1][0]` (donde empieza el trazo de la línea inferior). De manera idéntica, $x_0$ se ubica un píxel a la derecha de la línea divisoria izquierda (`cols[c][1] + 1`) y $x_1$ en el borde de la derecha. De esta forma, la submatriz resultante queda estrictamente circunscrita al espacio interior en blanco de la celda, garantizando que todo píxel encendido en su interior provenga exclusivamente del texto escrito por el alumno.

#### Fundamento del etiquetado de componentes conexas
Sobre cada celda binaria recortada se ejecuta el algoritmo de etiquetado de componentes conexas a través de la función nativa de OpenCV:

```python
n, labels, stats, centroids = cv2.connectedComponentsWithStats(
    celda.astype(np.uint8), 8, cv2.CV_32S
)
```

Un **componente conexo** es un conjunto maximal de píxeles $S$ donde cada par de píxeles $p, q \in S$ está conectado mediante una trayectoria de píxeles contiguos que comparten el mismo valor de intensidad (en nuestro caso, píxeles con valor 1 correspondiente a tinta).

#### Justificación técnica de la conectividad 8
En la teoría de topología digital de la Unidad 2 y 3, un píxel $p(x, y)$ posee dos vecindades fundamentales:
* **Vecindad 4 ($N_4(p)$):** Píxeles adyacentes horizontal y verticalmente: $\{(x\pm1, y), (x, y\pm1)\}$.
* **Vecindad 8 ($N_8(p)$):** Vecindad 4 más los 4 vecinos diagonales: $N_4(p) \cup \{(x\pm1, y\pm1)\}$.

El código selecciona **taxativamente conectividad 8**. La razón técnica es contundente: en tipografías digitalizadas a resoluciones medias o bajas, los trazos curvos y las diagonales de las letras (como la pata oblicua de la 'R', la diagonal de la 'Z', los vértices del número '2' o las ramas del '7') a menudo se tocan únicamente a través de sus esquinas diagonales. Si utilizáramos conectividad 4, esos trazos continuos se quebrarían en múltiples fragmentos disconexos, inflando artificialmente el conteo de componentes y provocando el rechazo erróneo del registro. La conectividad 8 preserva la continuidad del trazo tipográfico humano.

#### Extracción de propiedades geométricas y descarte de fondo
La llamada retorna la matriz de estadísticas `stats`, cuyas columnas corresponden a:
1. `cv2.CC_STAT_LEFT`: Coordenada X mínima del recuadro contenedor (*bounding box*).
2. `cv2.CC_STAT_TOP`: Coordenada Y superior del recuadro.
3. `cv2.CC_STAT_WIDTH`: Ancho horizontal del recuadro contenedor.
4. `cv2.CC_STAT_HEIGHT`: Altura vertical del recuadro.
5. `cv2.CC_STAT_AREA`: Superficie escalar en píxeles (conteo de píxeles de tinta de la componente).

El algoritmo descarta inmediatamente la fila de índice 0 (`stats = stats[1:]`), dado que en OpenCV la etiqueta 0 se reserva obligatoriamente para modelar el fondo de la imagen.

#### Diferencia conceptual entre componente conexo y carácter
Es imperativo señalar ante la mesa examinadora que **un componente conexo no equivale necesariamente a un carácter**. Existen múltiples situaciones en las que esta correspondencia biunívoca no se cumple:
* Caracteres desconexos por diseño ortográfico: Letras minúsculas como la 'i' y la 'j' constan de dos componentes conexas independientes (el cuerpo principal y el punto superior). Los signos diacríticos (tildes, diéresis 'ü', la virgulilla de la 'Ñ') agregan componentes aisladas.
* Caracteres fragmentados por baja resolución: Rupturas en trazos finos que dividen una letra en dos pedazos.
* Ligaduras o caracteres fusionados: Caracteres contiguos cuya tinta se solapa formando una única componente.

En nuestro caso, el diseño del formulario restringe los caracteres admisibles a mayúsculas sin acentos (`A-Z`), números (`0-9`), barra (`/`) y guion (`-`). Por lo tanto, cada carácter válido constituye una sola componente conexa, salvo casos patológicos que la etapa de filtrado y validación debe mitigar.

---

### Etapa 5. Filtrado de ruido y segmentación de palabras

#### Criterio de filtrado de ruido por área mínima
Luego de aislar las componentes conexas de una celda, no todas corresponden a glifos válidos. Es común que subsistan motas de polvo del escáner, ruido impulsivo de binarización o pequeños puntos residuales de 1 píxel. Para depurar el conjunto, implementamos una regla de filtrado basada en el área escalar:

```python
UMBRAL_AREA = 2  # Área mínima en píxeles
stats = stats[stats[:, cv2.CC_STAT_AREA] > UMBRAL_AREA]
```

Cualquier componente cuya área sea menor o igual a 2 píxeles es inmediatamente descartada.

#### Justificación experimental frente al caso crítico del guion (`-`)
Una pregunta recurrente en la defensa oral es: *¿Por qué no filtraron el ruido evaluando la altura del componente en lugar de su área?*

La respuesta se sustenta en el comportamiento de la Planilla 4 y el formato del campo Legajo. En un legajo típico (por ejemplo, `C-1557/1`), el guion medio (`-`) posee una morfología extremadamente particular: debido a la baja resolución de escaneo, su espesor vertical es de **apenas 1 píxel de altura**, extendiéndose a lo largo de 4 a 6 píxeles de ancho. Si hubiéramos implementado un filtro clásico por altura (por ejemplo, descartar componentes con $\text{altura} < 3\text{ px}$), el guion hubiera sido eliminado como si fuera ruido. Esto hubiera provocado un **falso rechazo sistemático**, reduciendo el conteo del legajo a 7 caracteres en vez de los 8 requeridos.

En cambio, al evaluar el área total, el guion acumula entre 4 y 6 píxeles de tinta, superando holgadamente el `UMBRAL_AREA = 2 px`. Por su parte, el ruido impulsivo y los defectos de borde rara vez superan 1 o 2 píxeles de superficie. De esta manera, el criterio de área preserva símbolos de bajo grosor vertical mientras elimina el ruido con total eficacia.

#### Lógica de ordenamiento y segmentación de palabras por análisis de *gaps*
Una vez depurada la matriz de estadísticas, las componentes conexas se encuentran en un orden arbitrario dictado por el algoritmo de recorrido de etiquetas de OpenCV. Para reconstruir el orden sintáctico de lectura humana (de izquierda a derecha), aplicamos una ordenación vectorial basada en la coordenada horizontal de inicio:

```python
stats = stats[np.argsort(stats[:, cv2.CC_STAT_LEFT])]
```

Con las componentes ordenadas espacialmente, el algoritmo aborda la **segmentación de palabras sin necesidad de utilizar OCR**. El razonamiento técnico se fundamenta en la medición de las distancias horizontales euclídeas (*gaps* o huecos) entre las cajas delimitadoras de componentes consecutivas:

```python
fin_anterior = stats[:-1, cv2.CC_STAT_LEFT] + stats[:-1, cv2.CC_STAT_WIDTH]
huecos = stats[1:, cv2.CC_STAT_LEFT] - fin_anterior
cant_palabras = int(np.sum(huecos > UMBRAL_ESPACIO)) + 1
```

```
COMPONENTE i (Letra)                COMPONENTE i+1 (Letra)
+-------------+                     +-------------+
|             |                     |             |
|             | <--- HUECO (gap) -> |             |
|             |                     |             |
+-------------+                     +-------------+
x_i           x_i + w_i             x_i+1
              ^                     ^
              +---------------------+
                 gap = x_i+1 - (x_i + w_i)
                 Si gap <= 6 px  --> Inter-letra (misma palabra)
                 Si gap > 6 px   --> Inter-palabra (espacio en blanco)
```

#### Justificación del umbral de espacio bimodal (`UMBRAL_ESPACIO = 6 px`)
Al medir experimentalmente los histogramas de distancias horizontales en los nombres de las planillas de prueba (como `JUAN CARLINI`), se detecta una **distribución claramente bimodal**:
1. **Espaciado inter-letra (dentro de una misma palabra):** Oscila de manera muy consistente entre **1 y 3 píxeles**.
2. **Espaciado inter-palabra (la separación real entre Nombre y Apellido):** Oscila entre **10 y 13 píxeles**.

Existe un **valle vacío absoluto entre 4 y 9 píxeles**. Un umbral inferior a 4 px produciría sobre-segmentación (dividiendo palabras compuestas por caracteres separados); un umbral superior a 10 px fusionaría el nombre y el apellido en una sola palabra. Fijar `UMBRAL_ESPACIO = 6 px` ubica la frontera de decisión en el baricentro exacto de la brecha bimodal, confiriendo al sistema una tolerancia de $\pm 3\text{ px}$ frente a variaciones en la caligrafía o escala de la fuente.

Puesto que cada espacio en blanco detectado divide dos términos continuos, la cantidad de palabras del campo es simplemente la cantidad de huecos mayores al umbral más uno:
$$\text{cant\_palabras} = \sum [\text{hueco} > \text{UMBRAL\_ESPACIO}] + 1$$

---

### Etapa 6. Validación de campos y reglas sintácticas

La información extraída en las etapas previas para cada celda de un registro se reduce a dos números enteros determinísticos:
* `cant_caracteres`: Número de componentes conexas válidas (área $> 2\text{ px}$).
* `cant_palabras`: Número de agrupaciones léxicas separadas por *gaps* mayores a 6 px.

La función `validar(campo, cant_caracteres, cant_palabras)` toma estos descriptores y los confronta de manera exhaustiva contra las restricciones sintácticas establecidas en el enunciado oficial de la cátedra:

```python
def validar(campo, cant_caracteres, cant_palabras):
    if campo == "Legajo":
        ok = (cant_caracteres == 8 and cant_palabras == 1)
    elif campo == "Nombre y apellido":
        ok = (cant_palabras >= 2 and cant_caracteres <= 12)
    elif campo == "Condición Final":
        ok = (cant_caracteres == 1)
    else:   # Parcial 1, 2 y 3
        ok = (cant_palabras == 1 and 1 <= cant_caracteres <= 2)
    return "OK" if ok else "MAL"
```

A continuación se detalla el análisis de cada regla sintáctica:

| Campo Evaluado | Regla Sintáctica Implementada | Fundamento Técnico y Justificación | Comportamiento frente a Cumplimiento | Comportamiento frente a Incumplimiento |
| :--- | :--- | :--- | :--- | :--- |
| **Legajo** | `cant_caracteres == 8 and cant_palabras == 1` | El formato institucional exige códigos alfanuméricos continuos de 8 símbolos (e.g. `C-1557/1`). No admite espacios intermedios. | Asigna `OK` si el conteo es exactamente 8 glifos contiguos. | Asigna `MAL` si faltan símbolos, si el guion fue absorbido por ruido, o si hay espacios. |
| **Nombre y Apellido** | `cant_palabras >= 2 and cant_caracteres <= 12` | Se exige al menos dos palabras (nombre y apellido separados por espacio) y una longitud máxima de 12 caracteres de tinta. | Asigna `OK` si registra $\ge 1$ espacio inter-palabra y un máximo de 12 glifos. | Asigna `MAL` si contiene una sola palabra (falta apellido) o supera 12 caracteres. |
| **Parciales 1, 2 y 3** | `cant_palabras == 1 and 1 <= cant_caracteres <= 2` | Las calificaciones académicas son números enteros entre 1 y 10 (1 o 2 dígitos, sin espacios). | Asigna `OK` si hay 1 dígito (notas 1 a 9) o 2 dígitos contiguos (nota 10). | Asigna `MAL` si la celda está vacía (0 glifos) o si contiene texto alfabético ("AUS"). |
| **Condición Final** | `cant_caracteres == 1` | La condición académica final se codifica con una letra mayúscula única: `'L'`, `'R'` o `'A'`. | Asigna `OK` si detecta exactamente un carácter en la celda. | Asigna `MAL` si la celda está vacía o contiene más de un glifo. |

#### Casos de estudio en las planillas de prueba
La evaluación experimental revela comportamientos de alto valor diagnóstico:
* **El caso de las celdas vacías (Planilla 3, filas 6, 18, 19, 20):** Al no haber tinta en los campos de notas, `cant_caracteres = 0`, lo que incumple la condición `1 <= cant_caracteres <= 2` y arroja legítimamente `MAL`.
* **El caso del texto no numérico (Planilla 3, fila 14):** En el Parcial 1 se consignó la palabra `"AUS"` (Ausente). El conteo arroja 3 componentes conexas en una sola palabra. Como $3 \notin [1, 2]$, el algoritmo dictamina `MAL`.
* **El caso de la nota con coma o punto decimal (Planilla 3, fila 15):** En el segundo parcial se escribió `"1.0"`. El punto decimal tiene una superficie microscópica ($< 2\text{ px}$), por lo que es filtrado como ruido puntual. Sobreviven los dígitos `'1'` y `'0'`, contabilizando 2 caracteres válidos y asignando `OK`.
* **Relación estructural de registro:** La validez de cada campo es independiente; sin embargo, para que un registro sea considerado apto para la etapa de selección visual de alumnos no aprobados, **todos sus campos deben ser simultáneamente `OK`** (`all(e == "OK" for e in estados_registro)`). Un solo fallo sintáctico descarta automáticamente al estudiante del reporte visual, garantizando la confiabilidad del padrón.

---

### Etapa 7. Clasificador topológico de condición

Cuando un registro supera satisfactoriamente todas las pruebas sintácticas (todos sus campos son `OK`), el sistema debe inspeccionar el contenido semántico de la celda **"Condición Final"** para determinar si el estudiante debe ser incluido en la imagen de alumnos no aprobados (aquellos cuya condición sea `'L'` de Libre o `'R'` de Recupera).

En lugar de recurrir a técnicas estadísticas de Machine Learning (como clasificadores k-NN o SVM entrenados con descriptores HOG/SIFT) o redes neuronales, el problema se resolvió mediante un **clasificador topológico determinístico y explicable**, implementado en `clasificar_condicion`:

```python
def clasificar_condicion(celda, umbral_area=UMBRAL_AREA):
    stats = componentes(celda, umbral_area=umbral_area)
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
```

```
               [ CARÁCTER DE CONDICIÓN FINAL ]
                             |
                             v
           ¿Cuántos agujeros internos posee?
              (Jerarquía RETR_CCOMP: j[3] != -1)
                            / \
                           /   \
               agujeros = 0     agujeros >= 1
                     |                |
                     v                v
                 LETRA 'L'     ¿'R' o 'A'? (Ambas tienen 1 agujero)
                  (LIBRE)             |
                                      v
                       ¿Presencia de trazo continuo en el
                       primer 20% del ancho izquierdo?
                                     / \
                                    /   \
                             > 0.90      <= 0.90
                                |           |
                                v           v
                            LETRA 'R'   LETRA 'A'
                           (RECUPERA)  (APROBADO)
```

#### Fundamento topológico: Número de Euler y cavidades interiores
El clasificador opera mediante un árbol de decisión en dos niveles:

**Nivel 1: Invariante topológico de cavidades cerradas (Característica de Euler)**  
Recortamos el cuadro delimitador exacto del carácter (`letra = celda[y:y+h, x:x+w]`) y extraemos sus contornos con `cv2.findContours` utilizando el modo de organización jerárquica de dos niveles `cv2.RETR_CCOMP`. En este modo, el vector `jerarquia[0][i]` almacena para cada contorno $i$ una tupla de cuatro elementos: `[Next, Previous, First_Child, Parent]`.
* Si un contorno no tiene padre (`Parent == -1`), corresponde a un contorno perimetral exterior del trazo.
* Si el contorno posee padre (`Parent != -1`), representa la frontera de un **agujero interior o cavidad cerrada** (fondo encerrado por la tinta).

La topología diferencial de los caracteres admisibles arroja un invariante axiomático:
* **Letra 'L':** Es una estructura topológicamente abierta formada por dos trazos perpendiculares sin regiones cerradas. Por ende, encierra **0 agujeros** (`agujeros == 0`).
* **Letras 'R' y 'A':** Ambas encierran exactamente **1 cavidad interna cerrada** (el lazo superior semicircular de la 'R' y el triángulo superior delimitado por el travesaño en la 'A').

Por lo tanto, la comprobación `if agujeros == 0` clasifica a la letra **'L'** con una confiabilidad teórica del 100%, de forma completamente invariante a rotaciones moderadas, grosor de trazo o cambios de escala.

**Nivel 2: Desempate morfológico entre 'R' y 'A' mediante perfil lateral de trazo**  
Para discriminar 'R' de 'A' (ambas con 1 agujero), no podemos utilizar únicamente propiedades topológicas de conectividad. El algoritmo recurre a una propiedad morfológica lateral: la presencia de un **mástil vertical recto**:
* La letra tipográfica **'R'** posee un trazo vertical rectilíneo continuo a lo largo de todo su lateral izquierdo (desde la parte superior hasta la base).
* La letra tipográfica **'A'** posee una geometría piramidal/triangular con laterales inclinados y una base inferior abierta entre sus patas.

Para cuantificar esta diferencia, aislamos una franja correspondiente al primer quinto del ancho del glifo (`franja_izq = letra[:, :max(1, w // 5)]`). Evaluamos luego la proporción de filas a lo largo de la altura que contienen al menos un píxel de tinta en dicha columna:

$$\text{columna\_izquierda} = \frac{1}{h} \sum_{y=0}^{h-1} \mathbb{I}\left( \sum_{x=0}^{\lfloor w/5 \rfloor} \text{letra}(y, x) > 0 \right)$$

En las pruebas empíricas, la 'R' exhibe una cobertura de trazo superior al **93%**, mientras que la 'A' arroja coberturas significativamente inferiores al **80%**. Fijando el umbral de corte en `0.9` (90%), el clasificador separa determinísticamente ambas letras: si la cobertura supera 0.9 se etiqueta como `'R'`; en caso contrario se clasifica como `'A'`.

---

### Etapa 8. Exportación de resultados: CSV, CCD e imágenes de los no aprobados

Al concluir el procesamiento de cada imagen de planilla, el programa genera en disco un conjunto estructurado de artefactos que permiten auditar los resultados, documentar el comportamiento del algoritmo e integrar la información en flujos académicos posteriores.

#### 1. Archivo tabular CSV (`validacion_grade_sheet_X.csv`)
En cumplimiento estricto del inciso c del enunciado, el algoritmo exporta un archivo en formato CSV estructurado para cada planilla procesada.
* **Nomenclatura y destino:** Se guardan bajo el directorio `TP1_2026-C2/resultados_problema2/` con el nombre `validacion_grade_sheet_<id>.csv` (en `TP1b.py`, se guardaban como `resultados_<id>.csv`).
* **Cabecera y columnas:** Las columnas se generan de forma dinámica según los campos identificados en la grilla:
  `ID,Legajo,Nombre y Apellido,Parcial 1,Parcial 2,Parcial 3,Condición Final`
* **Contenido de las celdas:** Cada fila corresponde a uno de los 20 registros evaluados. La primera columna contiene el identificador numérico correlativo (1 a 20) y las celdas subsiguientes registran exclusivamente los valores textuales en mayúsculas `"OK"` o `"MAL"`.

Este archivo constituye la evidencia formal y reproducible de la auditoría sintáctica, permitiendo que sistemas externos verifiquen qué campos específicos presentaron fallas de carga.

#### 2. Esclarecimiento técnico sobre la denominación "CCD"
En la consigna del trabajo práctico y en las pautas de estudio suele surgir la consulta sobre la generación de un "archivo o formato CCD".

> [!IMPORTANT]
> **Aclaración rigurosa basada en el código existente:**  
> Tras inspeccionar exhaustivamente la totalidad de los archivos fuente (`problema2_validacion_planillas.py`, `TP1b.py`, cuadernos `.ipynb` e históricos de compilación), **se confirma categóricamente que el programa no genera, no exporta ni utiliza ningún archivo con formato o extensión `.ccd`**.

Desde el punto de vista conceptual y terminológico en el contexto de la cátedra de PDI, la referencia a **CCD** puede obedecer a dos posibles acepciones que el alumno debe saber defender:
1. **Connected Component Data / Descriptors (Datos de Componentes Conexas):** Corresponde a la matriz numérica bidimensional que retorna la función `cv2.connectedComponentsWithStats` (parámetro `stats`), que almacena en memoria los 5 descriptores esenciales de cada objeto (`LEFT`, `TOP`, `WIDTH`, `HEIGHT`, `AREA`). Estas estadísticas se manipulan exclusivamente en memoria volátil (RAM) dentro de las funciones `componentes` y `contar` para tomar decisiones de filtrado y segmentación, pero no son persistidas a un archivo físico independiente con esa extensión.
2. **Sensor CCD (Charge-Coupled Device):** Concepto teórico fundamental de la **Unidad 1 y 2** relativo a los transductores de adquisición de imágenes (dispositivos de acoplamiento de carga utilizados en escáneres planos y cámaras digitales para convertir fotones incidentes en cargas eléctricas discretizadas).

Por lo tanto, si la mesa examinadora interroga sobre el archivo CCD, la respuesta correcta es explicar con total transparencia que en la implementación real del proyecto las únicas salidas físicas generadas a disco son los archivos estructurados CSV y las imágenes compuestas PNG, manteniéndose los datos descriptivos de componentes conexas estrictamente en memoria durante la ejecución del pipeline.

#### 3. Panel visual de alumnos no aprobados (`no_aprobados_grade_sheet_X.png`)
En cumplimiento del inciso b, el sistema genera una imagen de salida sintetizada que aísla visualmente a los alumnos que no lograron promocionar la materia.
* **Criterio de selección estricto:** Únicamente se incorporan alumnos que satisfacen dos condiciones concurrentes:
  1. Su registro fue auditado como **100% válido** (todos sus campos individuales dieron `"OK"`).
  2. Su Condición Final fue clasificada topológicamente como no aprobada: `'L'` (Libre) o `'R'` (Recupera).
* **Renderizado y diseño estético:** La función `renderizar_imagen_no_aprobados` compone un canvas de estilo profesional:
  * Encabezado superior oscuro (pizarra `#212B36`) con el título de la planilla y tipografía antialiasing `cv2.LINE_AA`.
  * Tarjetas individuales para cada estudiante donde se incrusta el número de registro (`#01`, `#02`, etc.) y el recorte directo a color original (`crop`) del campo "Nombre y Apellido", redimensionado con interpolación lineal preservando su relación de aspecto.
  * **Insignias cromáticas diferenciadas:**
    * **LIBRE ('L'):** Insignia rectangular en color **rojo** `(50, 50, 220)` en espacio BGR.
    * **RECUPERA ('R'):** Insignia rectangular en color **naranja/ámbar** `(30, 140, 240)` en espacio BGR.

#### El caso singular de la Planilla 3
En `grade_sheet_3.png`, se produce un caso límite de enorme valor pedagógico:
* De los 20 registros presentes, 19 contienen anomalías sintácticas (campos vacíos, notas en formato alfabético "AUS", nombres incompletos).
* El **único registro 100% válido es el Registro 15**, cuya condición académica es `'A'` (Aprobado).
* Como la consigna estipula reportar únicamente alumnos **no aprobados con datos válidos**, el panel resultante no incluye a ningún alumno. Para este escenario, la función renderiza de forma transparente un canvas informativo con la leyenda:  
  `"No se registraron alumnos en condicion L o R"`.

#### 4. Gráficos diagnósticos de auditoría paso a paso
Adicionalmente, el script principal incluye la función `generar_graficos_paso_a_paso`, la cual genera dos figuras consolidadas de alta resolución:
* `paso_a_paso_deteccion_rejilla.png`: Presenta la máscara de Otsu, los perfiles proyectivos horizontal y vertical con sus respectivas líneas de umbral y la superposición de la cuadrícula detectada sobre la imagen original.
* `paso_a_paso_deteccion_celdas.png`: Despliega las celdas recortadas, los *bounding boxes* numerados secuencialmente, el gráfico de barras comparativo de *gaps* horizontales con el umbral de 6 px y el diagrama de contornos de Euler con el perfil lateral de trazo.

---

## 4. Conclusión integradora: cómo se conectan las etapas

Al contemplar el flujo completo del Problema 2, se observa una arquitectura en cascada (*pipeline*) donde la salida de cada etapa condiciona rigurosamente el éxito de la subsiguiente:

```
[Imagen Original] 
       │ (BGR: sustrato de alta resolución y referencia espacial)
       ▼
[Escala de Grises y Otsu]
       │ (Máscara binaria que desacopla tinta de papel adaptándose al contraste)
       ▼
[Proyecciones 1D y Rejilla]
       │ (Coordenadas de líneas ortogonales independientes de resolución o traslación)
       ▼
[Recorte Celular y Margen de Seguridad]
       │ (ROIs puras aisladas de los marcos oscuros de la tabla)
       ▼
[Componentes Conexas (8-C) y Filtrado por Área]
       │ (Objetos conexos limpios, preservando el guion '-' y eliminando motas)
       ▼
[Ordenamiento Horizontal y Análisis de Gaps]
       │ (Detección de espacios inter-palabra sin necesidad de reconocer letras)
       ▼
[Auditoría Sintáctica Booleana]
       │ (Filtro de integridad: sólo los registros 100% OK acceden a la clasificación)
       ▼
[Clasificador Topológico de Euler y Mástil Lateral]
       │ (Separación determinística de 'L', 'R' y 'A')
       ▼
[Exportación Consolidada: CSV y Paneles Gráficos]
```

La elegancia metodológica de este diseño radica en que resuelve un problema de procesamiento documental complejo —que comúnmente se abordaría con pesadas arquitecturas de Deep Learning o bibliotecas de OCR de terceros— mediante **herramientas fundamentales del procesamiento clásico de imágenes**:
1. Operadores estadísticos globales (Otsu).
2. Transformadas integrales de proyección (Radon 1D).
3. Conectividad y análisis morfológico de vecindades discretas.
4. Topología algebraica básica (característica de Euler) combinada con perfiles direccionales.

Cada etapa actúa como un filtro reductor de complejidad: de una matriz de más de un millón de píxeles BGR pasamos a una máscara booleana, luego a coordenadas lineales de cuadrícula, después a métricas de componentes conexas, a continuación a tuplas sintácticas `(caracteres, palabras)` y finalmente a un estado de auditoría binario `OK`/`MAL` con clasificación de condición semántica.

---

## 5. Preguntas posibles de defensa oral y respuestas modelo

A continuación se presentan 16 preguntas técnicas altamente probables en una mesa de examen oral, distribuidas en pares por cada una de las 8 etapas, acompañadas de sus respuestas modelo fundamentadas en el código real:

### Etapa 1: Imagen original
* **P1: ¿Por qué no fue posible utilizar coordenadas fijas (en píxeles) para recortar las celdas de las cuatro planillas?**  
  * **Respuesta modelo:** Porque las planillas presentan heterogeneidad geométrica y de captura. En particular, mientras que las planillas 1 a 3 miden aproximadamente $867 \times 1107$ píxeles, la planilla 4 fue digitalizada a una resolución y escala diferentes ($1082 \times 917$ píxeles), presentando además leves traslaciones en los márgenes. Un recorte basado en coordenadas rígidas provocaría que las ventanas caigan sobre las líneas de la grilla o fuera de las celdas en la planilla 4. El algoritmo debe inferir las coordenadas relativas en tiempo de ejecución a partir del contenido de cada imagen.
* **P2: ¿Qué control previo realiza el programa sobre la imagen de entrada y qué riesgo mitiga?**  
  * **Respuesta modelo:** En la función `procesar_planilla`, inmediatamente después de invocar `cv2.imread`, se evalúa `if img_bgr is None: raise FileNotFoundError(...)`. Este control valida que el archivo exista en la ruta indicada, que el formato sea soportado y que los bytes de la cabecera no estén corruptos. Mitiga el riesgo de que el programa intente procesar matrices nulas (`NoneType`), lo cual provocaría excepciones de desbordamiento o fallas crípticas en `cv2.cvtColor`.

### Etapa 2: Escala de grises y binarización de Otsu
* **P3: ¿Cuál es el fundamento probabilístico del método de Otsu y por qué supera a un umbral estático de 128 o 150?**  
  * **Respuesta modelo:** Otsu modela el histograma normalizado de intensidades como una distribución bimodal de probabilidad discreta y busca automáticamente el umbral $T$ que maximiza la varianza inter-clase $\sigma_b^2(T) = \omega_0 \omega_1 (\mu_0 - \mu_1)^2$, lo que equivale a minimizar la varianza intra-clase entre la tinta y el papel. Un umbral estático como 150 fracasa ante variaciones en la iluminación del escaneo: si una hoja es más clara (como la planilla 4, donde Otsu fijó 143), un umbral bajo puede fragmentar trazos tenues; si la hoja es oscura, puede engrosar las letras hasta fundirlas entre sí. Otsu se autocalibra a la luminancia real de cada hoja.
* **P4: ¿Por qué la binarización se implementa como `img_gray < umbral_binario` y no con el operador mayor (`>`)?**  
  * **Respuesta modelo:** Porque en las imágenes originales el papel es blanco (intensidades altas, cercanas a 255) y la tinta del texto es oscura (intensidades bajas, cercanas a 0). En la teoría formal de visión y morfología matemática, el objeto de interés a procesar debe estar representado por valores booleanos `True` (o 1) y el fondo por `False` (o 0). Al usar el operador menor (`<`), los píxeles de tinta oscura se convierten en 1 y el fondo en 0, cumpliendo con la polaridad requerida para calcular sumatorias y componentes conexas.

### Etapa 3: Detección estructural de la rejilla con proyecciones 1D
* **P5: ¿Cómo diferencian las líneas de la tabla del texto manuscrito y de los logotipos al analizar las proyecciones 1D?**  
  * **Respuesta modelo:** Mediante la imposición de umbrales proporcionales a las dimensiones totales: $0.5 \times \text{ancho}$ para filas y $0.4 \times \text{alto}$ para columnas. Las líneas de la cuadrícula son trazos continuos que cruzan casi todo el ancho y alto del formulario, acumulando picos masivos de tinta que superan holgadamente el 50% y 40% respectivamente. En cambio, los caracteres del texto están separados por espacios y el logo superior contiene discontinuidades, generando picos de baja energía que quedan por debajo de dichos umbrales.
* **P6: ¿Qué problema causaría no agrupar los píxeles contiguos en la función `detectar_lineas`?**  
  * **Respuesta modelo:** Las líneas físicas de la cuadrícula tienen un grosor real de 2 a 4 píxeles. Si consideráramos cada píxel que supera el umbral como una línea independiente, una misma línea divisoria generaría entre 2 y 4 divisiones paralelas superpuestas separadas por 1 píxel. Esto produciría celdas espurias de ancho nulo o microscópico y descalibraría el conteo de filas y columnas. La función `detectar_lineas` calcula `np.diff(idx) > 1` para detectar los saltos reales y agrupar píxeles adyacentes en intervalos de `(inicio, fin)`, tratando a cada línea física como una entidad geométrica única.

### Etapa 4: Recorte de celdas y análisis de componentes conexos
* **P7: ¿Por qué en el recorte de celdas se suma un píxel en las coordenadas iniciales (`y0 = filas[r][1] + 1`, `x0 = cols[c][1] + 1`)?**  
  * **Respuesta modelo:** Para implementar un margen de seguridad perimetral hacia adentro. La coordenada `filas[r][1]` indica el último píxel del trazo de la línea divisoria superior. Al sumar 1 píxel, garantizamos que el recorte comience estrictamente en el espacio vacío interior de la celda. Si recortáramos sobre la línea, los residuos de tinta del borde de la grilla se conectarían formando componentes conexas perimetrales que alterarían falsamente el conteo de caracteres del alumno.
* **P8: ¿Por qué utilizaron conectividad 8 y no conectividad 4 en `cv2.connectedComponentsWithStats`?**  
  * **Respuesta modelo:** Porque en tipografías digitalizadas a resoluciones estándar, los trazos diagonales y curvos de los caracteres (como la diagonal de la 'Z', los vértices del número '2' o el trazo oblicuo de la 'R') tocan sus píxeles vecinos únicamente por sus esquinas diagonales. Si usáramos conectividad 4, esos píxeles diagonales se considerarían inconexos, fragmentando una sola letra en 2 o 3 componentes separadas e inflando artificialmente el conteo de caracteres. La conectividad 8 modela fielmente la continuidad física del trazo.

### Etapa 5: Filtrado de ruido y segmentación de palabras
* **P9: ¿Por qué fijaron el umbral de descarte de ruido en `UMBRAL_AREA = 2 px` en lugar de filtrar por la altura del componente?**  
  * **Respuesta modelo:** Porque en la planilla 4, el símbolo guion (`-`) del legajo posee apenas 1 píxel de altura geométrica debido a la resolución de escaneo, pero acumula un ancho de 4 a 6 píxeles, totalizando un área de entre 4 y 6 píxeles. Un filtro basado en altura (por ejemplo $\text{alto} \ge 3\text{ px}$) hubiera eliminado el guion, provocando un falso rechazo del legajo con 7 caracteres. Filtrar por área escalar con `UMBRAL_AREA = 2 px` descarta eficazmente motas de polvo y ruido impulsivo de 1 o 2 píxeles sin mutilar trazos horizontales finos.
* **P10: ¿Cómo justifica matemáticamente la elección de `UMBRAL_ESPACIO = 6 px` para contar palabras?**  
  * **Respuesta modelo:** La justificación es empírica y responde a una distribución claramente bimodal de las distancias horizontales (*gaps*): la distancia entre letras consecutivas de una misma palabra oscila entre 1 y 3 píxeles, mientras que el espacio en blanco real entre dos palabras mide entre 10 y 13 píxeles. Entre 4 y 9 píxeles existe un valle vacío sin observaciones. El valor de 6 píxeles se posiciona en el centro geométrico de dicho valle, garantizando un margen simétrico de $\pm 3\text{ px}$ frente a variaciones de espaciado o escala antes de cometer un error de fusión o sobre-segmentación de palabras.

### Etapa 6: Validación de campos y reglas sintácticas
* **P11: Si en un campo de notas se consigna la palabra "AUS" (Ausente), ¿cómo reacciona el algoritmo y por qué?**  
  * **Respuesta modelo:** El campo es clasificado determinísticamente como `"MAL"`. La regla sintáctica para las notas de los parciales exige `cant_palabras == 1 and 1 <= cant_caracteres <= 2` (para cubrir notas del 1 al 10). La palabra "AUS" genera 3 componentes conexas correspondientes a las letras 'A', 'U' y 'S'. Como $3 \notin [1, 2]$, la condición booleana evalúa a falso y el sistema rechaza el campo sin necesidad de interpretar semánticamente el texto.
* **P12: ¿Por qué la nota "1.0" de la Fila 15 en la Planilla 3 fue validada como "OK"?**  
  * **Respuesta modelo:** Porque el punto decimal `'.'` fue escrito con un tamaño diminuto cuya área superficial es menor o igual a 2 píxeles. Durante la etapa de filtrado morfológico de ruido, el punto fue descartado por no superar el `UMBRAL_AREA`. En consecuencia, la celda conservó únicamente las componentes correspondientes a los dígitos `'1'` y `'0'`, contabilizando exactamente 2 caracteres contiguos en 1 palabra, lo cual satisface la regla sintáctica de nota válida.

### Etapa 7: Clasificador topológico de condición
* **P13: ¿Cómo identifica el algoritmo a la letra 'L' sin utilizar plantillas de correlación ni redes neuronales?**  
  * **Respuesta modelo:** Mediante su invariante topológico de cavidades cerradas (característica de Euler). Al analizar los contornos con `cv2.findContours(cv2.RETR_CCOMP)`, el vector de jerarquía identifica como agujeros internos a aquellos contornos con `j[3] != -1`. Dado que la letra 'L' es una figura abierta sin lazos cerrados, encierra exactamente **0 agujeros**. Por el contrario, tanto la 'R' como la 'A' encierran 1 cavidad interior. La condición `agujeros == 0` clasifica a la 'L' de forma 100% determinística e invariante a escala.
* **P14: ¿Cómo desempataron las letras 'R' y 'A' si ambas presentan exactamente un agujero interior?**  
  * **Respuesta modelo:** Evaluando la presencia de trazo en el primer quinto del ancho (columna lateral izquierda). La tipografía de la 'R' se caracteriza por poseer un mástil vertical recto continuo que abarca de arriba a abajo, arrojando una presencia de tinta superior al 90% a lo largo de su altura. En contraste, la 'A' tiene una estructura triangular con lados diagonales inclinados y una base inferior abierta, arrojando una cobertura sustancialmente menor al 80%. Evaluando `columna_izquierda > 0.9` se discrimina con total precisión la 'R' de la 'A'.

### Etapa 8: Exportación de resultados: CSV, CCD e imágenes de no aprobados
* **P15: Si la mesa examinadora le pregunta por el "archivo CCD" del proyecto, ¿qué debe responder?**  
  * **Respuesta modelo:** Debo aclarar que en la implementación real del proyecto no existe ningún archivo generado con extensión `.ccd`. Las únicas salidas exportadas a disco son los archivos CSV de auditoría y los paneles de imágenes PNG de no aprobados. En la literatura de PDI, "CCD" alude al sensor físico *Charge-Coupled Device* de la etapa de captura, o a las matrices *Connected Component Data* (`stats`) de OpenCV, las cuales se manipulan exclusivamente en memoria RAM para tomar decisiones algorítmicas, pero no se persisten en disco como un formato de archivo independiente.
* **P16: ¿Por qué en la Planilla 3 la imagen de no aprobados no muestra a ningún estudiante?**  
  * **Respuesta modelo:** Porque la consigna b exige reportar alumnos no aprobados ('L' o 'R') **únicamente pertenecientes a registros cuyos datos hayan sido cargados de forma 100% válida**. En la Planilla 3, 19 de los 20 registros presentaron errores sintácticos de formato. El único registro completamente válido fue el Registro 15, pero dicho alumno obtuvo condición 'A' (Aprobado). Al no haber ningún alumno no aprobado con registro íntegro, el sistema genera de forma rigurosa y correcta un panel informativo indicando que no hubo alumnos para informar.

---

## 6. Conceptos fundamentales que se deben dominar sin dudar

Para asegurar una defensa oral sólida, el estudiante debe interiorizar y explicar fluidamente los siguientes conceptos sin necesidad de recurrir a la lectura del código:

1. **Varianza inter-clase en el método de Otsu ($\sigma_b^2$):** Medida estadística de separación entre los baricentros de intensidad del objeto y del fondo, ponderada por sus probabilidades de ocurrencia. Maximizarla equivale a hallar el umbral óptimo de binarización.
2. **Transformada discreta de Radon / Perfiles de proyección 1D:** Operador de reducción dimensional que acumula valores de intensidad sumando filas o columnas. Permite detectar líneas ortogonales continuas como picos dominantes en la señal proyectada.
3. **Derivada discreta hacia adelante (`np.diff`):** Operador en diferencias finitas que calcula la tasa de cambio local entre índices vecinos, utilizado para segmentar el inicio y fin de líneas con espesor mayor a 1 píxel.
4. **Vecindad y Conectividad 8 vs Conectividad 4:** Relación de adyacencia espacial en retículas discretas. La 8-conectividad incluye vecinos diagonales, evitando la fractura artificial de trazos tipográficos continuos.
5. **Invariantes topológicos y Característica de Euler ($\chi = C - H$):** Propiedades geométricas que se mantienen inalteradas frente a transformaciones elásticas continuas (rotación, traslación, escala). En nuestro caso, el número de agujeros o cavidades cerradas interiores ($H$).
6. **Distribución bimodal de espaciados (*gaps*):** Fenómeno cuantitativo donde las distancias horizontales inter-letra (1 a 3 px) y las distancias inter-palabra (10 a 13 px) conforman dos cúmulos claramente separados por un valle vacío.
7. **Filtrado morfológico por área versus dimensión lineal:** Principio de diseño por el cual caracteres con bajo espesor unidimensional (como el guion con altura = 1 px) son preservados por su área acumulada, mientras que el ruido puntual es descartado.
8. **Relación de aspecto en escalado de imágenes:** Razón matemática entre el ancho y el alto de una matriz visual ($W / H$). Su preservación durante el redimensionamiento evita distorsiones de aplastamiento o estiramiento de los caracteres.

---

## 7. Referencias al código y correspondencia de módulos

Para facilitar la consulta rápida de líneas de código durante el interrogatorio de la mesa, la siguiente tabla detalla la correspondencia exacta entre la teoría explicada y la implementación en `problema2_validacion_planillas.py` (y su versión de exploración `TP1b.py`):

| Etapa del Algoritmo | Archivo Principal | Función / Bloque | Variables / Constantes Clave | Operaciones de OpenCV / NumPy |
| :--- | :--- | :--- | :--- | :--- |
| **1. Imagen Original** | `problema2_validacion_planillas.py` | `procesar_planilla` (L443-447) | `sheet_path`, `img_bgr` | `cv2.imread(sheet_path)` |
| **2. Binarización de Otsu** | `problema2_validacion_planillas.py` | `calcular_umbral_binario` (L48-54) | `umbral_binario`, `img_th` | `cv2.cvtColor`, `cv2.threshold(..., cv2.THRESH_OTSU)`, `img_gray < umbral` |
| **3. Proyecciones 1D y Grilla** | `problema2_validacion_planillas.py` | `detectar_lineas` (L75-86), `detectar_celdas` (L89-115) | `filas`, `cols`, `proy_h`, `proy_v` | `np.sum(axis=1)`, `np.sum(axis=0)`, `np.where`, `np.diff` |
| **4. Recorte Celular y C.C.** | `problema2_validacion_planillas.py` | `componentes` (L118-130), `detectar_celdas` (L106-114) | `celdas_coords`, `stats`, `y0, y1, x0, x1` | `cv2.connectedComponentsWithStats(..., 8)`, slicing matricial con margen `+ 1` |
| **5. Filtrado y Gaps** | `problema2_validacion_planillas.py` | `contar` (L132-140) | `UMBRAL_AREA = 2`, `UMBRAL_ESPACIO = 6`, `huecos` | `np.argsort(stats[:, CC_STAT_LEFT])`, `np.sum(huecos > umbral) + 1` |
| **6. Validación Sintáctica** | `problema2_validacion_planillas.py` | `validar` (L142-153) | `cant_caracteres`, `cant_palabras`, `estado` | Comparaciones booleanas de formato (`OK` / `MAL`) |
| **7. Clasificador Topológico** | `problema2_validacion_planillas.py` | `clasificar_condicion` (L155-174) | `agujeros`, `columna_izquierda` | `cv2.findContours(..., cv2.RETR_CCOMP)`, jerarquía `j[3] != -1`, `mean() > 0.9` |
| **8. Exportación CSV y PNG** | `problema2_validacion_planillas.py` | `renderizar_imagen_no_aprobados` (L176-244), `procesar_planilla` (L503-517) | `alumnos_no_aprobados`, `csv_filename`, `img_out_filename` | `csv.DictWriter`, `cv2.imwrite`, `cv2.rectangle`, `cv2.putText(..., cv2.LINE_AA)` |
| **Versión Exploratoria Base**| `TP1b.py` | Script secuencial completo (L1-216) | `UMBRAL_BINARIO = 150`, `CAMPOS` | Prototipo inicial de exploración interactiva con gráficos matplotlib |
| **Cuaderno Demostrativo** | `TP1_2026_C2_Resolucion.ipynb` | Celdas 9 a 18 | `UMBRAL_AREA = 3`, demostraciones paso a paso | Visualización inline de celdas, bounding boxes y tablas de registros |
