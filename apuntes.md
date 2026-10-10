# Apuntes para la defensa oral — Problema 1

## Ecualización local de histograma

Estos apuntes explican `problema1_ecualizacion_local.py` en relación con la consigna y el informe del TP. El foco está en qué hace cada parte, por qué se hace y cómo explicarla oralmente.

## 1. ¿Qué pide el problema?

La consigna tiene tres partes:

1. Implementar una función que reciba una imagen y una ventana de tamaño `M × N` y aplique ecualización local.
2. Usar esa función para revelar e identificar los detalles ocultos de la imagen.
3. Comparar resultados al cambiar el tamaño de la ventana.

La idea central es recorrer la imagen con una ventana deslizante. En cada posición se calcula el histograma de los píxeles de esa ventana y se transforma el nivel de gris del píxel que está en el centro. Se repite el proceso para todos los píxeles.

## 2. Idea de ecualización de histograma

En una imagen en escala de grises de 8 bits, cada píxel tiene un nivel entre 0 y 255. El histograma cuenta cuántos píxeles hay de cada nivel. La función de distribución acumulada (CDF, *cumulative distribution function*) suma esos conteos desde el nivel 0 hasta un nivel dado.

En este código, para el nivel `r` del píxel central, el nuevo nivel se calcula como:

```text
nuevo nivel = redondear((cantidad de píxeles de la ventana con nivel <= r
                         / cantidad total de píxeles de la ventana) × 255)
```

Es decir, `cdf[r] / (M*N)` estima qué fracción de los píxeles de la ventana tiene intensidad menor o igual que la del píxel central. Multiplicar por 255 lleva esa fracción al rango completo de grises.

**Intuición:** cada píxel se contrasta contra la distribución de intensidades de su propio entorno, en vez de compararlo contra toda la imagen. Por eso pueden aparecer detalles que globalmente quedaban opacados por el fondo dominante.

> Nota conceptual: esta implementación usa directamente la CDF acumulada, sin restar el primer valor no nulo de la CDF. Es una variante concreta de mapeo por CDF; conviene describir la fórmula del código tal cual en la defensa.

## 3. Imports y preparación del entorno (líneas 12–26)

- `os` permite construir rutas y comprobar si existen archivos o directorios.
- `sys` permite agregar rutas de paquetes a `sys.path`.
- El ciclo de las líneas 16–21 busca `site-packages` en dos ubicaciones posibles de `.venv`. Si encuentra una, la agrega a `sys.path` para que Python pueda importar las dependencias de ese entorno.
- `cv2` (OpenCV) carga y procesa imágenes; `numpy` crea arreglos y calcula histogramas acumulados; `matplotlib.pyplot` genera las figuras; `time` mide cuánto tarda cada ecualización.

La búsqueda de `.venv` es una comodidad de ejecución, no forma parte del algoritmo de procesamiento.

## 4. La función principal: `ecualizacion_local_histograma` (líneas 28–70)

### 4.1 Parámetros y precondición

```python
def ecualizacion_local_histograma(img, ksize=(31, 31), border_type=cv2.BORDER_REPLICATE):
```

- `img`: imagen de entrada, esperada como arreglo bidimensional `uint8` en escala de grises.
- `ksize`: tamaño de la ventana `(M, N)`. El valor predeterminado es `31 × 31`.
- `border_type`: política para extender la imagen en los bordes; por defecto replica el píxel de borde.

La aserción `assert len(img.shape) == 2` verifica que la imagen sea monocromática: tiene filas y columnas, no un tercer eje de canales de color. No verifica explícitamente el tipo de dato ni que `M` y `N` sean positivos.

### 4.2 Tamaño de ventana y padding (líneas 47–52)

```python
M, N = ksize
pad_y = M // 2
pad_x = N // 2
padded = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, border_type)
```

`M` es el tamaño vertical y `N` el horizontal. Se agrega un margen por arriba y por abajo de `M // 2` píxeles, y por izquierda y derecha de `N // 2` píxeles. Ese margen permite centrar una ventana también sobre los píxeles de los bordes sin salir de los límites del arreglo.

`cv2.BORDER_REPLICATE` completa el margen copiando los valores de los píxeles más cercanos del borde. Así no se introducen ceros artificiales que alterarían el histograma local en el perímetro.

Las ventanas probadas en el script son impares, por lo que tienen un centro único. Para tamaños pares, `M // 2` no representa simétricamente un centro único y la elección del píxel central necesita una convención adicional.

### 4.3 Preparar salida y recorrer la imagen (líneas 54–61)

```python
H, W = img.shape
total_pixels = float(M * N)
img_out = np.zeros((H, W), dtype=np.uint8)
```

- `H, W` son alto y ancho de la imagen original.
- `total_pixels` es la cantidad de muestras de cada ventana. Se convierte a `float` para que la división produzca un resultado real.
- `img_out` reserva una imagen de salida del mismo tamaño, inicialmente llena de ceros, y con el tipo de píxel requerido para una imagen de 8 bits.

Los dos bucles `for` recorren cada coordenada `(i, j)` de la imagen original. En cada iteración:

```python
patch = padded[i:i+M, j:j+N]
```

Se recorta de la imagen extendida el vecindario de tamaño `M × N` que corresponde a esa posición. Gracias al padding, el recorte siempre queda completo.

### 4.4 Histograma, CDF y transformación (líneas 63–68)

```python
hist, _ = np.histogram(patch, bins=256, range=[0, 256])
cdf = hist.cumsum()
val_centro = patch[pad_y, pad_x]
img_out[i, j] = np.uint8(np.round((cdf[val_centro] / total_pixels) * 255.0))
```

1. `np.histogram` cuenta los píxeles del parche en 256 intervalos: uno para cada intensidad posible de 0 a 255. El rango `[0, 256]` incluye el valor 255 en el último bin.
2. `hist.cumsum()` produce la CDF: en la posición `r` contiene el número de píxeles del parche con intensidad menor o igual que `r`.
3. `patch[pad_y, pad_x]` obtiene el valor de gris del centro de la ventana.
4. Se consulta la CDF para ese valor central, se divide por `M*N`, se multiplica por 255 y se redondea.
5. `np.uint8` convierte el resultado al formato de intensidad de la imagen de salida.

La ventana no transforma todos los píxeles del parche en esa iteración: determina el nuevo valor del píxel central. Luego el centro se desplaza un píxel y se repite el cálculo. Al terminar, se devuelve `img_out`.

### 4.5 Costo computacional

Para cada uno de los `H × W` píxeles se vuelve a construir un histograma de `M × N` valores. Por eso el procedimiento es costoso: al crecer la ventana, crece el trabajo de cada posición. El código mide experimentalmente ese tiempo para cada tamaño.

## 5. Función que organiza el análisis (líneas 73–189)

### 5.1 Localizar y cargar la imagen (líneas 73–86)

`ejecutar_analisis_problema1(custom_image_path=None)` determina el directorio del script y busca la imagen:

1. Si se pasó una ruta personalizada y existe, la usa.
2. Si no, intenta `Imagen_con_detalles_escondidos.tif` junto al script.
3. Como alternativa, busca `Imagen_con_objetos_ocultos.tiff`.
4. Si se había pasado una ruta personalizada pero no se encontró con las condiciones anteriores, vuelve a asignarla para que la lectura produzca un error claro si no puede cargarse.

`cv2.imread(..., cv2.IMREAD_GRAYSCALE)` carga directamente la imagen en escala de grises. Si OpenCV no logra leerla, `img` queda como `None` y se lanza `FileNotFoundError`.

La consigna nombra `Imagen_con_objetos_ocultos.tiff`; en la carpeta y en el README se usa `Imagen_con_detalles_escondidos.tif`. El script contempla ambos nombres como alternativas.

Después imprime las dimensiones y el mínimo/máximo de intensidad. Esos datos describen la imagen original; no cambian su contenido.

### 5.2 Comparación global y procesamiento local (líneas 94–107)

```python
eq_global = cv2.equalizeHist(img)
ventanas = [(7, 7), (15, 15), (31, 31), (65, 65), (101, 101)]
```

`cv2.equalizeHist` calcula una ecualización usando el histograma de toda la imagen. Sirve como referencia para contrastar el método local; no reemplaza la función implementada.

Luego el programa ejecuta la función local para cinco tamaños. `time.time()` registra el instante antes y después de cada ejecución y la diferencia `dt` da el tiempo transcurrido. Los resultados se guardan en un diccionario cuya clave es el tamaño de ventana, por ejemplo `resultados_locales[(31, 31)]`.

### 5.3 Guardar resultados (líneas 109–115)

Se crea `resultados_problema1/` si no existe. `cv2.imwrite` guarda la imagen original, la ecualización global y las cinco salidas locales. El nombre de cada archivo incorpora las dimensiones de la ventana.

### 5.4 Armar la figura comparativa (líneas 117–139)

Matplotlib crea una figura de seis paneles: imagen original, ecualización global y resultados locales para `7×7`, `15×15`, `31×31` y `65×65`. En este panel comparativo no se incluye `101×101`, aunque ese resultado sí se calcula y se guarda como archivo individual. `cmap="gray"` muestra los arreglos con una escala de grises y `axis("off")` quita los ejes. Al final se guarda `comparativa_ventanas.png` y se cierra la figura.

### 5.5 Recortar y mostrar los cinco detalles (líneas 141–167)

`eq_optima` selecciona el resultado de `31×31`, que el análisis del informe considera el mejor balance visual. Si esa clave no estuviera disponible, usa el primer resultado del diccionario como alternativa.

El diccionario `rois` define cinco regiones de interés (*Regions of Interest*, ROI) usando cortes de NumPy `[fila_inicial:fila_final, columna_inicial:columna_final]`. A cada límite se le suma o resta un margen `m = 2` píxeles. Se guardan individualmente y se presentan juntas en un panel de cinco ampliaciones.

Las ubicaciones y objetos identificados son:

1. Superior izquierda: cuadrado concéntrico más claro.
2. Superior derecha: segmento diagonal a 45°.
3. Centro: letra minúscula `a`.
4. Inferior izquierda: cuatro líneas horizontales paralelas.
5. Inferior derecha: círculo o disco relleno.

Los índices de esos recortes están escritos para la geometría de esta imagen; no son un detector automático de objetos y tendrían que recalibrarse si cambia la imagen o su tamaño.

### 5.6 Conclusiones que imprime el script (líneas 169–189)

El programa informa los cinco detalles y resume la observación cualitativa sobre las ventanas:

- **`7×7`**: usa muy pocos píxeles (49), así que el histograma local cambia mucho ante pequeñas variaciones; puede aumentar el ruido y producir halos.
- **`15×15`**: revela objetos, aunque todavía se observa grano y halos.
- **`31×31`**: el informe lo elige como compromiso entre realzar los objetos y no amplificar excesivamente el ruido.
- **`65×65` y `101×101`**: incorporan una región más amplia del fondo; la transformación se adapta menos a cada detalle pequeño y se acerca al comportamiento global.

La ventana es un parámetro de escala espacial: una ventana chica responde a cambios muy cercanos; una grande resume un entorno más amplio. El tamaño óptimo no es universal: es una conclusión experimental para esta imagen y estos resultados.

## 6. Punto de entrada del programa (líneas 191–196)

```python
if __name__ == "__main__":
```

Este bloque ejecuta el análisis solamente cuando el archivo se corre directamente, no cuando se importa como módulo. `argparse` acepta `--image` para indicar otra imagen. Si no se proporciona, el argumento vale `None` y se activa la búsqueda de nombres predeterminados.

Ejemplos:

```bash
python problema1_ecualizacion_local.py
python problema1_ecualizacion_local.py --image ruta/a/otra_imagen.tif
```

## 7. Cómo contarlo en la defensa oral

> “Implementamos una ecualización local con una ventana deslizante. Para cada píxel, tomamos el parche centrado en él, calculamos el histograma de 256 niveles y su acumulada. El nuevo nivel del píxel central es la fracción acumulada correspondiente a su intensidad, escalada al rango 0–255. Extendemos la imagen con replicación de bordes para que las ventanas de los píxeles perimetrales sigan teniendo tamaño completo. Después comparamos cinco tamaños de ventana y usamos la ecualización global como referencia. En esta imagen, `31×31` dio el mejor equilibrio entre revelar los cinco objetos y evitar el exceso de ruido de ventanas pequeñas o la pérdida de adaptación local de las grandes.”

## 8. Preguntas que podrían hacerte

### ¿Por qué la ecualización global no alcanza?

Porque usa una sola distribución para toda la imagen. Si una región o nivel tonal dominante concentra la mayoría de los píxeles, los detalles pequeños y de bajo contraste pueden ocupar muy poco del rango transformado. La versión local calcula una distribución distinta para cada vecindario.

### ¿Qué hace la CDF?

Para cada intensidad `r`, cuenta cuántos píxeles del parche tienen intensidad menor o igual que `r`. Al normalizar por `M*N`, se obtiene una proporción entre 0 y 1; al multiplicarla por 255 se obtiene una intensidad de salida.

### ¿Por qué se usa `BORDER_REPLICATE`?

En los bordes no existe imagen por fuera del dominio. Replicar los valores de borde completa el vecindario sin agregar intensidades artificiales, como los ceros de un padding constante.

### ¿Qué pasa si hago la ventana más chica o más grande?

La chica es más sensible a cambios locales, pero también al ruido. La grande suaviza las diferencias locales de la distribución porque incluye más contexto y puede perder contraste en detalles pequeños. Por eso se evalúa empíricamente el tamaño adecuado.

### ¿Se ecualiza todo el parche en cada paso?

No. El parche sirve para construir la transformación local, pero el código asigna el resultado solo al píxel central. El siguiente paso mueve la ventana un píxel.

### ¿Qué significa que sea una ventana `M×N`?

`M` es su cantidad de filas y `N` su cantidad de columnas. La función admite una ventana rectangular, aunque los tamaños usados en el experimento son cuadrados.

## 9. Detalles para no confundir en la exposición

- La ecualización global es una comparación hecha con OpenCV; la función propia implementa la ecualización local.
- El script procesa una imagen en escala de grises y los resultados son arreglos de 8 bits.
- El cálculo local se repite para cada píxel, por eso el tamaño de ventana impacta en el tiempo.
- La selección de `31×31` y los cinco recortes de objetos son decisiones experimentales específicas de esta imagen.
- El panel de comparación incluye hasta `65×65`; el resultado `101×101` igualmente se procesa y se guarda.
- La consigna y el archivo usan nombres distintos para la imagen de entrada; el script contempla ambos.
