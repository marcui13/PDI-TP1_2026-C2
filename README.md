# Trabajo Práctico N° 1: Procesamiento Digital de Imágenes
**Tecnicatura Universitaria en Inteligencia Artificial (TUIA)**  
**Facultad de Ciencias Exactas, Ingeniería y Agrimensura (FCEIA) — Universidad Nacional de Rosario (UNR)**  
**Año 2026 — 2° Semestre**

### 👥 Integrantes del Grupo
* **Agustín Marquardt**
* **Damián Turco**
* **Fabián Alvarez**

---

## 📌 Descripción del Proyecto
Este repositorio contiene la resolución integral del **Trabajo Práctico N° 1** de la asignatura **Procesamiento de Imágenes I (IA 4.4)**. Se abordan dos problemas centrales:

1. **Problema 1 — Ecualización Local de Histograma**:
   - Implementación de un operador no lineal de ventana deslizante $M \times N$ con extrapolación de bordes por replicación (`cv2.BORDER_REPLICATE`).
   - Revelado de detalles y geometrías ocultas en zonas de bajo contraste de la imagen `Imagen_con_detalles_escondidos.tif`.
   - Estudio paramétrico de la influencia del tamaño de la ventana ($7\times 7$, $15\times 15$, $31\times 31$, $65\times 65$ y $101\times 101$).

2. **Problema 2 — Validación Automática de Planillas de Calificaciones**:
   - Detección estructural de la tabla mediante aperturas morfológicas direccionales (horizontales y verticales) y proyecciones 1D (`np.sum`), adaptándose invariante a variaciones de resolución y escala (`grade_sheet_1.png` a `grade_sheet_4.png`).
   - Segmentación de caracteres y palabras mediante componentes conectadas (`cv2.connectedComponentsWithStats`) y análisis de espaciado inter-letra vs inter-palabra.
   - Clasificador topológico determinístico de la condición final ('L', 'R', 'A') basado en la característica de Euler (conteo de huecos interiores) y distribución vertical de masa.
   - Generación de reportes por terminal, exportación estructurada en archivos `.csv` y renderizado de una imagen resumen con los recortes de los alumnos no aprobados ('L' o 'R').

> [!NOTE]
> Conforme al reglamento de entrega de la cátedra: **"SOLO se aceptarán entregas de repositorios cuyo contenido esté compuesto únicamente de archivos con formato .py, .pdf y .md"**. El repositorio respeta estrictamente esta restricción.

---

## 📂 Contenido del Repositorio

| Archivo | Formato | Descripción |
| :--- | :---: | :--- |
| [`README.md`](README.md) | `.md` | Documentación técnica, versiones de dependencias e instrucciones de uso. |
| [`Informe_TP1_PDI.pdf`](Informe_TP1_PDI.pdf) | `.pdf` | Informe académico formal de 4 páginas con desarrollo teórico, capturas intermedias, análisis y conclusiones. |
| [`problema1_ecualizacion_local.py`](problema1_ecualizacion_local.py) | `.py` | Script ejecutable para la resolución y análisis del Problema 1. |
| [`problema2_validacion_planillas.py`](problema2_validacion_planillas.py) | `.py` | Script ejecutable para la validación cíclica de las planillas (Problema 2). |

---

## ⚙️ Requisitos y Versiones de Librerías

El código fue desarrollado y verificado en **Python 3.9+** (compatible con Python $\ge 3.8$).

### Versiones de librerías utilizadas:
*   `numpy == 2.0.2` (o compatible `numpy >= 1.24.0`)
*   `opencv-python == 5.0.0.93` (o compatible `opencv-python >= 4.7.0`)
*   `matplotlib == 3.9.4` (o compatible `matplotlib >= 3.6.0`)
*   `pillow == 11.3.0` (o compatible `pillow >= 9.4.0`)

### Instalación de dependencias:
```bash
pip install numpy opencv-python matplotlib pillow
```

---

## 🚀 Instrucciones de Ejecución

### 1. Problema 1: Ecualización Local de Histograma
El script busca por defecto la imagen `Imagen_con_detalles_escondidos.tif` en el directorio de trabajo (o permite especificar otra ruta):

```bash
# Ejecución estándar:
python problema1_ecualizacion_local.py

# Opcional: especificando ruta a la imagen
python problema1_ecualizacion_local.py --image Imagen_con_detalles_escondidos.tif
```

**Salidas generadas en tiempo de ejecución:**
*   Crea el directorio `resultados_problema1/` con las imágenes ecualizadas para cada tamaño de ventana ($7\times 7$, $15\times 15$, $31\times 31$, $65\times 65$, $101\times 101$).
*   Genera la figura comparativa `comparativa_ventanas.png`.
*   Imprime por consola el informe detallado de las figuras ocultas descubiertas y el análisis de compromiso de la ventana.

---

### 2. Problema 2: Validación de Planilla de Calificaciones
El script procesa de forma cíclica y automática las 4 imágenes de planillas (`grade_sheet_1.png` a `grade_sheet_4.png`):

```bash
# Ejecución estándar (procesa las 4 planillas en ciclo):
python problema2_validacion_planillas.py

# Opcional: procesando una o más planillas específicas
python problema2_validacion_planillas.py --sheets grade_sheet_1.png grade_sheet_2.png
```

**Salidas generadas en tiempo de ejecución:**
*   **Terminal**: Muestra el reporte registro por registro indicando `OK` o `MAL` para cada campo (Legajo, Nombre y Apellido, Parcial 1, Parcial 2, Parcial 3 y Condición Final), según las especificaciones de la cátedra.
*   **Archivos CSV**: En la carpeta `resultados_problema2/`, genera `validacion_grade_sheet_<id>.csv` con la estructura:
    ```csv
    ID,Legajo,Nombre y Apellido,Parcial 1,Parcial 2,Parcial 3,Condición Final
    1,OK,OK,OK,OK,OK,OK
    2,OK,OK,OK,OK,OK,OK
    3,MAL,MAL,MAL,MAL,MAL,MAL
    ...
    ```
*   **Imágenes de Alumnos No Aprobados**: En `resultados_problema2/`, genera `no_aprobados_grade_sheet_<id>.png` con los recortes de Nombre y Apellido y etiquetas visuales distintivas: `LIBRE (L)` en rojo y `RECUPERA (R)` en ámbar, descartando automáticamente registros con campos erróneos o alumnos aprobados (`A`).

---

## 📊 Síntesis de Resultados

### Problema 1: Detalles Ocultos Revelados
1. **Zona Superior Izquierda**: Cuadrado concéntrico más claro.
2. **Zona Superior Derecha**: Línea diagonal a 45°.
3. **Zona Central**: Carácter tipográfico minúscula: letra **"a"**.
4. **Zona Inferior Izquierda**: Cuatro líneas horizontales paralelas equidistantes.
5. **Zona Inferior Derecha**: Disco sólido relleno (círculo).
6. **Ventana Óptima**: $31 \times 31$ píxeles (equilibrio ideal entre amplificación del contraste local y preservación de bordes sin saturación de ruido).

### Problema 2: Validación de Planillas
| Planilla | Resolución | Registros Evaluados | Registros Válidos (100% OK) | Alumnos No Aprobados Reportados |
| :--- | :---: | :---: | :---: | :---: |
| `grade_sheet_1.png` | $867 \times 1107$ | 20 | 15 | 10 (7 Libres, 3 Recuperan) |
| `grade_sheet_2.png` | $869 \times 1029$ | 20 | 3 | 3 (2 Libres, 1 Recupera) |
| `grade_sheet_3.png` | $869 \times 1106$ | 20 | 1 | 0 (1 Aprobado) |
| `grade_sheet_4.png` | $1082 \times 917$ | 20 | 6 | 4 (2 Libres, 2 Recuperan) |

Para un desglose matemático y conceptual exhaustivo con capturas de pasos intermedios, consulte el archivo **[`Informe_TP1_PDI.pdf`](Informe_TP1_PDI.pdf)**.
