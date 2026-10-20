#!/usr/bin/env python3
"""
Script para generar el PDF de resumen para la defensa del TP1 - Problema 2
Relevamiento exhaustivo de métodos, constantes, funciones y retornos de OpenCV y NumPy.
"""

import os
import sys
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas de dos pasadas para calcular dinámicamente el total de páginas."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Encabezado (en todas las páginas)
        self.drawString(
            32,
            560,
            "TUIA - FCEIA - UNR | Procesamiento de Imágenes I (2026) — Guía de Defensa: Problema 2",
        )
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(32, 554, 810, 554)

        # Pie de página
        self.line(32, 34, 810, 34)
        self.drawString(
            32,
            23,
            "Relevamiento Exhaustivo de Métodos, Retornos y Constantes | OpenCV (cv2) y NumPy (np)",
        )
        page_text = f"Página {self._pageNumber} de {page_count}"
        self.drawRightString(810, 23, page_text)
        self.restoreState()


def crear_pdf(ruta_salida):
    doc = SimpleDocTemplate(
        ruta_salida,
        pagesize=landscape(A4),
        leftMargin=32,
        rightMargin=32,
        topMargin=38,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Estilos tipográficos
    titulo_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
    )

    subtitulo_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )

    seccion_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=4,
        spaceAfter=3,
    )

    seccion_verde = ParagraphStyle(
        "SectionHeadingGreen",
        parent=seccion_style,
        textColor=colors.HexColor("#065F46"),
    )

    seccion_rojo = ParagraphStyle(
        "SectionHeadingRed",
        parent=seccion_style,
        textColor=colors.HexColor("#991B1B"),
    )

    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    cell_bold_style = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#0F172A"),
    )

    cell_style = ParagraphStyle(
        "CellText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#1E293B"),
    )

    cell_tip_style = ParagraphStyle(
        "CellTip",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.8,
        leading=8.8,
        textColor=colors.HexColor("#166534"),
    )

    cell_alert_style = ParagraphStyle(
        "CellAlert",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=6.8,
        leading=8.8,
        textColor=colors.HexColor("#991B1B"),
    )

    pregunta_style = ParagraphStyle(
        "QStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#991B1B"),
    )

    respuesta_style = ParagraphStyle(
        "AStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.3,
        leading=9.8,
        textColor=colors.HexColor("#1E293B"),
    )

    story = []

    # =========================================================================
    # PÁGINA 1: MÉTODOS Y CONSTANTES DE OPENCV (cv2)
    # =========================================================================
    story.append(Paragraph("GUÍA RÁPIDA PARA DEFENSA: RELEVAMIENTO DE OPENCV Y NUMPY", titulo_style))
    story.append(
        Paragraph(
            "<b>Trabajo Práctico N° 1 (Problema 2: Validación de Planillas)</b> — TUIA · FCEIA · UNR (2026) | "
            "Desglose técnico de funciones, parámetros, estructuras de retorno y justificaciones para la defensa oral.",
            subtitulo_style,
        )
    )
    story.append(Spacer(1, 4))
    story.append(Paragraph("1. Métodos, Constantes y Flags de OpenCV (cv2)", seccion_style))

    encabezados_cv2 = [
        Paragraph("Elemento / Constante", th_style),
        Paragraph("Parámetros y Uso en el Código", th_style),
        Paragraph("¿Qué hace técnicamente?", th_style),
        Paragraph("Tip Clave para la Defensa Oral (¿Por qué se usa así?)", th_style),
    ]

    filas_cv2 = [
        encabezados_cv2,
        [
            Paragraph("<b>cv2.imread</b>", cell_bold_style),
            Paragraph("<code>sheet_path</code>", cell_style),
            Paragraph("Carga la imagen de la planilla desde disco como array NumPy de 3 canales en memoria.", cell_style),
            Paragraph("<b>Atención al formato:</b> Carga en <b>BGR</b> (no RGB) con tipo <code>uint8</code>. Se debe convertir antes de operar o graficar.", cell_tip_style),
        ],
        [
            Paragraph("<b>cv2.cvtColor</b><br/>"
                      "• <code>COLOR_BGR2GRAY</code><br/>"
                      "• <code>COLOR_BGR2RGB</code><br/>"
                      "• <code>COLOR_GRAY2BGR</code>", cell_bold_style),
            Paragraph("<code>img, código_conversión</code>", cell_style),
            Paragraph("Convierte entre espacios de color:<br/>"
                      "• <b>BGR2GRAY:</b> canal de luminancia (grises).<br/>"
                      "• <b>BGR2RGB:</b> para mostrar en Matplotlib.<br/>"
                      "• <b>GRAY2BGR:</b> 3 canales para pintar recuadros color.", cell_style),
            Paragraph("<b>Justificación:</b> Grises reduce de 3 canales a 1 para Otsu. Matplotlib espera RGB (si se pasa BGR se ve azul). GRAY2BGR permite dibujar bounding boxes en color sobre máscaras binarias.", cell_tip_style),
        ],
        [
            Paragraph("<b>cv2.threshold</b><br/>"
                      "• <code>THRESH_OTSU</code><br/>"
                      "• <code>THRESH_BINARY_INV</code>", cell_bold_style),
            Paragraph("<code>img_gray, 0, 255,<br/>cv2.THRESH_BINARY_INV +<br/>cv2.THRESH_OTSU</code>", cell_style),
            Paragraph("Binarización global adaptativa. Minimiza la varianza intra-clase (Otsu). Devuelve <code>(umbral, img_th)</code>.", cell_style),
            Paragraph("<b>¿Por qué thresh=0?</b> Otsu calcula el umbral solo; el valor 0 se ignora.<br/>"
                      "<b>¿Por qué BINARY_INV?</b> El texto original es oscuro sobre papel claro. En PDI los objetos deben ser blancos (255) y el fondo negro (0).", cell_tip_style),
        ],
        [
            Paragraph("<b>cv2.connectedComponentsWithStats</b><br/>"
                      "• <code>cv2.CV_32S</code><br/>"
                      "• <code>8 (conectividad)</code>", cell_bold_style),
            Paragraph("<code>celda.astype(np.uint8), 8, cv2.CV_32S</code><br/>Retorna: <code>n, labels, stats, centroids</code>", cell_style),
            Paragraph("Segmenta y calcula estadísticas morfológicas de cada componente de la celda en un único pase.", cell_style),
            Paragraph("<b>¿Por qué 8-conectividad?</b> Evita que trazos diagonales de letras se quiebren en piezas separadas.<br/>"
                      "<b>¿Por qué stats[1:]?</b> El índice 0 siempre representa el <b>fondo</b>, no una letra.", cell_tip_style),
        ],
        [
            Paragraph("Constantes de <code>stats</code>:<br/>"
                      "• <code>cv2.CC_STAT_LEFT</code><br/>"
                      "• <code>cv2.CC_STAT_WIDTH</code><br/>"
                      "• <code>cv2.CC_STAT_AREA</code>", cell_bold_style),
            Paragraph("Columnas de indexación de la matriz <code>stats</code> de componentes conexas.", cell_style),
            Paragraph("• <b>LEFT / WIDTH:</b> posición X y ancho del bounding box.<br/>"
                      "• <b>AREA:</b> cantidad de píxeles encendidos del objeto.", cell_style),
            Paragraph("<b>Filtrado y palabras:</b><br/>"
                      "• <code>AREA > 2 px</code>: elimina ruido puntual y restos de grilla.<br/>"
                      "• Distancia horizontal (gap): <code>LEFT[i+1] - (LEFT[i] + WIDTH[i])</code>. Si gap > 6 px, es espacio entre palabras.", cell_tip_style),
        ],
        [
            Paragraph("<b>cv2.findContours</b><br/>"
                      "• <code>cv2.RETR_CCOMP</code><br/>"
                      "• <code>cv2.CHAIN_APPROX_SIMPLE</code>", cell_bold_style),
            Paragraph("<code>letra, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE</code><br/>Retorna: <code>contornos, jerarquía</code>", cell_style),
            Paragraph("Extrae contornos vectoriales y construye una jerarquía topológica de dos niveles (externo vs agujeros internos).", cell_style),
            Paragraph("<b>Euler y clasificación ('L' vs 'R'/'A'):</b><br/>"
                      "• Si <code>jerarquia[0][i][3] != -1</code>, tiene padre (es un hueco interno).<br/>"
                      "• 'L' = 0 agujeros.<br/>"
                      "• 'R' y 'A' = 1 agujero (se desempatan por trazo vertical).", cell_tip_style),
        ],
        [
            Paragraph("<b>cv2.drawContours</b>", cell_bold_style),
            Paragraph("<code>vis, contornos, idx, color, grosor</code>", cell_style),
            Paragraph("Dibuja contornos poligonales en una imagen (usado en el gráfico de diagnóstico de Euler).", cell_style),
            Paragraph("Permite visualizar en colores distintos el perímetro externo (amarillo) de los agujeros cerrados (rojo).", cell_tip_style),
        ],
        [
            Paragraph("<b>cv2.resize</b>", cell_bold_style),
            Paragraph("<code>crop, (target_w, target_h)</code>", cell_style),
            Paragraph("Redimensiona el recorte de Nombre y Apellido para encajarlo de forma estilizada en la tarjeta de no aprobados.", cell_style),
            Paragraph("<b>Ojo al orden:</b> En OpenCV el tamaño es <code>(ancho, alto)</code>; en NumPy la forma es <code>(alto, ancho)</code>.", cell_tip_style),
        ],
        [
            Paragraph("<b>cv2.rectangle</b>", cell_bold_style),
            Paragraph("<code>canvas, pt1, pt2, color, thickness</code>", cell_style),
            Paragraph("Dibuja tarjetas, fondos de encabezados, insignias (badges) y recuadros de componentes.", cell_style),
            Paragraph("<b>Espesor -1:</b> Rellena la figura con color sólido (muy usado para fondos de tarjetas y botones de estado).", cell_tip_style),
        ],
        [
            Paragraph("<b>cv2.line</b>", cell_bold_style),
            Paragraph("<code>grid_vis, pt1, pt2, color, thickness</code>", cell_style),
            Paragraph("Traza líneas rectas entre dos coordenadas para superponer la rejilla detectada en el diagnóstico.", cell_style),
            Paragraph("Verifica visualmente si las proyecciones 1D coincidieron exactamente con las líneas impresas de la grilla.", cell_tip_style),
        ],
        [
            Paragraph("<b>cv2.putText</b><br/>"
                      "• <code>FONT_HERSHEY_SIMPLEX</code><br/>"
                      "• <code>LINE_AA</code>", cell_bold_style),
            Paragraph("<code>canvas, text, org, font, scale, color, thick, LINE_AA</code>", cell_style),
            Paragraph("Renderiza texto vectorial sobre las imágenes de diagnóstico y el panel de salida de no aprobados.", cell_style),
            Paragraph("<b>¿Por qué LINE_AA?</b> Aplica suavizado anti-aliasing (elimina el pixelado serruchado en los bordes de las letras).", cell_tip_style),
        ],
    ]

    col_widths_cv2 = [135, 175, 215, 253]
    tabla_cv2 = Table(filas_cv2, colWidths=col_widths_cv2, repeatRows=1)
    tabla_cv2.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#1E3A8A")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(tabla_cv2)

    story.append(PageBreak())

    # =========================================================================
    # PÁGINA 2: ANATOMÍA DETALLADA DE RETORNOS (OPENCV: COMPONENTES Y CONTORNOS)
    # =========================================================================
    story.append(Paragraph("2. Desglose en Detalle de Retornos Críticos (OpenCV)", seccion_style))
    story.append(
        Paragraph(
            "Estructura exacta de datos, dimensiones, significado de cada índice y cómo responder en la defensa oral.",
            subtitulo_style,
        )
    )
    story.append(Spacer(1, 4))

    # Subtabla A: connectedComponentsWithStats
    story.append(Paragraph("A. Retorno de <code>cv2.connectedComponentsWithStats</code>: <code>( n, labels, stats, centroids )</code>", seccion_style))

    encabezados_ret_cc = [
        Paragraph("Variable Retornada", th_style),
        Paragraph("Tipo y Forma (Shape)", th_style),
        Paragraph("Significado y Contenido de los Datos", th_style),
        Paragraph("Uso en el Código y Pregunta de Examen", th_style),
    ]

    filas_ret_cc = [
        encabezados_ret_cc,
        [
            Paragraph("<b>n</b><br/>(num_labels)", cell_bold_style),
            Paragraph("<code>int</code> (escalar)", cell_style),
            Paragraph("Cantidad total de componentes conexas halladas <b>+ 1</b> (porque incluye al fondo como etiqueta 0).", cell_style),
            Paragraph("Si en una celda dice 'SOY' (3 letras aisladas), <code>n = 4</code> (etiqueta 0 fondo, 1, 2 y 3 letras).", cell_tip_style),
        ],
        [
            Paragraph("<b>labels</b>", cell_bold_style),
            Paragraph("<code>ndarray 2D</code><br/><code>(alto, ancho)</code><br/>tipo: <code>int32</code>", cell_style),
            Paragraph("<b>Mapa de segmentación píxel a píxel.</b> Cada coordenada $(y, x)$ contiene el ID de la componente a la que pertenece ($0$ para fondo, $1$ a $n-1$ para letras).", cell_style),
            Paragraph("En el código se descarta con <code>_</code> (guion bajo) porque no coloreamos píxel por píxel; nos alcanzan las estadísticas globales de <code>stats</code>.", cell_tip_style),
        ],
        [
            Paragraph("<b>stats</b><br/>(La más importante del TP)", cell_bold_style),
            Paragraph("<code>ndarray 2D</code><br/><code>(n, 5)</code><br/>tipo: <code>int32</code>", cell_style),
            Paragraph("<b>Tabla geométrica de 5 columnas</b> por componente ($i = 0$ a $n-1$):<br/>"
                      "• <code>[0] CC_STAT_LEFT</code>: $X_{\\min}$ del bounding box.<br/>"
                      "• <code>[1] CC_STAT_TOP</code>: $Y_{\\min}$ del bounding box.<br/>"
                      "• <code>[2] CC_STAT_WIDTH</code>: Ancho en píxeles del box.<br/>"
                      "• <code>[3] CC_STAT_HEIGHT</code>: Alto en píxeles del box.<br/>"
                      "• <code>[4] CC_STAT_AREA</code>: Píxeles reales encendidos.", cell_style),
            Paragraph("<b>¿Por qué hacemos stats[1:]?</b><br/>"
                      "La fila <code>stats[0]</code> es el <b>fondo</b> (área enorme que cubre toda la celda). Se descarta con <code>[1:]</code> para quedarse exclusivamente con las letras.<br/>"
                      "<b>Área > 2 px:</b> elimina ruido.<br/>"
                      "<b>Huecos:</b> <code>LEFT[i+1] - (LEFT[i] + WIDTH[i]) > 6 px</code> marca espacio.", cell_alert_style),
        ],
        [
            Paragraph("<b>centroids</b>", cell_bold_style),
            Paragraph("<code>ndarray 2D</code><br/><code>(n, 2)</code><br/>tipo: <code>float64</code>", cell_style),
            Paragraph("Centro de masa o baricentro geométrico $(c_x, c_y)$ de cada componente, calculado promediando las coordenadas de sus píxeles.", cell_style),
            Paragraph("Se descarta con <code>_</code> en el código porque para ordenar de izquierda a derecha usamos el borde izquierdo (<code>CC_STAT_LEFT</code>).", cell_tip_style),
        ],
    ]

    col_widths_ret_cc = [135, 140, 245, 258]
    tabla_ret_cc = Table(filas_ret_cc, colWidths=col_widths_ret_cc, repeatRows=1)
    tabla_ret_cc.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#1E3A8A")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(tabla_ret_cc)
    story.append(Spacer(1, 6))

    # Subtabla B: findContours
    story.append(Paragraph("B. Retorno de <code>cv2.findContours</code>: <code>( contornos, jerarquia )</code>", seccion_style))

    encabezados_ret_fc = [
        Paragraph("Variable Retornada", th_style),
        Paragraph("Tipo y Forma (Shape)", th_style),
        Paragraph("Significado y Contenido de los Datos", th_style),
        Paragraph("Uso en el Código y Pregunta de Examen", th_style),
    ]

    filas_ret_fc = [
        encabezados_ret_fc,
        [
            Paragraph("<b>contornos</b>", cell_bold_style),
            Paragraph("<code>tuple[ndarray]</code><br/>Longitud: $M$<br/>Cada uno: <code>(K, 1, 2)</code> tipo: <code>int32</code>", cell_style),
            Paragraph("Lista de polígonos frontera. Cada elemento contiene las $K$ coordenadas <code>[x, y]</code> de los vértices del contorno.<br/>"
                      "Con <code>CHAIN_APPROX_SIMPLE</code> solo se almacenan puntos de inflexión (ahorra memoria comprimiendo líneas rectas).", cell_style),
            Paragraph("Se usan en <code>cv2.drawContours</code> para dibujar el contorno perimetral externo (amarillo) y las cavidades internas (rojo) en el gráfico diagnóstico.", cell_tip_style),
        ],
        [
            Paragraph("<b>jerarquia</b><br/>(hierarchy)", cell_bold_style),
            Paragraph("<code>ndarray 3D</code><br/><code>(1, M, 4)</code><br/>tipo: <code>int32</code>", cell_style),
            Paragraph("Árbol de relaciones topológicas de los $M$ contornos. Para cada contorno $i$, contiene un vector de 4 enteros:<br/>"
                      "<code>[ Next, Previous, First_Child, Parent ]</code><br/>"
                      "• <b>[0] Next:</b> Siguiente en mismo nivel (-1 si no hay).<br/>"
                      "• <b>[1] Previous:</b> Anterior en mismo nivel (-1 si no hay).<br/>"
                      "• <b>[2] First_Child:</b> Primer contorno hijo interior (-1 si no hay).<br/>"
                      "• <b>[3] Parent:</b> Contorno padre que lo encierra (-1 si es exterior).", cell_style),
            Paragraph("<b>¿Por qué miramos j[3] != -1?</b><br/>"
                      "Al usar <code>RETR_CCOMP</code> se crea una jerarquía de exactamente 2 niveles: contornos exteriores y agujeros.<br/>"
                      "Si <code>Parent != -1</code> (la posición [3]), <b>el contorno es un agujero cerrado interno</b>.<br/>"
                      "• 'L': 0 agujeros (Euler).<br/>"
                      "• 'R' y 'A': 1 agujero (desempate por trazo vertical).", cell_alert_style),
        ],
    ]

    col_widths_ret_fc = [135, 140, 245, 258]
    tabla_ret_fc = Table(filas_ret_fc, colWidths=col_widths_ret_fc, repeatRows=1)
    tabla_ret_fc.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#1E3A8A")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(tabla_ret_fc)

    story.append(PageBreak())

    # =========================================================================
    # PÁGINA 3: NUMPY (np) - MÉTODOS Y DETALLE DE RETORNOS
    # =========================================================================
    story.append(Paragraph("3. Métodos y Funciones de NumPy (numpy / np)", seccion_verde))

    encabezados_np = [
        Paragraph("Función / Operación", th_style),
        Paragraph("Uso en el Código y Parámetros", th_style),
        Paragraph("¿Qué hace técnicamente?", th_style),
        Paragraph("Tip Clave para la Defensa Oral (¿Por qué se usa así?)", th_style),
    ]

    filas_np = [
        encabezados_np,
        [
            Paragraph("<b>np.sum</b><br/>(Proyecciones 1D)", cell_bold_style),
            Paragraph("• <code>np.sum(img_th, axis=1)</code> (filas)<br/>"
                      "• <code>np.sum(img_th, axis=0)</code> (columnas)<br/>"
                      "• <code>np.sum(huecos > umbral)</code>", cell_style),
            Paragraph("Calcula la suma marginal a lo largo de un eje:<br/>"
                      "• <b>axis=1:</b> suma horizontal (proyección en Y).<br/>"
                      "• <b>axis=0:</b> suma vertical (proyección en X).<br/>"
                      "• Sin eje: suma elementos booleanos.", cell_style),
            Paragraph("<b>¿Por qué proyecciones y no Hough?</b> Las líneas de la tabla son perfectamente ortogonales. La proyección 1D es lineal en tiempo $\\mathcal{O}(N \\times M)$, sin costo de acumulación ni falsos positivos por ruido.", cell_tip_style),
        ],
        [
            Paragraph("<b>np.where</b>", cell_bold_style),
            Paragraph("• <code>np.where(proyeccion > umbral)[0]</code><br/>"
                      "• <code>np.where(np.diff(idx) > 1)[0]</code>", cell_style),
            Paragraph("Devuelve los índices de las posiciones donde la condición booleana se cumple como verdadera (True).", cell_style),
            Paragraph("Permite aislar exactamente qué filas y columnas contienen líneas negras densas de la tabla, y detectar saltos entre líneas consecutivas.", cell_tip_style),
        ],
        [
            Paragraph("<b>np.diff</b>", cell_bold_style),
            Paragraph("<code>np.diff(idx)</code>", cell_style),
            Paragraph("Calcula la diferencia discreta hacia adelante entre elementos contiguos: $\\Delta x_i = x_{i+1} - x_i$.", cell_style),
            Paragraph("<b>Agrupamiento de líneas físicas:</b> Si una línea impresa mide 2 o 3 px de ancho, <code>np.diff(idx) == 1</code>. Cuando <code>np.diff(idx) > 1</code>, se detecta un salto a la <i>siguiente</i> línea de la tabla.", cell_tip_style),
        ],
        [
            Paragraph("<b>np.concatenate</b>", cell_bold_style),
            Paragraph("• <code>np.concatenate(([idx[0]], idx[cortes + 1]))</code><br/>"
                      "• <code>np.concatenate((idx[cortes], [idx[-1]]))</code>", cell_style),
            Paragraph("Une arreglos 1D a lo largo del eje especificado para construir los vectores de <code>inicios</code> y <code>fines</code>.", cell_style),
            Paragraph("Permite delimitar el intervalo <code>(inicio, fin)</code> de cada línea gruesa de la rejilla para luego indexar las celdas limpiamente.", cell_tip_style),
        ],
        [
            Paragraph("<b>np.argsort</b>", cell_bold_style),
            Paragraph("<code>stats[np.argsort(stats[:, cv2.CC_STAT_LEFT])]</code>", cell_style),
            Paragraph("Devuelve los índices que ordenarían el array de menor a mayor según la columna de posición X izquierda.", cell_style),
            Paragraph("<b>Orden natural de lectura:</b> OpenCV devuelve las componentes conexas en orden arbitrario. Con <code>argsort</code> las ordenamos estrictamente de izquierda a derecha.", cell_tip_style),
        ],
        [
            Paragraph("<b>np.ones / np.zeros</b>", cell_bold_style),
            Paragraph("• <code>np.ones((h, w, 3), dtype=np.uint8) * 250</code><br/>"
                      "• <code>np.zeros((h, w, 3), dtype=np.uint8)</code>", cell_style),
            Paragraph("Crea matrices inicializadas con unos o ceros con el tamaño y tipo de dato especificados.", cell_style),
            Paragraph("• <code>np.ones(...) * 250</code> genera un lienzo blanco/gris claro limpio para renderizar el panel de no aprobados.<br/>"
                      "• <code>np.zeros(...)</code> genera máscaras negras vacías para trazar contornos.", cell_tip_style),
        ],
        [
            Paragraph("<b>np.uint8</b>", cell_bold_style),
            Paragraph("• <code>celda.astype(np.uint8)</code><br/>"
                      "• <code>dtype=np.uint8</code>", cell_style),
            Paragraph("Tipo entero sin signo de 8 bits (rango 0 a 255). Es el tipo nativo exigido por casi todas las funciones de OpenCV.", cell_style),
            Paragraph("Si se le pasa un array booleano o float a funciones como <code>findContours</code> o <code>connectedComponents</code>, OpenCV lanza error. <code>uint8</code> garantiza compatibilidad estricta.", cell_tip_style),
        ],
    ]

    col_widths_np = [135, 175, 215, 253]
    tabla_np = Table(filas_np, colWidths=col_widths_np, repeatRows=1)
    tabla_np.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#065F46")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#065F46")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(tabla_np)
    story.append(Spacer(1, 6))

    # Subtabla C: Retornos de NumPy en Detalle
    story.append(Paragraph("Detalle Anatómico de Retornos en Métodos de NumPy", seccion_verde))

    encabezados_ret_np = [
        Paragraph("Función NumPy", th_style),
        Paragraph("Tipo y Forma (Shape)", th_style),
        Paragraph("¿Qué retorna exactamente?", th_style),
        Paragraph("¿Por qué se indexa así en el código?", th_style),
    ]

    filas_ret_np = [
        encabezados_ret_np,
        [
            Paragraph("<b>np.where(...)</b>", cell_bold_style),
            Paragraph("<code>tuple[ndarray]</code><br/>Longitud = dimensiones del input.", cell_style),
            Paragraph("Una tupla con un array de índices por cada dimensión donde la condición se cumple.", cell_style),
            Paragraph("<b>¿Por qué [0] al final?</b> Como la proyección es un vector 1D, devuelve <code>(array, )</code>. El <code>[0]</code> extrae directamente el array NumPy de índices sin la tupla envolvente.", cell_tip_style),
        ],
        [
            Paragraph("<b>np.diff(...)</b>", cell_bold_style),
            Paragraph("<code>ndarray 1D</code><br/>Longitud: $N - 1$", cell_style),
            Paragraph("Array con las diferencias finitas sucesivas: $\\text{out}[i] = \\text{idx}[i+1] - \\text{idx}[i]$.", cell_style),
            Paragraph("Permite saber si dos píxeles contiguos pertenecen a la misma línea gruesa ($\\Delta = 1$) o si hay un salto a la siguiente línea de la tabla ($\\Delta > 1$).", cell_tip_style),
        ],
        [
            Paragraph("<b>np.argsort(...)</b>", cell_bold_style),
            Paragraph("<code>ndarray 1D</code><br/>Longitud: $N$, enteros", cell_style),
            Paragraph("Arreglo con los <b>índices de permutación</b> que ordenarían los datos de menor a mayor.", cell_style),
            Paragraph("No ordena un vector suelto; nos permite <b>reordenar filas enteras de la matriz <code>stats</code></b> según la columna X (`CC_STAT_LEFT`) en orden de lectura natural.", cell_tip_style),
        ],
        [
            Paragraph("<b>.any(axis=1).mean()</b>", cell_bold_style),
            Paragraph("Paso 1: <code>ndarray (H,) bool</code><br/>Paso 2: <code>float</code> (escalar)", cell_style),
            Paragraph("• <code>any(axis=1)</code>: True si hay píxel de trazo en esa fila.<br/>"
                      "• <code>mean()</code>: Promedio aritmético (proporción entre 0.0 y 1.0).", cell_style),
            Paragraph("Mide la cobertura vertical del trazo en la franja izquierda. Si <code>cobertura > 0.90</code>, es la columna recta de la letra **'R'**; si no, es la letra **'A'**.", cell_alert_style),
        ],
    ]

    col_widths_ret_np = [135, 140, 245, 258]
    tabla_ret_np = Table(filas_ret_np, colWidths=col_widths_ret_np, repeatRows=1)
    tabla_ret_np.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#065F46")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#065F46")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(tabla_ret_np)

    story.append(PageBreak())

    # =========================================================================
    # PÁGINA 4: PREGUNTAS FRECUENTES Y RESPUESTAS CLAVE EN LA DEFENSA
    # =========================================================================
    story.append(Paragraph("4. Banco de Preguntas Clave para la Defensa Oral", seccion_rojo))
    story.append(Paragraph("Preguntas conceptuales y metodológicas típicas de la mesa evaluadora de la cátedra:", subtitulo_style))
    story.append(Spacer(1, 5))

    preguntas = [
        (
            "1. ¿Por qué no utilizaron una librería de OCR como Tesseract para leer los campos?",
            "Porque el trabajo práctico evalúa explícitamente los fundamentos de las <b>Unidades 1 a 3</b> (perfiles de proyección, segmentación, morfología, componentes conexas y Euler). Tesseract sería una 'caja negra' que no demuestra el dominio del análisis de bajo y medio nivel. La solución implementada es completamente determinística, no depende de modelos entrenados externos y cumple estrictamente el programa de la materia."
        ),
        (
            "2. ¿Por qué se utilizó una proyección 1D con np.sum en lugar de la Transformada de Hough para la grilla?",
            "La Transformada de Hough busca líneas en cualquier ángulo en un espacio de acumulación discretizado $(\\rho, \\theta)$, lo que demanda alto costo computacional y requiere post-filtrado para descartar líneas falsas por ruido o texto. Dado que el formulario está alineado y su grilla es ortogonal (horizontal y vertical), las <b>proyecciones 1D</b> reducen el problema a dos sumas marginales de complejidad lineal $\\mathcal{O}(N \\times M)$, siendo matemáticamente exactas, rápidas y robustas."
        ),
        (
            "3. ¿Cómo resuelve el código que las líneas de la grilla impresa tengan más de 1 píxel de espesor?",
            "Con la combinación de <code>np.diff</code> y <code>np.where</code> en la función <code>detectar_lineas</code>. Se agrupan píxeles contiguos cuya proyección supera el umbral. Cuando la diferencia entre índices consecutivos es 1 ($\\Delta = 1$), pertenecen a la misma línea gruesa. Al superarse la distancia de 1 px ($\\Delta > 1$), se marca el fin de la línea actual y el inicio de la siguiente, evitando contabilizar múltiples líneas por un mismo trazo."
        ),
        (
            "4. ¿Cómo diferencian una palabra de dos palabras sin reconocer caracteres?",
            "A través de los <b>gaps horizontales</b> entre componentes conexas ordenadas por su coordenada X izquierda (<code>cv2.CC_STAT_LEFT</code>). La distancia entre dos caracteres consecutivos es $\\text{gap} = \\text{LEFT}_{i+1} - (\\text{LEFT}_i + \\text{WIDTH}_i)$. Si ese espacio en blanco supera el umbral empírico calibrado (<code>UMBRAL_ESPACIO = 6 px</code>), se contabiliza como un espacio inter-palabra. Así: $\\text{Palabras} = \\text{espacios} + 1$."
        ),
        (
            "5. ¿Por qué se utiliza la jerarquía de contornos (findContours) para la Condición Final en lugar de plantillas?",
            "Porque las tres letras posibles ('L', 'R', 'A') tienen propiedades topológicas invariantes a tamaño, traslación y fuente: la 'L' es simplemente conexa y posee <b>0 agujeros internos</b> (género de Euler 1), mientras que la 'R' y la 'A' poseen exactamente <b>1 agujero cerrado</b>. Al usar <code>cv2.RETR_CCOMP</code>, cualquier contorno con <code>Parent != -1</code> es un agujero interior. Finalmente, 'R' se diferencia de 'A' porque la 'R' tiene un asta vertical continua a la izquierda (cobertura $> 90\\%$)."
        ),
        (
            "6. ¿Qué funciones propias componen el pipeline del script y qué devuelven?",
            "• <code>detectar_lineas(proyeccion, umbral)</code>: devuelve lista de tuplas <code>(inicio, fin)</code> de cada línea gruesa.<br/>"
            "• <code>detectar_celdas(img_th)</code>: devuelve <code>filas, cols, registros_coords, campos</code> con los límites de corte $(y_0, y_1, x_0, x_1)$ de cada celda.<br/>"
            "• <code>componentes(celda)</code>: devuelve la matriz <code>stats</code> filtrada (sin fondo y con $\\text{área} > 2\\text{ px}$) y ordenada de izquierda a derecha.<br/>"
            "• <code>contar(celda)</code>: devuelve la tupla <code>(cant_caracteres, cant_palabras)</code>."
        ),
    ]

    filas_faq = []
    for q, a in preguntas:
        filas_faq.append([
            Paragraph(f"<b>{q}</b>", pregunta_style),
            Paragraph(a, respuesta_style),
        ])

    tabla_faq = Table(filas_faq, colWidths=[240, 538])
    tabla_faq.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E2E8F0")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#FEF2F2"), colors.white]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(tabla_faq)

    # Compilar PDF con canvas numerado de dos pasadas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generado exitosamente en: {ruta_salida}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    salida = os.path.join(base_dir, "resumen_defensa_cv2_numpy.pdf")
    crear_pdf(salida)
