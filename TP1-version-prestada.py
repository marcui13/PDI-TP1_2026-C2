import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

CAMPOS = ["Legajo", "Nombre y apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"]
UMBRAL_BINARIO = 150       # píxel < 150 se considera tinta
UMBRAL_ESPACIO = 6         # hueco horizontal (px) a partir del cual empieza otra palabra
UMBRAL_AREA = 2            # manchas con área <= 2 px se descartan (restos de líneas / ruido)


def detectar_lineas(proyeccion, umbral):
    """Devuelve (inicio, fin) de cada línea, agrupa píxeles consecutivos por si la línea tiene más de 1 px."""
    idx = np.where(proyeccion > umbral)[0]
    if len(idx) == 0:
        return []
    cortes = np.where(np.diff(idx) > 1)[0]
    inicios = np.concatenate(([idx[0]], idx[cortes + 1]))
    fines = np.concatenate((idx[cortes], [idx[-1]]))
    return list(zip(inicios, fines))


def detectar_celdas(img_th):
    """Devuelve una lista (una por registro) de 6 sub-imágenes binarias, una por campo."""
    alto, ancho = img_th.shape
    filas = detectar_lineas(np.sum(img_th, 1), 0.5 * ancho)
    cols = detectar_lineas(np.sum(img_th, 0), 0.4 * alto)
    # filas[0] = borde superior de la tabla, filas[1] = fin del encabezado, luego una línea por registro
    # cols[0] = borde izquierdo, cols[1] = fin de "Nro.", luego una línea por campo
    registros = []
    for r in range(1, len(filas) - 1):
        y0, y1 = filas[r][1] + 1, filas[r + 1][0]
        celdas = []
        for c in range(1, len(cols) - 1):
            x0 = cols[c][1] + 1      # un píxel después de donde termina la línea vertical de la izquierda
            x1 = cols[c + 1][0]      # donde empieza la línea vertical de la derecha
            celda = img_th[y0:y1, x0:x1]
            celdas.append(celda)
        registros.append(celdas)
    return registros


def componentes(celda):
    """Stats de las componentes (sin el fondo ni las muy chicas), ordenadas de izquierda a derecha."""
    # connectedComponents necesita números, no True/False
    n, _, stats, _ = cv2.connectedComponentsWithStats(celda.astype(np.uint8), 8, cv2.CV_32S)
    stats = stats[1:]   # sacamos el fondo
    stats = stats[stats[:, cv2.CC_STAT_AREA] > UMBRAL_AREA]
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
    if campo == "Legajo":
        ok = cant_caracteres == 8 and cant_palabras == 1
    elif campo == "Nombre y apellido":
        ok = cant_palabras >= 2 and cant_caracteres <= 12
    elif campo == "Condición Final":
        ok = cant_caracteres == 1
    else:   # Parcial 1, 2 y 3
        ok = cant_palabras == 1 and 1 <= cant_caracteres <= 2
    return "OK" if ok else "MAL"


def clasificar_condicion(celda):
    """
    Distingue L / R / A por forma:
     - L no tiene agujeros.
     - R y A tienen un agujero. R tiene un trazo vertical completo en su lado izquierdo, A no.
    """
    stats = componentes(celda)
    x, y, w, h = stats[0, :4]
    letra = celda[y:y + h, x:x + w].astype(np.uint8)
    contornos, jerarquia = cv2.findContours(letra, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    agujeros = sum(1 for j in jerarquia[0] if j[3] != -1)
    if agujeros == 0:
        return "L"
    columna_izquierda = letra[:, :max(1, w // 5)].any(axis=1).mean()
    return "R" if columna_izquierda > 0.9 else "A"


def generar_imagen_salida(no_aprobados, ruta_salida):
    if not no_aprobados:
        img_salida = np.full((40, 520), 255, np.uint8)
        cv2.putText(img_salida, "Sin alumnos L o R para informar", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    else:
        ancho = max(c.shape[1] for c, _ in no_aprobados) + 120
        filas = []
        for crop, condicion in no_aprobados:
            nombre = np.where(crop, 0, 255).astype(np.uint8)            # texto negro sobre blanco
            fila = np.full((nombre.shape[0] + 10, ancho), 255, np.uint8)
            fila[5:5 + nombre.shape[0], 5:5 + nombre.shape[1]] = nombre
            etiqueta = "RECUPERA" if condicion == "R" else "LIBRE"
            cv2.rectangle(fila, (0, 0), (ancho - 1, fila.shape[0] - 1), 0, 2)
            cv2.putText(fila, etiqueta, (ancho - 110, fila.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)
            filas.append(fila)
        img_salida = np.vstack(filas)
    cv2.imwrite(ruta_salida, img_salida)
    return img_salida


def procesar_planilla(ruta, ruta_csv, ruta_salida):
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer {ruta}")
    img_th = img < UMBRAL_BINARIO   # True donde el píxel es oscuro (tinta), False en el fondo
    registros = detectar_celdas(img_th)

    # a) validación por registro
    resultados = []
    for id_registro, celdas in enumerate(registros, start=1):
        print(f"> Registro {id_registro}:")
        estados = []
        for campo, celda in zip(CAMPOS, celdas):
            estado = validar(campo, *contar(celda))
            print(f"> {campo}: {estado}")
            estados.append(estado)
        print(">")
        resultados.append(estados)

    # c) CSV
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Legajo", "Nombre y Apellido", "Parcial 1", "Parcial 2", "Parcial 3", "Condición Final"])
        for id_registro, estados in enumerate(resultados, start=1):
            writer.writerow([id_registro] + estados)

    # b) imagen de salida con los alumnos no aprobados (solo registros completamente OK)
    no_aprobados = []
    for celdas, estados in zip(registros, resultados):
        if all(e == "OK" for e in estados):
            condicion = clasificar_condicion(celdas[5])
            if condicion in ("L", "R"):
                no_aprobados.append((celdas[1], condicion))
    generar_imagen_salida(no_aprobados, ruta_salida)
    return resultados, [c for _, c in no_aprobados]


if __name__ == "__main__":
    # Exploración paso a paso sobre la planilla 1
    img = cv2.imread("grade_sheet_1.png", cv2.IMREAD_GRAYSCALE)
    #print(img.shape)
    plt.imshow(img, cmap='gray')
    plt.title("Planilla original")
    plt.show()

    img_th = img < UMBRAL_BINARIO   # True donde el píxel es oscuro (tinta), False en el fondo
    plt.imshow(img_th, cmap='gray')
    plt.title("Imagen umbralada")
    plt.show()

    alto, ancho = img_th.shape
    plt.plot(np.sum(img_th, 1))   # suma cada fila: cuántos píxeles de tinta tiene
    plt.axhline(0.5 * ancho, color='r', linestyle='--')
    plt.title("Píxeles de tinta por fila")
    plt.xlabel("Número de fila")
    plt.show()
    '''
    De 0 a 200 aprox.: es el encabezado de la planilla, con la imagen de los robots, el título
    "Procesamiento de Imágenes 1" y el logo. Tiene bastantes píxeles oscuros, pero ninguna fila llega a ser una línea completa.
    Pico en ~210: es la línea de arriba de la tabla, donde empieza el encabezado "Nro., Legajo, Nombre y Apellido...".
    Pico más bajo en ~260, que llega a ~380: es la línea que separa "Notas" de "Parcial 1, Parcial 2, Parcial 3".
    Es más corta porque solo cruza esas tres columnas, no todo el ancho, así que queda debajo del umbral (línea roja).
    Desde ~285 hasta ~805: una seguidilla de picos altos, muy parejos y a distancias iguales. Son las líneas que separan los 20 registros.
    Entre los picos, valores de ~100: es el texto de cada fila (el número de registro y lo que esté escrito).
    '''

    plt.plot(np.sum(img_th, 0))   # suma cada columna
    plt.axhline(0.4 * alto, color='r', linestyle='--')
    plt.title("Píxeles de tinta por columna")
    plt.xlabel("Número de columna")
    plt.show()
    # Los picos que pasan la línea roja son las 8 líneas verticales de la tabla

    registros = detectar_celdas(img_th)
    plt.figure(figsize=(15, 3))
    for i in range(6):
        plt.subplot(1, 6, i + 1)
        plt.imshow(registros[0][i], cmap='gray')
        plt.title(CAMPOS[i])
    plt.show()
    # Las celdas del registro 1 quedan recortadas sin restos de las líneas de la tabla

    stats = componentes(registros[0][1])   # Nombre y apellido del registro 1
    for i in range(1, len(stats)):
        fin_anterior = stats[i - 1, 0] + stats[i - 1, 2]   # x + ancho
        print("hueco:", stats[i, 0] - fin_anterior)
    '''
    10 huecos (11 letras), todos de entre 1 y 3 píxeles, salvo uno de 12. Ese es el espacio entre el nombre y el
    apellido, así que el Nombre y Apellido son dos palabras. Un umbral de 6 separa bien los dos casos, con margen de los dos lados:
     - Hueco menor o igual a 6: siguiente letra de la misma palabra.
     - Hueco mayor a 6: empieza una palabra nueva.
    La cantidad de palabras es la cantidad de espacios más uno: con un espacio hay dos palabras, con cero hay una.
    '''

    print("Caracteres y palabras:", contar(registros[0][1]))
    # Para JUAN CARLINI debería imprimir (11, 2)

    # d) aplicación cíclica sobre las 4 planillas
    for id_planilla in range(1, 5):
        ruta = f"grade_sheet_{id_planilla}.png"
        print(f"===== Planilla {id_planilla} =====")
        try:
            _, condiciones = procesar_planilla(ruta, f"resultados_{id_planilla}.csv", f"no_aprobados_{id_planilla}.png")
            print("No aprobados:", condiciones)
        except FileNotFoundError as e:
            print(e)
