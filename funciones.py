import os
import numpy as np
import matplotlib.image as mpimg

# --- VALIDACIONES Y UTILIDADES ---
dimensiones_validas_imagen = [2, 3]

def _es_imagen_valida(matriz):
    """Valida si la matriz no es None y tiene un formato soportado (2D o 3D)."""
    if matriz is None:
        return False
    return matriz.ndim in dimensiones_validas_imagen

def _obtener_extension(ruta):
    """Obtiene la extensión del archivo en formato .EXT (mayúsculas)."""
    return os.path.splitext(ruta)[1].upper()

def _escalar_a_8bit(matriz):
    """
    Transformación lineal para remapear cualquier rango al intervalo [0, 255]
    y evitar pérdida de información.
    """
    min_val = np.min(matriz)
    max_val = np.max(matriz)

    # Caso imagen uniforme para evitar división por cero
    if max_val - min_val == 0:
        return np.zeros(matriz.shape, dtype=np.uint8)

    # s = (r - min) * (255 / (max - min))
    matriz_escalada = (matriz - min_val) * (255.0 / (max_val - min_val))

    return matriz_escalada.astype(np.uint8)

# --- INICIO TP 0 ---
# --- CARGA Y GUARDADO ---

def _cargar_raw_8bit(ruta, ancho, alto):
    """Procesa archivos RAW. Detecta Gris o RGB por peso de archivo."""
    datos = np.fromfile(ruta, dtype=np.uint8)
    tamanio_esperado = ancho * alto

    # Validación de tamaño físico vs resolución solicitada
    if len(datos) < tamanio_esperado:
        raise ValueError(
            f"El archivo no contiene suficientes datos para la resolución indicada ({ancho}x{alto})."
            " Verifique las dimensiones o si el archivo está corrupto."
        )

    if len(datos) >= tamanio_esperado * 3:
        # Caso RGB
        return datos[:tamanio_esperado * 3].reshape((alto, ancho, 3))
    else:
        # Caso Gris
        return datos[:tamanio_esperado].reshape((alto, ancho))

def _cargar_formato_con_cabecera(ruta):
    """Carga formatos estándar (JPG, PNG, etc.) y normaliza a uint8 (0-255)."""
    matriz = mpimg.imread(ruta)
    # Si la imagen viene en punto flotante, la escalamos
    if matriz.dtype == np.float32 or matriz.dtype == np.float64:
        matriz = (matriz * 255).astype(np.uint8)
    return matriz

def cargar_imagen(ruta, ancho=None, alto=None):
    """Punto de entrada único para cargar cualquier imagen soportada."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    ext = _obtener_extension(ruta)
    if ext == '.RAW':
        return _cargar_raw_8bit(ruta, ancho, alto)
    return _cargar_formato_con_cabecera(ruta)

def guardar_imagen(matriz, ruta):
    """
    Guarda la matriz en disco detectando el formato por la extensión.
    Para RAW guarda el binario directo; para otros usa matplotlib.
    """
    if not _es_imagen_valida(matriz):
        raise ValueError("Matriz inválida para guardado.")

    ext = _obtener_extension(ruta)

    if ext == '.RAW':
        # Clipping de seguridad antes de convertir a uint8 
        m_save = np.clip(matriz, 0, 255).astype(np.uint8)
        m_save.tofile(ruta)
    else:
        # matplotlib maneja el guardado de formatos estándar
        if matriz.ndim == 2:
            mpimg.imsave(ruta, matriz, cmap='gray')
        else:
            mpimg.imsave(ruta, matriz)

# --- MANIPULACIÓN DE PÍXELES Y REGIONES ---

def obtener_pixel(matriz, x, y):
    """Retorna intensidad (int) o color (array [R,G,B])."""
    if not _es_imagen_valida(matriz):
        raise ValueError("Imagen no soportada.")
    alto, ancho = matriz.shape[:2]
    if not (0 <= x < ancho and 0 <= y < alto):
        raise IndexError(f"Coordenadas fuera de rango: {x},{y}")
    return matriz[y, x]

def modificar_pixel(matriz, x, y, nuevo_valor):
    """Retorna una copia de la matriz con el valor del píxel modificado."""
    if not _es_imagen_valida(matriz):
        raise ValueError("Imagen no soportada.")
    alto, ancho = matriz.shape[:2]
    if not (0 <= x < ancho and 0 <= y < alto):
        raise IndexError("Coordenadas fuera de rango.")

    nueva = matriz.copy()
    nueva[y, x] = nuevo_valor
    return nueva

def copiar_region(matriz, x1, y1, x2, y2):
    """Extrae una sub-matriz validando límites."""
    if not _es_imagen_valida(matriz):
        raise ValueError("Imagen no soportada.")
    alto, ancho = matriz.shape[:2]

    # Validación
    if not (0 <= x1 <= ancho and 0 <= y1 <= alto):
      raise IndexError("Punto 1 fuera de límites.")
    if not (0 <= x2 <= ancho and 0 <= y2 <= alto):
      raise IndexError("Punto 2 fuera de límites.")

    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)

    return matriz[ymin:ymax, xmin:xmax].copy()

# --- OPERACIONES ARITMÉTICAS ---

def restar_imagenes(img1, img2):
    """Realiza la resta img1 - img2. Soporta Gris-Gris, RGB-RGB y Mixtos."""
    if not (_es_imagen_valida(img1) and _es_imagen_valida(img2)):
        raise ValueError("Formatos de imagen no soportados.")
    if img1.shape[:2] != img2.shape[:2]:
        raise ValueError("Las resoluciones deben ser idénticas para la resta.")

    # Convertimos a float para el cálculo y evitar overflow/underflow
    m1 = img1.astype(np.float32)
    m2 = img2.astype(np.float32)

    if m1.ndim == m2.ndim:
        resultado = m1 - m2
    elif m1.ndim == 2: # Gris - RGB
        r_r = m1 - m2[:, :, 0]
        r_g = m1 - m2[:, :, 1]
        r_b = m1 - m2[:, :, 2]
        resultado = (r_r + r_g + r_b) / 3
    else: # RGB - Gris
        r_r = m1[:, :, 0] - m2
        r_g = m1[:, :, 1] - m2
        r_b = m1[:, :, 2] - m2
        resultado = (r_r + r_g + r_b) / 3

    min_val = np.min(resultado)
    max_val = np.max(resultado)

    # EVITAR DIVISIÓN POR CERO
    if max_val - min_val == 0:
        return np.zeros(resultado.shape, dtype=np.uint8)

    # ESCALADO
    resultado = (resultado - min_val) * (255.0 / (max_val - min_val))
    
    return resultado.astype(np.uint8)

# --- ESTADÍSTICAS DE REGIÓN ---

def obtener_estadisticas_region(matriz, x1, y1, x2, y2):
    """ Calcula cantidad de píxeles y el promedio de gris o color de la región. """
    region = copiar_region(matriz, x1, y1, x2, y2)

    alto, ancho = region.shape[:2]
    total_pixeles = alto * ancho

    if total_pixeles == 0:
        return 0, 0

    if region.ndim == 2:
        # Promedio simple para niveles de gris
        promedio = np.uint8(np.mean(region))
    else:
        # Promedio por componente para RGB
        promedio_rgb = np.mean(region, axis=(0, 1))
        promedio = promedio_rgb.astype(np.uint8)

    return total_pixeles, promedio

# --- INICIO TP 1 ---
# --- AUMENTO DE CONTRASTE (POTENCIA / GAMMA) ---

def aplicar_transformacion_potencia(matriz_original, gamma):
    """
    Realiza el realce de contraste mediante la función de potencia.
    s = c * r^gamma, donde 'r' es la intensidad original y 's' la transformada.
    Gamma < 1: Aclara la imagen (realce de sombras).
    Gamma > 1: Oscurece la imagen (realce de altas luces).
    """
    if not _es_imagen_valida(matriz_original):
        raise ValueError("Imagen no soportada.")

    if not (0 < gamma < 2):
        raise ValueError("El valor de gamma debe estar en el rango (0, 2).")

    if gamma == 1:
        return matriz_original.copy() # Evitamos cálculos innecesarios para la identidad

    # 'c' es la constante de normalización para asegurar que el rango dinámico sea [0, 255]
    # c = 255 / (255^gamma) => 255^(1-gamma)
    constante_c = 255**(1 - gamma)

    # Conversión a float32 para evitar errores de precisión durante la potencia
    matriz_float = matriz_original.astype(np.float32)
    matriz_transformada = constante_c * (matriz_float ** gamma)

    # Retornamos el escalado
    return _escalar_a_8bit(matriz_transformada)

# --- NEGATIVO ---

def obtener_negativo(matriz_original):
    """
    Obtiene el negativo de la imagen.
    """
    if not _es_imagen_valida(matriz_original):
        raise ValueError("Imagen no soportada.")

    return 255 - matriz_original.astype(np.uint8)

# --- HISTOGRAMA ---

def obtener_histograma_gris(matriz_original):
    """
    Calcula el histograma normalizado (frecuencias relativas).
    Retorna un arreglo de 256 elementos.
    """
    if not _es_imagen_valida(matriz_original):
        raise ValueError("Imagen no soportada.")

    # Inicialización del arreglo de frecuencias para 256 niveles
    histograma = np.zeros(256, dtype=int)
    alto, ancho = matriz_original.shape[:2]

    for y in range(alto):
        for x in range(ancho):
            if matriz_original.ndim == 3:
                # Promedio manual para convertir a gris
                suma = 0.0
                for canal in range(3):
                    # Convertimos explícitamente a float para evitar el overflow del uint8
                    suma += float(matriz_original[y, x, canal])
                valor = int(suma / 3)
            else:
                valor = int(matriz_original[y, x])

            # Incremento de la frecuencia del nivel detectado
            histograma[valor] += 1
    return histograma

# --- UMBRALIZACIÓN ---

def obtener_umbralizacion(matriz_original, umbral):
    """
    Aplica umbralización binaria a la imagen.
    """
    if not _es_imagen_valida(matriz_original):
        raise ValueError("Imagen no soportada.")

    # Validación de dominio para el umbral
    if not (0 <= umbral <= 255):
        raise ValueError("El umbral debe estar entre 0 y 255.")

    alto, ancho = matriz_original.shape[:2]

    # Preparar matriz de salida (siempre 2D para imagen binaria)
    resultado = np.zeros((alto, ancho), dtype=np.uint8)

    # Recorrido manual
    for y in range(alto):
        for x in range(ancho):

            # Obtener valor de gris (x,y)
            if matriz_original.ndim == 3:
                # Promedio de los 3 canales (R+G+B)/3
                suma_canales = 0.0
                for canal in range(3):
                    suma_canales += float(matriz_original[y, x, canal])
                valor_gris = suma_canales / 3
            else:
                valor_gris = matriz_original[y, x]

            # Aplicar lógica de umbral
            if valor_gris >= umbral:
                resultado[y, x] = 255
            else:
                resultado[y, x] = 0

    return resultado

# --- ECUALIZACIÓN DEL HISTOGRAMA ---

def ecualizar_histograma(matriz_original):
    """
    Aplica la ecualización global basada en la probabilidad de ocurrencia de los niveles de gris.
    s_k = T(r_k) = (L-1) * sum_{j=0}^{k} p_r(r_j)
    """
    if not _es_imagen_valida(matriz_original):
        raise ValueError("Imagen no soportada.")

    histograma = obtener_histograma_gris(matriz_original)
    alto, ancho = matriz_original.shape[:2]
    total_pixeles = alto * ancho

    # Cálculo de la probabilidad acumulada
    fda = np.zeros(256, dtype=float)
    acumulado = 0
    for i in range(256):
        acumulado += histograma[i]
        fda[i] = acumulado / total_pixeles

    resultado = np.zeros_like(matriz_original)

    # Mapeo de píxeles
    for y in range(alto):
        for x in range(ancho):
            if matriz_original.ndim == 3:
                for canal in range(3):
                    nuevo_valor = int(fda[matriz_original[y, x, canal]] * 255)
                    resultado[y, x, canal] = max(0, min(255, nuevo_valor))
            else:
                nuevo_valor = int(fda[matriz_original[y, x]] * 255)
                resultado[y, x] = max(0, min(255, nuevo_valor))
    return resultado

# --- GENERADORES ALEATORIOS ---
# --- GENERADOR GAUSSIANO ---

def generar_gaussiano(sigma, forma, media=0):
    """
    Genera ruido con distribución Normal (Gaussiana).
    """
    if sigma < 0:
        raise ValueError(f"El desvío estándar (sigma={sigma}) no puede ser negativo.")

    ruido_gaussiano = np.random.normal(loc=media, scale=sigma, size=forma)
    return ruido_gaussiano

# --- GENERADOR EXPONENCIAL ---

def generar_exponencial(lambd, forma):
    """
    Genera ruido con distribución Exponencial.
    """
    if lambd <= 0:
        raise ValueError("El parámetro lambda debe ser mayor a 0.")

    ruido_exponencial = np.random.exponential(scale=1.0/lambd, size=forma)
    return ruido_exponencial

# --- RUIDO ADITIVO GAUSSIANO ---

def aplicar_ruido_aditivo_gaussiano(matriz_original, densidad, sigma):
    """
    Contamina la imagen sumando ruido con distribución N(0, sigma) al conjunto D.
    Ic(i,j) = I(i,j) + ruido  si (i,j) pertenece al conjunto de píxeles D.
    """
    if sigma < 0:
        raise ValueError(f"El desvío (sigma={sigma}) debe ser mayor o igual a 0.")
    if not (0 <= densidad <= 100):
        raise ValueError("La densidad d debe estar entre 0 y 100.")

    matriz_float = matriz_original.astype(np.float64)

    # Definir el porcentaje de contaminación
    probabilidad = densidad / 100.0

    # Máscara booleana que define aleatoriamente qué píxeles serán contaminados
    mascara = np.random.choice([True, False], size=matriz_original.shape[:2], p=[probabilidad, 1 - probabilidad])

    # Generar ruido gaussiano solo para la forma de la imagen
    # con valores aleatorios con media 0 y desvío estándar sigma
    ruido_generado = generar_gaussiano(sigma, matriz_original.shape[:2])

    # Aplicar la suma solo a los píxeles seleccionados
    if matriz_original.ndim == 3: # Imagen color
        for canal in range(3):
            matriz_float[:,:,canal][mascara] += ruido_generado[mascara]
    else: # Imagen gris
        matriz_float[mascara] += ruido_generado[mascara]

    # Clipping para mantener el rango uint8
    return np.clip(matriz_float, 0, 255).astype(np.uint8)

# --- RUIDO MULTIPLICATIVO EXPONENCIAL ---

def aplicar_ruido_multiplicativo_exponencial(matriz_original, porcentaje, lambd):
    """
    Multiplica un porcentaje de los píxeles por ruido Exponencial.
    """
    if lambd <= 0:
        raise ValueError("El parámetro lambda debe ser mayor a 0.")
    if not (0 <= porcentaje <= 100):
        raise ValueError("El porcentaje debe estar entre 0 y 100.")

    res = matriz_original.astype(np.float64)

    # Definir el porcentaje de contaminación d
    probabilidad = porcentaje / 100.0

    # Elegir aleatoriamente d pixels para el conjunto D
    mascara = np.random.choice([True, False], size=matriz_original.shape[:2], p=[probabilidad, 1 - probabilidad])

    # Generar ruido exponencial solo para la forma de la imagen
    ruido = generar_exponencial(lambd, matriz_original.shape[:2])

    # Aplicar la multiplicación solo a los seleccionados
    if matriz_original.ndim == 3:
        for i in range(3):
            res[:,:,i][mascara] *= ruido[mascara]
    else:
        res[mascara] *= ruido[mascara]

    return np.clip(res, 0, 255).astype(np.uint8)

# --- RUIDO SAL Y PIMIENTA ---

def aplicar_ruido_sal_y_pimienta(matriz_original, densidad):
    """
    Sustituye píxeles por 0 (pimienta) o 255 (sal) según una densidad p.
    """
    if not (0 <= densidad <= 100):
        raise ValueError("La densidad debe estar entre 0 y 100.")

    matriz_ruidosa = matriz_original.copy()

    # p es la probabilidad para cada extremo (sal o pimienta)
    # Si la densidad total es 10%, p = 0.05 (5% sal, 5% pimienta)
    probabilidad = (densidad / 100.0) / 2.0

    # Generamos una matriz aleatoria uniforme [0, 1]
    # Usamos shape[:2] para que funcione igual en gris y color
    variables_aleatorias = np.random.rand(*matriz_original.shape[:2])

    # Aplicamos las sustituciones
    matriz_ruidosa[variables_aleatorias <= probabilidad] = 0           # Pimienta
    matriz_ruidosa[variables_aleatorias > (1 - probabilidad)] = 255    # Sal

    return matriz_ruidosa

# --- FILTROS DESLIZANTES ---
# --- FILTRO DE LA MEDIA ---

def aplicar_filtro_media(matriz_original, tamano_mascara):
    """
    Filtro de la Media.
    Consiste en pasar una máscara con pesos (núcleo) donde todos los elementos valen 1/N.
    Reemplaza el centro por el promedio de la vecindad.
    """
    if tamano_mascara % 2 == 0:
        raise ValueError(f"El tamaño de la máscara debe ser impar.")

    alto, ancho = matriz_original.shape[:2]
    offset = tamano_mascara // 2
    resultado = np.copy(matriz_original)
    area = tamano_mascara * tamano_mascara

    # Recorremos la imagen evitando los bordes
    for y in range(offset, alto - offset):
        for x in range(offset, ancho - offset):
            if matriz_original.ndim == 3:
                for canal in range(3):
                    suma = 0
                    # Convolución de la vecindad
                    for my in range(-offset, offset + 1):
                        for mx in range(-offset, offset + 1):
                            suma += int(matriz_original[y + my, x + mx, canal])
                    resultado[y, x, canal] = suma // area
            else:
                suma = 0
                for my in range(-offset, offset + 1):
                    for mx in range(-offset, offset + 1):
                        suma += int(matriz_original[y + my, x + mx])
                resultado[y, x] = suma // area
    return resultado

# --- FILTRO DE LA MEDIANA ---

def aplicar_filtro_mediana(matriz_original, tamano_mascara):
    """
    Filtro de la Mediana.
    Consiste en ordenar los valores de la vecindad y tomar el del medio
    Efectivo para eliminar ruido Sal y Pimienta.
    """
    if tamano_mascara % 2 == 0:
        raise ValueError(f"El tamaño de la máscara debe ser impar.")

    alto, ancho = matriz_original.shape[:2]
    offset = tamano_mascara // 2
    matriz_filtrada = matriz_original.copy()

    for y in range(offset, alto - offset):
        for x in range(offset, ancho - offset):
            vecindad = matriz_original[y-offset:y+offset+1, x-offset:x+offset+1]

            # Ordenamos y tomamos el valor central
            matriz_filtrada[y, x] = np.median(vecindad, axis=(0, 1))

    return matriz_filtrada

# --- GENERA LA MASCARA PARA EL FILTRO DE LA MEDIANA PONDERADA ---

def generar_pesos_mediana_ponderada(tamano_mascara):
    """
    Genera una matriz de pesos enteros basada en una distribución gaussiana.
    El peso máximo está en el centro y decae hacia los bordes.
    """
    # Crear cuadrícula centrada
    radio = tamano_mascara // 2
    eje = np.linspace(-radio, radio, tamano_mascara)
    x, y = np.meshgrid(eje, eje)

    # Calcular importancia gaussiana
    distancia_cuadrada = x**2 + y**2
    sigma = tamano_mascara / 3
    pesos = np.exp(-distancia_cuadrada / (2 * sigma**2))

    # Escalar para obtener repeticiones enteras (mínimo 1)
    return (pesos / pesos.min()).astype(int)

# --- FILTRO DE LA MEDIANA PONDERADA ---

def aplicar_filtro_mediana_ponderada(matriz_original, tamano_mascara):
    """
    Filtro de la mediana ponderada.
    Halla la mediana de los valores del entorno repetidos según una matriz de pesos dinámicos.
    """
    if tamano_mascara % 2 == 0:
        raise ValueError(f"El tamaño de la máscara debe ser impar.")

    alto, ancho = matriz_original.shape[:2]
    offset = tamano_mascara // 2
    matriz_filtrada = matriz_original.copy()

    # Generar la máscara de pesos basada en la importancia del píxel central
    mascara_pesos = generar_pesos_mediana_ponderada(tamano_mascara)

    for y in range(offset, alto - offset):
        for x in range(offset, ancho - offset):
            vecindad = matriz_original[y-offset:y+offset+1, x-offset:x+offset+1]

            if matriz_original.ndim == 2:

                # Caso niveles de gris
                valores_expandidos = []
                for i in range(tamano_mascara):
                    for j in range(tamano_mascara):

                        # Repetir el pixel tantas veces como indique su peso
                        valores_expandidos.extend([vecindad[i, j]] * mascara_pesos[i, j])
                matriz_filtrada[y, x] = np.median(valores_expandidos)
            else:

                # Caso RGB: Procesar cada canal por separado
                for canal in range(3):
                    valores_expandidos = []
                    for i in range(tamano_mascara):
                        for j in range(tamano_mascara):
                            valores_expandidos.extend([vecindad[i, j, canal]] * mascara_pesos[i, j])
                    matriz_filtrada[y, x, canal] = np.median(valores_expandidos)

    return matriz_filtrada

# --- GENERA LA MASCARA PARA FILTRO GAUSSIANO ---

def generar_mascara_gaussiana(sigma, tamano_mascara):
    """
    Genera el núcleo (kernel) de pesos siguiendo la Función Gaussiana
    de dos variables: G(x,y) = (1 / 2πσ²) * exp(-(x² + y²) / σ²)
    donde μ=(0,0) es el centro de la máscara.
    """
    # Definición del entorno o vecindad respecto al pixel central (0,0)
    radio = tamano_mascara // 2
    mascara = np.zeros((tamano_mascara, tamano_mascara), dtype=float)

    # Coeficiente: 1 / (2 * π * σ²)
    coeficiente = 1 / (2 * np.pi * (sigma**2))
    suma_total = 0.0

    # Iteración para el cálculo de distancias y pesos
    for y in range(-radio, radio + 1):
        for x in range(-radio, radio + 1):
            
            # Distancia cuadrada: (x - μ1)² + (y - μ2)² con μ = (0,0)
            distancia_cuadrada = x**2 + y**2
            
            # Exponente: -(distancia_cuadrada) / σ²
            exponente = -distancia_cuadrada / (sigma**2)

            # Función completa G(x,y)
            valor = coeficiente * np.exp(exponente)
            
            mascara[y + radio, x + radio] = valor
            suma_total += valor
            
    # Normalización manual para que la suma de pesos sea 1
    for y in range(tamano_mascara):
        for x in range(tamano_mascara):
            mascara[y, x] /= suma_total
            
    return mascara

# --- FILTRO GAUSSIANO ---

def aplicar_filtro_gaussiano(matriz_original, sigma):
    """
    Reemplaza el valor original del pixel con la suma de los valores
    originales multiplicados por los pesos de la máscara.
    """
    if sigma <= 0:
        raise ValueError("El desvío estándar sigma debe ser mayor a 0.")

    # El tamaño de la máscara debe ser acorde al valor de σ
    # Se utiliza k = 2 * σ + 1
    tamano_mascara = int(2 * sigma + 1)
    
    if tamano_mascara % 2 == 0:
        raise ValueError(f"El desvío (sigma={sigma}) debe generar una máscara de tamaño impar. Ingrese otro valor.")

    # Obtención de la máscara de pesos
    mascara = generar_mascara_gaussiana(sigma, tamano_mascara)
    offset = tamano_mascara // 2
    alto, ancho = matriz_original.shape[:2]

    # Trabajamos en float32 para el cálculo
    resultado = np.zeros(matriz_original.shape, dtype=np.float32)
    matriz_float = matriz_original.astype(np.float32)

    # Navegación por la imagen (Ventana deslizante)
    for y in range(offset, alto - offset):
        for x in range(offset, ancho - offset):
            
            if matriz_original.ndim == 3:  # Caso multicanal (RGB)
                for canal in range(3):
                    acumulado = 0.0

                    # Convolución manual: suma de productos de la vecindad por pesos
                    for my in range(-offset, offset + 1):
                        for mx in range(-offset, offset + 1):
                            acumulado += matriz_float[y + my, x + mx, canal] * mascara[my + offset, mx + offset]
                    resultado[y, x, canal] = acumulado
            else:  # Caso escala de grises (2D)
                acumulado = 0.0
                for my in range(-offset, offset + 1):
                    for mx in range(-offset, offset + 1):
                        acumulado += matriz_float[y + my, x + mx] * mascara[my + offset, mx + offset]
                resultado[y, x] = acumulado
    
    # Retornamos el escalado
    return _escalar_a_8bit(resultado)

# --- GENERA LA MASCARA REALCE DE BORDES ---

def generar_mascara_realce_de_bordes(tamano_mascara):
    """
    Genera la máscara de realce de tamaño variable
    Posee coeficientes negativos en la periferia y el centro positivo 
    de manera que todos los elementos sumen 0.
    """
    mascara = np.zeros((tamano_mascara, tamano_mascara), dtype=float)
    centro = tamano_mascara // 2
    
    # Se asignan coeficientes negativos (-1) en la periferia
    # y se calcula el valor central para que la suma sea 0.
    valor_periferia = -1.0
    suma_periferia = 0.0
    
    for y in range(tamano_mascara):
        for x in range(tamano_mascara):
            if y == centro and x == centro:
                continue
            mascara[y, x] = valor_periferia
            suma_periferia += valor_periferia
            
    # El centro debe compensar la suma de la periferia: Centro = - (Suma Periferia)
    mascara[centro, centro] = -suma_periferia
    
    return mascara

# --- FILTRO REALCE DE BORDES ---

def aplicar_filtro_realce_de_bordes(matriz_original, tamano_mascara):
    """
    Realce de Bordes con máscara de tamaño variable.
    Destaca detalles intensificando las transiciones de intensidad.
    """
    if tamano_mascara % 2 == 0:
        raise ValueError(f"El tamaño de la máscara debe ser impar.")

    # Generación de la máscara acorde al tamaño solicitado
    mascara = generar_mascara_realce_de_bordes(tamano_mascara)
    offset = tamano_mascara // 2
    alto, ancho = matriz_original.shape[:2]

    # Matriz de salida en float para precisión en la suma de productos
    resultado = np.zeros_like(matriz_original, dtype=np.float32)
    matriz_float = matriz_original.astype(np.float32)

    # Navegación manual por la imagen (Ventana deslizante)
    for y in range(offset, alto - offset):
        for x in range(offset, ancho - offset):
            
            if matriz_original.ndim == 3:  # Caso multicanal (RGB)
                for canal in range(3):
                    acumulado = 0.0
                    
                    # Convolución manual: suma de productos de la vecindad por pesos
                    for my in range(-offset, offset + 1):
                        for mx in range(-offset, offset + 1):
                            
                            # (y + my, x + mx) define la posición en la imagen
                            # (my + offset, mx + offset) define la posición en la máscara
                            pixel_valor = matriz_float[y + my, x + mx, canal]
                            peso = mascara[my + offset, mx + offset]
                            acumulado += pixel_valor * peso
                    resultado[y, x, canal] = acumulado
            else:  # Caso escala de grises (2D)
                acumulado = 0.0
                for my in range(-offset, offset + 1):
                    for mx in range(-offset, offset + 1):
                        pixel_valor = matriz_float[y + my, x + mx]
                        peso = mascara[my + offset, mx + offset]
                        acumulado += pixel_valor * peso
                resultado[y, x] = acumulado

    # Retorno con escalado para preservar la integridad de las transiciones detectadas
    return _escalar_a_8bit(resultado)