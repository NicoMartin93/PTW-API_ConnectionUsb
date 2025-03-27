import re
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_line(line):
    """
    Parsea una línea del archivo, extrayendo los datos de Time, Dose y Unit.
    Separa el valor numérico y la unidad para Time y asocia la unidad de Dose.
    """
    # Usamos una expresión regular para extraer los tres campos.
    pattern = r"Time:\s*([\d\.Ee+-]+)([a-zA-Z/]+),\s*Dose:\s*([\d\.Ee+-]+),\s*Unit:\s*(.+)"
    match = re.match(pattern, line)
    if match:
        time_val = float(match.group(1))
        time_unit = match.group(2)
        dose_val = float(match.group(3))
        dose_unit = match.group(4).strip()
        # Se crea un diccionario anidado para cada registro
        return {
            "Time": {"valor": time_val, "unidad": time_unit},
            "Dose": {"valor": dose_val, "unidad": dose_unit}
        }
    else:
        return None

def leer_archivo(nombre_archivo):
    """
    Lee el archivo de texto, omitiendo la línea de cabecera (sin ":"),
    y retorna una lista de diccionarios con los datos de cada línea.
    """
    datos = []
    with open(nombre_archivo, 'r') as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea:
                continue
            # Se asume que la cabecera no contiene ':' y se omite
            if ":" not in linea:
                continue
            registro = parse_line(linea)
            if registro:
                datos.append(registro)
    return datos

def dict_to_array(datos):
    """
    Convierte una lista de diccionarios en un array de NumPy.

    Cada registro del array contendrá:
        - Columna 0: valor numérico de 'Time'
        - Columna 1: valor numérico de 'Dose'

    Parámetros:
        datos (list): Lista de diccionarios con el formato:
            {
                "Time": {"valor": <float>, "unidad": <str>},
                "Dose": {"valor": <float>, "unidad": <str>}
            }

    Retorna:
        np.ndarray: Array de dimensiones (n, 2) con los valores de Time y Dose.
    """
    arr = np.array([[item["Time"]["valor"], item["Dose"]["valor"]] for item in datos])
    return arr

def plot_data(data):
    """
    Genera un gráfico de dosis en función del tiempo, aceptando tanto un array de NumPy como una lista de diccionarios.

    Parámetros:
        data (np.ndarray o list de dicts):
            - Si es un array de NumPy, se asume que tiene la forma (n,2) con columnas [Time, Dose].
            - Si es una lista de diccionarios, se espera que cada diccionario contenga:
                {"Time": {"valor": <float>, "unidad": <str>}, "Dose": {"valor": <float>, "unidad": <str>}}.
    """
    if isinstance(data, np.ndarray):
        # Caso: Array de NumPy
        times, doses = data[:, 0], data[:, 1]
        time_unit, dose_unit = "s", "Gy/min"  # Unidades por defecto si no se proporcionan

    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        # Caso: Lista de diccionarios
        times = [item["Time"]["valor"] for item in data]
        doses = [item["Dose"]["valor"] for item in data]
        time_unit = data[0]["Time"]["unidad"]
        dose_unit = data[0]["Dose"]["unidad"]
    else:
        raise ValueError("Formato de datos no reconocido. Debe ser un array de NumPy o una lista de diccionarios.")

    # Graficar
    plt.figure(figsize=(8, 6))
    plt.scatter(times, doses, marker='o', linestyle='-', color='b')
    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel(f"Kerma ({dose_unit})")
    plt.title("Kerma vs Time")
    plt.grid(True)
    plt.show()

def ajustar_tiempos(kerma_dic):
    """
    Ajusta los valores de 'Time' en el diccionario para que comiencen desde 0, basándose en el primer valor de tiempo.

    Parámetros:
        kerma_dic (list): Lista de diccionarios con claves 'Time' y 'Dose', donde 'Time' contiene
                          'valor' (el tiempo en segundos) y 'unidad' (la unidad de tiempo).

    Retorna:
        list: Lista de diccionarios con los tiempos ajustados, comenzando desde 0 segundos.
    """
    # Obtener el primer valor de tiempo para calcular el inicio desde 0
    primer_tiempo = kerma_dic[0]['Time']['valor']

    # Recorrer la lista de diccionarios y ajustar el tiempo
    for diccionario in kerma_dic:
        # Calcular la diferencia con el primer tiempo
        tiempo_ajustado = diccionario['Time']['valor'] - primer_tiempo

        # Actualizar el tiempo en el diccionario
        diccionario['Time']['valor'] = tiempo_ajustado

    return kerma_dic

def plot_time(datos, tiempo_max):
    """
    Grafica la dosis en función del tiempo referenciado desde 0 segundos.

    Parámetros:
        datos (list): Lista de diccionarios con el formato:
            {
                "Time": {"valor": <float>, "unidad": "s"},
                "Dose": {"valor": <float>, "unidad": <str>},
            }
        tiempo_max (float): Tiempo máximo en segundos hasta el cual graficar.

    Retorna:
        None: Muestra la gráfica.
    """
    # Filtrar los datos hasta el tiempo máximo especificado
    datos_filtrados = [d for d in datos if d["Time"]["valor"] <= tiempo_max]

    if not datos_filtrados:
        print("No hay datos en el rango especificado.")
        return

    # Extraer valores para graficar
    tiempos = [d["Time"]["valor"] for d in datos_filtrados]
    dosis = [d["Dose"]["valor"] for d in datos_filtrados]
    unidad_dosis = datos_filtrados[0]["Dose"]["unidad"] if datos_filtrados else "Desconocido"

    # Crear el gráfico
    plt.figure(figsize=(8, 5))
    plt.scatter(tiempos, dosis, marker='o', linestyle='-', color='b', label="Kerma")

    # Etiquetas y título
    plt.xlabel("Tiempo (s)")
    plt.ylabel(f"Kerma ({unidad_dosis})")
    plt.title(f"Kerma vs Tiempo (hasta {tiempo_max} s)")
    plt.legend()
    plt.grid(True)

    # Mostrar gráfico
    plt.show()

def leer_parametros_txt(nombre_parameters):
    """
    Lee un archivo .txt con columnas de datos en formato tabulado:
    Time[s], Temperatura[C], Humedad [%], Presion [hPa]

    Parámetros:
        nombre_parameters (str): Ruta del archivo .txt.

    Retorna:
        dict: Diccionario con los datos estructurados.
    """
    datos = {
        "Time": {"valores": [], "unidad": "s"},
        "Temperatura": {"valores": [], "unidad": "C"},
        "Humedad": {"valores": [], "unidad": "%"},
        "Presion": {"valores": [], "unidad": "hPa"}
    }

    with open(nombre_parameters, 'r') as archivo:
        lineas = archivo.readlines()

        # Omitir la primera línea (encabezado)
        for linea in lineas[1:]:
            valores = linea.strip().split("\t")

            if len(valores) != 4:
                continue  # Ignorar líneas mal formateadas

            # Agregar los valores a las listas del diccionario
            datos["Time"]["valores"].append(float(valores[0]))
            datos["Temperatura"]["valores"].append(float(valores[1]))
            datos["Humedad"]["valores"].append(float(valores[2]))
            datos["Presion"]["valores"].append(float(valores[3]))

    return datos

def interpolar_datos(datos, num_puntos):
    """
    Interpola los valores de Temperatura y Presión a un número específico de puntos.

    Parámetros:
        datos (dict): Diccionario con el formato:
            {
                "Time": {"valores": [t1, t2, ..., tn], "unidad": "s"},
                "Temperatura": {"valores": [T1, T2, ..., Tm], "unidad": "C"},
                "Humedad": {"valores": [H1, H2, ..., Hk], "unidad": "%"},
                "Presion": {"valores": [P1, P2, ..., Pl], "unidad": "hPa"}
            }
        num_puntos (int): Número de puntos al que se desea interpolar los datos.

    Retorna:
        dict: Diccionario con los valores interpolados de Tiempo, Temperatura y Presión.
    """
    if num_puntos < 2:
        raise ValueError("El número de puntos debe ser al menos 2.")

    # Vector de tiempo original y vector de tiempo interpolado
    tiempo_original = np.array(datos["Time"]["valores"])
    tiempo_interp = np.linspace(tiempo_original[0], tiempo_original[-1], num_puntos)

    # Interpolación de Temperatura y Presión
    temperatura_interp = np.interp(tiempo_interp, tiempo_original, datos["Temperatura"]["valores"])
    presion_interp = np.interp(tiempo_interp, tiempo_original, datos["Presion"]["valores"])

    # Guardar los datos interpolados en un nuevo diccionario
    datos_interpolados = {
        "Time": {"valores": tiempo_interp.tolist(), "unidad": "s"},
        "Temperatura": {"valores": temperatura_interp.tolist(), "unidad": "C"},
        "Presion": {"valores": presion_interp.tolist(), "unidad": "hPa"}
    }

    return datos_interpolados

def corregir_kerma(params, datos, P_ref=1013, T_ref=293.15):
    """
    Corrige los valores de kerma según la presión y temperatura interpoladas usando la ecuación del gas ideal.

    Parámetros:
        datos (dict): Diccionario con los datos interpolados de presión y temperatura.
                      Debe incluir las claves "Presion" y "Temperatura".
        kerma (dict/array): Diccionario o numpy array con los datos de kerma medido.
                           En caso de diccionario, debe incluir las claves "Time" y "Dose".
        P_ref (float): Presión de referencia en hPa (default = 1013 hPa).
        T_ref (float): Temperatura de referencia en K (default = 293.15 K).

    Retorna:
        dict: Diccionario con los mismos datos que `kerma_dic`, pero con los valores de "Dose" corregidos.
    """
    # Extraer datos de presión y temperatura interpolados
    if isinstance(datos, np.ndarray):
        # Caso: Array de NumPy
        if len(datos.shape) != 1:
            times, kerma = datos[:, 0], datos[:, 1]
            time_unit, kerma_unit = "s", "Gy/min"  # Unidades por defecto si no se proporcionan
        else:
            kerma = datos[:]

    elif isinstance(datos, list) and all(isinstance(item, dict) for item in datos):
        # Caso: Lista de diccionarios
        times = [item["Time"]["valor"] for item in datos]
        kerma = [item["Dose"]["valor"] for item in datos]
        time_unit = datos[0]["Time"]["unidad"]
        dose_unit = datos[0]["Dose"]["unidad"]
    else:
        raise ValueError("Formato de datos no reconocido. Debe ser un array de NumPy o una lista de diccionarios.")

    P = np.array(params["Presion"]["valores"])  # Presión en hPa
    T_Celsius = np.array(params["Temperatura"]["valores"])  # Temperatura en °C

    # Convertir Temperatura a Kelvin
    T = T_Celsius + 273.15

    kerma_corr = []
    for i, dato in enumerate(kerma):
        # Aplicar la ecuación del gas ideal
        kerma_corregido = dato * (P_ref / P[i]) * (T[i] / T_ref)
        kerma_corr.append(kerma_corregido)
    return kerma_corr


def plot_presion_temperatura(datos):
    """
    Grafica la variación de la presión y la temperatura con el tiempo en el mismo gráfico.

    Parámetros:
        datos (dict): Diccionario con claves 'Time', 'Temperatura' y 'Presion', donde cada una contiene
                      una lista de valores y sus respectivas unidades.

    Retorna:
        None (Muestra el gráfico).
    """
    tiempo = datos['Time']['valores']
    temperatura = datos['Temperatura']['valores']
    presion = datos['Presion']['valores']

    unidad_tiempo = datos['Time']['unidad']
    unidad_temp = datos['Temperatura']['unidad']
    unidad_presion = datos['Presion']['unidad']

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Primer eje Y - Temperatura
    ax1.set_xlabel(f"Tiempo ({unidad_tiempo})")
    ax1.set_ylabel(f"Temperatura ({unidad_temp})", color='tab:red')
    ax1.plot(tiempo, temperatura, color='tab:red', marker='o', linestyle='-', label="Temperatura")
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Segundo eje Y - Presión
    ax2 = ax1.twinx()
    ax2.set_ylabel(f"Presión ({unidad_presion})", color='tab:blue')
    ax2.plot(tiempo, presion, color='tab:blue', marker='s', linestyle='--', label="Presión")
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Título y leyenda
    plt.title("Variación de la Presión y la Temperatura con el Tiempo")
    fig.tight_layout()
    plt.show()

# FUNCIONES PARA GRUPOS DE DATOS

def cargarDatos(pathfolder, nombre_archivos, nombre_parametros, interval=20, max_time=3555):

    num_mediciones = len(nombre_archivos)

    datos_corregidos = []
    for i in range(num_mediciones):
        # Leer archivos
        pathfile_data = os.path.join(pathfolder, nombre_archivos[i])
        pathfile_param = os.path.join(pathfolder, nombre_parametros[i])

        datos = leer_archivo(pathfile_data)
        params = leer_parametros_txt(pathfile_param)

        # Determinar paso de tiempo
        time_step = datos[1]["Time"]["valor"] - datos[0]["Time"]["valor"]
        num_points_per_group = int(interval // time_step)
        tiempo_max = int(max_time / time_step)

        # Filtrar hasta tiempo máximo y extraer dosis
        datos = np.array([d["Dose"]["valor"] for j, d in enumerate(datos) if j <= tiempo_max])

        # Agrupar los datos
        num_groups = len(datos) // num_points_per_group
        time = np.arange(interval, len(datos) * time_step, interval)

        group_avg = [np.mean(datos[j * num_points_per_group:(j + 1) * num_points_per_group]) for j in range(num_groups)]
        group_std = [np.std(datos[j * num_points_per_group:(j + 1) * num_points_per_group]) for j in range(num_groups)]

        # Interpolar parámetros y corregir el kerma
        params_interp = interpolar_datos(params, len(group_avg))
        kerma_corregido = corregir_kerma(params_interp, np.array(group_avg))
        datos_corregidos.append(kerma_corregido)

    datos_corregidos = np.array(datos_corregidos)
    return datos_corregidos

def procesar_y_graficar_kerma(pathfolder, nombre_archivos, nombre_parametros, interval=20, max_time=3555, colores=None, labels=None):
    """
    Procesa archivos de datos de dosis y parámetros, y grafica el kerma crudo y corregido con barras de error.

    Parámetros:
        - pathfolder: Ruta a la carpeta donde se encuentran los archivos.
        - nombre_archivos: Lista con los nombres de los archivos de datos.
        - nombre_parametros: Lista con los nombres de los archivos de parámetros.
        - interval: Intervalo de agrupación en segundos (default: 20).
        - max_time: Tiempo máximo de análisis en segundos (default: 3555 s).
        - colores: Lista de colores opcional para cada medición.
        - labels: Lista de etiquetas opcional para cada medición.
    """
    num_mediciones = len(nombre_archivos)
    if colores is None:
        colores = ['b', 'g', 'r'][:num_mediciones]
    if labels is None:
        labels = [f'{i+1}° Medición' for i in range(num_mediciones)]

    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    for i in range(num_mediciones):
        # Leer archivos
        pathfile_data = os.path.join(pathfolder, nombre_archivos[i])
        pathfile_param = os.path.join(pathfolder, nombre_parametros[i])

        datos = leer_archivo(pathfile_data)
        params = leer_parametros_txt(pathfile_param)

        # Determinar paso de tiempo
        time_step = datos[1]["Time"]["valor"] - datos[0]["Time"]["valor"]
        num_points_per_group = int(interval // time_step)
        tiempo_max = int(max_time / time_step)

        # Filtrar hasta tiempo máximo y extraer dosis
        datos = np.array([d["Dose"]["valor"] for j, d in enumerate(datos) if j <= tiempo_max])

        # Agrupar los datos
        num_groups = len(datos) // num_points_per_group
        time = np.arange(interval, len(datos) * time_step, interval)

        group_avg = [np.mean(datos[j * num_points_per_group:(j + 1) * num_points_per_group]) for j in range(num_groups)]
        group_std = [np.std(datos[j * num_points_per_group:(j + 1) * num_points_per_group]) for j in range(num_groups)]

        # Interpolar parámetros y corregir el kerma
        params_interp = interpolar_datos(params, len(group_avg))
        kerma_corregido = corregir_kerma(params_interp, np.array(group_avg))

        # Graficar con barras de error
        axs[0].errorbar(time, group_avg, yerr=group_std, fmt='o', capsize=3, label=labels[i], color=colores[i])
        axs[1].scatter(time, kerma_corregido, label=labels[i], color=colores[i])

    # Configuración de gráficos
    axs[0].set_ylabel("Kerma crudo [Gy/min]")
    axs[0].set_title(f"Promedio - {interval} seg")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_xlabel("Tiempo [s]")
    axs[1].set_ylabel("Kerma corregido [Gy/min]")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def procesar_y_graficar_kerma_mediana(pathfolder, nombre_archivos, nombre_parametros, interval=20, max_time=3555, colores=None, labels=None):
    """
    Procesa archivos de datos de dosis y parámetros, y grafica el kerma crudo (mediana) y corregido con barras de error.

    Parámetros:
        - pathfolder: Ruta a la carpeta donde se encuentran los archivos.
        - nombre_archivos: Lista con los nombres de los archivos de datos.
        - nombre_parametros: Lista con los nombres de los archivos de parámetros.
        - interval: Intervalo de agrupación en segundos (default: 20).
        - max_time: Tiempo máximo de análisis en segundos (default: 3555 s).
        - colores: Lista de colores opcional para cada medición.
        - labels: Lista de etiquetas opcional para cada medición.
    """
    num_mediciones = len(nombre_archivos)
    if colores is None:
        colores = ['b', 'g', 'r'][:num_mediciones]
    if labels is None:
        labels = [f'{i+1}° Medición' for i in range(num_mediciones)]

    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    for i in range(num_mediciones):
        # Leer archivos
        pathfile_data = os.path.join(pathfolder, nombre_archivos[i])
        pathfile_param = os.path.join(pathfolder, nombre_parametros[i])

        datos = leer_archivo(pathfile_data)
        params = leer_parametros_txt(pathfile_param)

        # Determinar paso de tiempo
        time_step = datos[1]["Time"]["valor"] - datos[0]["Time"]["valor"]
        num_points_per_group = int(interval // time_step)
        tiempo_max = int(max_time / time_step)

        # Filtrar hasta tiempo máximo y extraer dosis
        datos = np.array([d["Dose"]["valor"] for j, d in enumerate(datos) if j <= tiempo_max])

        # Agrupar los datos
        num_groups = len(datos) // num_points_per_group
        time = np.arange(interval, len(datos) * time_step, interval)

        group_median = [np.median(datos[j * num_points_per_group:(j + 1) * num_points_per_group]) for j in range(num_groups)]
        group_std = [np.std(datos[j * num_points_per_group:(j + 1) * num_points_per_group]) for j in range(num_groups)]

        # Interpolar parámetros y corregir el kerma
        params_interp = interpolar_datos(params, len(group_median))
        kerma_corregido = corregir_kerma(params_interp, np.array(group_median))

        # Graficar con barras de error
        axs[0].errorbar(time, group_median, yerr=group_std, fmt='o', capsize=3, label=labels[i], color=colores[i])
        axs[1].scatter(time, kerma_corregido, label=labels[i], color=colores[i])

    # Configuración de gráficos
    axs[0].set_ylabel("Kerma crudo [Gy/min]")
    axs[0].set_title(f"Mediana - {interval} seg")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_xlabel("Tiempo [s]")
    axs[1].set_ylabel("Kerma corregido [Gy/min]")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def graficar_datos_kerma(pathfolder, nombre_archivos, nombre_parametros, interval=20, max_time=3555, colores=None, labels=None):
    """
    Grafica los datos de kerma crudo y corregido con bandas de desviación estándar.

    Parámetros:
        - datos_mediciones: Lista de listas con datos de dosis por medición.
        - parametros_mediciones: Lista de listas con parámetros de cada medición.
        - interval: Intervalo de agrupación en segundos (default: 20).
        - time_step: Paso de tiempo entre mediciones (default: 2 segundos).
        - labels: Lista de etiquetas para cada medición.
        - colores: Lista de colores para cada medición.
    """

    num_mediciones = len(nombre_archivos)
    if colores is None:
        colores = ['b', 'g', 'r'][:num_mediciones]
    if labels is None:
        labels = [f'{i+1}° Medición' for i in range(num_mediciones)]

    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    for i in range(num_mediciones):
        # Leer archivos
        pathfile_data = os.path.join(pathfolder, nombre_archivos[i])
        pathfile_param = os.path.join(pathfolder, nombre_parametros[i])

        datos = leer_archivo(pathfile_data)
        params = leer_parametros_txt(pathfile_param)

        # Determinar paso de tiempo
        time_step = datos[1]["Time"]["valor"] - datos[0]["Time"]["valor"]
        num_points_per_group = int(interval // time_step)
        tiempo_max = int(max_time / time_step)

        # Filtrar hasta tiempo máximo y extraer dosis
        datos = np.array([d["Dose"]["valor"] for j, d in enumerate(datos) if j <= tiempo_max])

        # Agrupar los datos
        num_groups = len(datos) // num_points_per_group
        time = np.arange(interval, len(datos) * time_step, interval)

        group_avg = [np.mean(datos[j * num_points_per_group:(j + 1) * num_points_per_group]) for j in range(num_groups)]
        group_std = [np.std(datos[j * num_points_per_group:(j + 1) * num_points_per_group]) for j in range(num_groups)]

        # Interpolar parámetros y corregir el kerma
        params_interp = interpolar_datos(params, len(group_avg))
        kerma_corregido = corregir_kerma(params_interp, np.array(group_avg))

        # Graficar con banda de desviación estándar
        axs[0].plot(time, group_avg, label=labels[i], color=colores[i])
        axs[0].fill_between(time, np.array(group_avg) - np.array(group_std),
                            np.array(group_avg) + np.array(group_std), color=colores[i], alpha=0.2)

        axs[1].plot(time, kerma_corregido, label=labels[i], color=colores[i])
        axs[1].fill_between(time, np.array(kerma_corregido) - np.array(group_std),
                            np.array(kerma_corregido) + np.array(group_std), color=colores[i], alpha=0.2)

    axs[0].set_ylabel("Kerma crudo [Gy/min]")
    axs[0].set_title(f"Promedio por Medición - {interval} seg")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_xlabel("Tiempo [s]")
    axs[1].set_ylabel("Kerma corregido [Gy/min]")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # ------------------------ #
    # (1) LECTURA INDIVIDUAL   #
    # ------------------------ #
    # Definir rutas de archivos
    pathfolder = r"D:\Proyectos_Investigacion\Proyectos_de_Doctorado\Proyectos\XFCT_Experimental\Resultados_VariacionFuente\VarFuente_Tiempo"
    nombre_archivo = 'doseRate-1hora_18-03-25_V150-5mA_1.txt'
    pathfile_data = os.path.join(pathfolder, nombre_archivo)

    # Leer y procesar datos
    datos = leer_archivo(pathfile_data)
    time_step = datos[1]["Time"]["valor"] - datos[0]["Time"]["valor"]
    tiempo_max = int(3555/time_step)
    datos = np.array([d["Dose"]["valor"] for i,d in enumerate(datos) if i <= tiempo_max])

    # Parámetros de agrupación
    interval = 20
    time_step = 2
    num_points_per_group = interval // time_step
    num_groups = len(datos) // num_points_per_group
    time = np.arange(interval,len(datos)*time_step, interval)


    # Calcular promedios y desviaciones
    group_avg = [np.mean(datos[i * num_points_per_group:(i + 1) * num_points_per_group]) for i in range(num_groups)]
    group_std = [np.std(datos[i * num_points_per_group:(i + 1) * num_points_per_group]) for i in range(num_groups)]
    group_avg = np.array(group_avg)
    group_std = np.array(group_std)

    # Leer parámetros e interpolar
    nombre_parameters = r'Parameters_18-03-25_2.txt'
    pathfile_param = os.path.join(pathfolder, nombre_parameters)
    params = leer_parametros_txt(pathfile_param)
    params_interp = interpolar_datos(params, len(group_avg))
    plot_presion_temperatura(params)

    # Corregir el kerma
    kerma_corregido = corregir_kerma(params_interp, group_avg)

    # Crear figura con dos gráficos
    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Gráfico 1: Promedio con barras de error
    axs[0].errorbar(time, group_avg, yerr=group_std, fmt='o', capsize=3, label="Promedio cada 20s")
    axs[0].set_ylabel("Medición promedio")
    axs[0].set_title("Promedio cada 20 segundos con barra de error")
    axs[0].legend()
    axs[0].grid(True)

    # Gráfico 2: Kerma corregido
    axs[1].scatter(time, kerma_corregido, label="Kerma corregido", color='r')
    axs[1].set_xlabel("Tiempo [s]")
    axs[1].set_ylabel("Kerma [Gy/min]")
    axs[1].legend()
    axs[1].grid(True)

    # Mostrar figura
    plt.tight_layout()
    plt.show()

    # --------------------------- #
    # (2) VISTA GRUPAL DE DATOS   #
    # --------------------------- #
    pathfolder = r"D:\Proyectos_Investigacion\Proyectos_de_Doctorado\Proyectos\XFCT_Experimental\Resultados_VariacionFuente\VarFuente_Tiempo"
    # nombre_archivos = ['doseRate-1hora_18-03-25_V150-5mA_1.txt',
    #                    'doseRate-1hora_18-03-25_V150-5mA_2.txt',
    #                    'doseRate-1hora_18-03-25_V150-5mA_3.txt']
    #
    # nombre_parametros = ['Parameters_18-03-25_1.txt',
    #                      'Parameters_18-03-25_2.txt',
    #                      'Parameters_18-03-25_3.txt']

    nombre_archivos = ['doseRate-1hora_25-03-25_V150-5mA_1.txt',
                       'doseRate-1hora_25-03-25_V150-5mA_2.txt',
                       'doseRate-1hora_25-03-25_V150-5mA_3.txt']

    nombre_parametros = ['Parameters_25-03-25_1.txt',
                         'Parameters_25-03-25_2.txt',
                         'Parameters_25-03-25_3.txt']

    # nombre_archivos = ['doseRate-1hora_26-03-25_V150-5mA_1.txt',
    #                    'doseRate-1hora_26-03-25_V150-5mA_2.txt',
    #                    'doseRate-1hora_26-03-25_V150-5mA_3.txt']
    #
    # nombre_parametros = ['Parameters_26-03-25_1.txt',
    #                      'Parameters_26-03-25_2.txt',
    #                      'Parameters_26-03-25_3.txt']

    # Parámetros de agrupación
    time_step = 2
    interval = 300
    datos = cargarDatos(pathfolder, nombre_archivos, nombre_parametros, interval=interval, max_time=3555)



    procesar_y_graficar_kerma(pathfolder, nombre_archivos, nombre_parametros, interval=interval, max_time=3555, colores=None, labels=None)
    procesar_y_graficar_kerma_mediana(pathfolder, nombre_archivos, nombre_parametros, interval=20, max_time=3555, colores=None, labels=None)
    graficar_datos_kerma(pathfolder, nombre_archivos, nombre_parametros, interval=interval, max_time=3555, colores=None, labels=None)

    from scipy.stats import pearsonr


    def analizar_estabilidad_y_correlacion(datos, labels, ventana_mov=30):
        """
        Analiza la estabilidad temporal de las mediciones y calcula la correlación entre ellas.

        Parámetros:
        - datos: lista de arrays con las tres mediciones
        - labels: lista de nombres para cada medición
        - ventana_mov: tamaño de la ventana para el promedio móvil (en número de puntos)
        """
        num_mediciones = len(datos)
        colores = ['b', 'g', 'r']
        tiempo = np.arange(len(datos[0]))*2

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # --- Estabilidad Temporal ---
        for i in range(num_mediciones):
            media_movil = np.convolve(datos[i], np.ones(ventana_mov)/ventana_mov, mode='valid')
            std_movil = np.array([np.std(datos[i][max(0, j-ventana_mov+1):j+1]) for j in range(ventana_mov-1, len(datos[i]))])
            t = tiempo[ventana_mov:ventana_mov+len(media_movil)].shape[0]
            media_movil = media_movil[:len(std_movil)]
            axs[0].plot(
                tiempo[ventana_mov:ventana_mov+len(media_movil)+4],
                media_movil[:t],
                label=f"{labels[i]} - Media Móvil",
                color=colores[i]
            )
            minMS = media_movil - std_movil
            maxMS = media_movil + std_movil
            axs[0].fill_between(
                tiempo[ventana_mov:ventana_mov+len(media_movil)],
                minMS[:t],
                maxMS[:t],
                color=colores[i], alpha=0.2
            )
        axs[0].set_title("Estabilidad Temporal de la Medición")
        axs[0].set_ylabel("Kerma [Gy/min]")
        axs[0].legend()
        axs[0].grid()

        # --- Correlación entre Mediciones ---
        matriz_corr = np.corrcoef(datos)
        sns.heatmap(matriz_corr, annot=True, xticklabels=labels, yticklabels=labels, cmap="coolwarm", ax=axs[1])
        axs[1].set_title("Matriz de Correlación entre Mediciones")

        plt.xlabel("Tiempo [s]")
        plt.tight_layout()
        plt.show()

        # Mostrar coeficientes de correlación de Pearson
        print("Coeficientes de correlación de Pearson entre mediciones:")
        for i in range(num_mediciones):
            for j in range(i + 1, num_mediciones):
                corr, _ = pearsonr(datos[i], datos[j])
                print(f"{labels[i]} vs {labels[j]}: {corr:.3f}")


    num_mediciones = len(nombre_archivos)
    labels = [f'{i+1}° Medición' for i in range(num_mediciones)]


    # datos_SA = detectar_eventos_anomalos(datos, labels, z_score_threshold=6, umbral_adaptativo=2.0)

    analizar_estabilidad_y_correlacion(datos, labels, ventana_mov=30)

def comparar_con_referencia(datos, labels):
    """
    Compara las mediciones con un valor de referencia global (promedio total).

    Parámetros:
    - datos: lista de arrays, cada uno con las mediciones de una sesión.
    - labels: etiquetas para cada conjunto de datos.
    """
    promedio_global = np.mean(np.concatenate(datos))  # Promedio de todas las mediciones combinadas

    plt.figure(figsize=(10, 5))

    for i in range(len(datos)):
        plt.hist(datos[i], bins=30, alpha=0.5, label=f"{labels[i]}")

    plt.axvline(promedio_global, color='red', linestyle='dashed', linewidth=2, label="Promedio Global")

    plt.xlabel("Valor de Medición")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de mediciones y comparación con referencia")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Promedio global de todas las mediciones: {promedio_global:.4f}")

from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

def detectar_picos(datos, labels, umbral_relativo=3):
    """
    Detecta picos en las mediciones para identificar eventos anómalos.

    Parámetros:
    - datos: lista de arrays con las mediciones.
    - labels: etiquetas de cada medición.
    - umbral_relativo: múltiplo de la desviación estándar para considerar un pico.
    """
    plt.figure(figsize=(10, 5))

    for i in range(len(datos)):
        media = np.mean(datos[i])
        std = np.std(datos[i])
        umbral = media + umbral_relativo * std  # Definir umbral de detección

        picos, _ = find_peaks(datos[i], height=umbral)

        tiempo = np.arange(len(datos[i]))

        plt.plot(tiempo, datos[i], label=f"{labels[i]}")
        plt.scatter(tiempo[picos], datos[i][picos], color='red', marker='o', label=f"Picos {labels[i]}")

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Medición")
    plt.title("Detección de Picos en la Señal")
    plt.legend()
    plt.grid(True)
    plt.show()

def analizar_frecuencia(datos, labels, intervalo_muestreo=2):
    """
    Aplica Transformada de Fourier (FFT) a los datos para detectar ruido periódico.

    Parámetros:
    - datos: lista de arrays con las mediciones.
    - labels: etiquetas de cada medición.
    - intervalo_muestreo: tiempo entre mediciones en segundos.
    """
    plt.figure(figsize=(10, 5))

    for i in range(len(datos)):
        N = len(datos[i])  # Número de puntos
        frecuencias = fftfreq(N, intervalo_muestreo)[:N//2]  # Frecuencias positivas
        espectro = np.abs(fft(datos[i]))[:N//2]  # Magnitud de la FFT

        plt.plot(frecuencias, espectro, label=f"{labels[i]}")

    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
    plt.title("Análisis de Frecuencia (FFT)")
    plt.legend()
    plt.grid(True)
    plt.show()
