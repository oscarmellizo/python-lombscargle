import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from scipy.signal import argrelmin
import re
from matplotlib.widgets import Slider, TextBox

# Cargar datos de la curva de luz
data_file = "asm_gx304-1_lc_b1.dat"
data = np.loadtxt(data_file, usecols=(0, 2))
fechas_observacion, tasas_cuentas = data[:, 0], data[:, 1]

# Definir los límites iniciales de MJD y frecuencia para los sliders
mjd_min, mjd_max = fechas_observacion.min(), fechas_observacion.max()
frecuencia_min, frecuencia_max = 0.001, 0.5

# Calcular una potencia máxima más representativa
analisis_lomb_scargle = LombScargle(fechas_observacion, tasas_cuentas)
potencia_ejemplo = analisis_lomb_scargle.power(np.linspace(frecuencia_min, frecuencia_max, 10000))

# Crear figura y ejes para la curva de luz, el periodograma y los resultados
fig, (ax_luz, ax_periodo) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.4)

# Configurar la curva de luz
ax_luz.set_xlabel("MJD")
ax_luz.set_ylabel("Count Rate (counts/s)")
ax_luz.set_title("Curva de Luz")
linea, = ax_luz.plot(fechas_observacion, tasas_cuentas, label="Datos originales")
ax_luz.legend()

# Configurar el periodograma
ax_periodo.set_xlabel("Frecuencia (día⁻¹)")
ax_periodo.set_ylabel("Potencia")
ax_periodo.set_title("Periodograma")
linea_periodo, = ax_periodo.plot([], [], label="Potencia Lomb-Scargle")
ax_periodo.legend()
ax_periodo.relim()
ax_periodo.autoscale_view()
plt.draw()

# Crear sliders debajo de la curva de luz
ax_mjd_min = plt.axes([0.1, 0.25, 0.35, 0.03])
ax_mjd_max = plt.axes([0.1, 0.20, 0.35, 0.03])
ax_fap_limit = plt.axes([0.1, 0.15, 0.35, 0.03])

slider_mjd_min = Slider(ax_mjd_min, "MJD Min", mjd_min, mjd_max, valinit=mjd_min)
slider_mjd_max = Slider(ax_mjd_max, "MJD Max", mjd_min, mjd_max, valinit=mjd_max)
slider_fap_limit = Slider(ax_fap_limit, "FAP Limit", 0, 1, valinit=0.1)

# Crear sliders debajo del periodograma
ax_freq_min = plt.axes([0.55, 0.20, 0.35, 0.03])
ax_freq_max = plt.axes([0.55, 0.15, 0.35, 0.03])

slider_freq_min = Slider(ax_freq_min, "Freq Min", frecuencia_min, frecuencia_max, valinit=frecuencia_min)
slider_freq_max = Slider(ax_freq_max, "Freq Max", frecuencia_min, frecuencia_max, valinit=frecuencia_max)

# Crear área para mostrar los resultados en una cuadrícula 2x2
ax_resultados = plt.axes([0.25, 0.01, 0.5, 0.05])
text_resultados = TextBox(ax_resultados, "")
text_resultados.set_val("Frecuencia: -- | Relación S/N: -- | Periodo: -- | Error: --")

# Función para actualizar las gráficas
def actualizar_graficas(val):
    rango_mjd_min = slider_mjd_min.val
    rango_mjd_max = slider_mjd_max.val
    freq_min = slider_freq_min.val
    freq_max = slider_freq_max.val
    fap_limite = slider_fap_limit.val
    
    mascara_mjd = (fechas_observacion >= rango_mjd_min) & (fechas_observacion <= rango_mjd_max)
    fechas_filtradas, tasas_filtradas = fechas_observacion[mascara_mjd], tasas_cuentas[mascara_mjd]
    
    # Actualizar curva de luz
    linea.set_xdata(fechas_filtradas)
    linea.set_ydata(tasas_filtradas)
    ax_luz.set_xlim(rango_mjd_min, rango_mjd_max)
    ax_luz.relim()
    ax_luz.autoscale_view()
    
    # Calcular el periodograma solo si hay suficientes datos
    if len(fechas_filtradas) > 2:
        frecuencias_totales = np.linspace(frecuencia_min, frecuencia_max, 10000)
        mascara_frec = (frecuencias_totales >= freq_min) & (frecuencias_totales <= freq_max)
        frecuencias = frecuencias_totales[mascara_frec]
        potencia_periodograma = analisis_lomb_scargle.power(frecuencias)
        mascara_frec_calc = (frecuencias >= slider_freq_min.val) & (frecuencias <= slider_freq_max.val)
        frecuencias_filtradas = frecuencias[mascara_frec_calc]
        potencia_filtrada = potencia_periodograma[mascara_frec_calc]
        frecuencia_maxima = frecuencias_filtradas[np.argmax(potencia_filtrada)]
        
        # Calcular la probabilidad de falsa alarma
        probabilidad_falsa_alarma = analisis_lomb_scargle.false_alarm_probability(potencia_periodograma.max(), method="baluev")
        
        if probabilidad_falsa_alarma < fap_limite:
            # Calcular resultados
            periodo_estimado = 1 / frecuencia_maxima
            ancho_pico = np.abs(frecuencias_filtradas[np.argmax(potencia_filtrada)] - frecuencias_filtradas[np.argmin(np.abs(potencia_filtrada - potencia_filtrada.max() / 2))])
            error_periodo = ancho_pico / (frecuencia_maxima ** 2)  # Suponiendo un 1% de error
            relacion_sn = potencia_periodograma.max() / np.mean(potencia_periodograma)
            
            # Actualizar el periodograma
            linea_periodo.set_xdata(frecuencias_filtradas)
            linea_periodo.set_ydata(potencia_filtrada)
            ax_periodo.set_xlim(slider_freq_min.val, slider_freq_max.val)
            ax_periodo.relim()
            ax_periodo.autoscale_view()
            
            # Marcar la frecuencia detectada en el periodograma
            for line in ax_periodo.lines[:]:
                if line.get_linestyle() == '--':
                    line.remove()
            error_frecuencia = ancho_pico
            ax_periodo.axvline(frecuencia_maxima, color='r', linestyle='--', label=f'Freq: {frecuencia_maxima:.6f} ± {error_frecuencia:.6f}')
            ax_periodo.legend()
            ax_periodo.relim()
            ax_periodo.autoscale_view()
            plt.draw()
            ax_periodo.legend()
            ax_periodo.relim()
            ax_periodo.autoscale_view()
            ax_periodo.legend()
            
            text_resultados.set_val(f"Frecuencia: {frecuencia_maxima:.6f} | Relación S/N: {relacion_sn:.2f} | Periodo: {periodo_estimado:.6f} | Error: {error_periodo:.6f}")
        else:
            text_resultados.set_val("FAP demasiado alta, no se puede calcular el periodo")
    
    plt.draw()

# Conectar los sliders a la función de actualización
slider_mjd_min.on_changed(actualizar_graficas)
slider_mjd_max.on_changed(actualizar_graficas)
slider_fap_limit.on_changed(actualizar_graficas)
slider_freq_min.on_changed(actualizar_graficas)
slider_freq_max.on_changed(actualizar_graficas)

plt.show()
