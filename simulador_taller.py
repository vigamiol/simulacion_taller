import numpy as np
import heapq
from collections import deque, defaultdict

# --- Parámetros del sistema (puedes modificar para análisis de sensibilidad) ---
LAMBDA_LLEGADAS_BASE = 2/60  # vehículos por minuto (base)
REV_PREVIA_MEDIA = 15
REP_LIGERA_MEDIA, REP_LIGERA_STD = 30, 5
REP_MEDIA_MEDIA, REP_MEDIA_STD = 60, 10
REP_COMPLEJA_MIN, REP_COMPLEJA_MODE, REP_COMPLEJA_MAX = 120, 280, 480
CHEQUEO_MEDIA, CHEQUEO_STD = 10, 2
P_FALLO_CALIDAD_BASE = 0.10 # Probabilidad de fallo de calidad (base)

# --- Configuración de simulación ---
TIEMPO_MAX = 2920 * 60  # 2920 horas en minutos (equivalente a 1 año laboral, aprox)
NUM_REPLICAS = 30       # Número de veces que se ejecutará cada escenario para análisis estadístico

# NÚMERO DE SERVIDORES POR FASE (configuración base)
N_SERVIDORES_REV_BASE = 2
N_SERVIDORES_REP_BASE = 5
N_SERVIDORES_CHEQ_BASE = 1

# --- Definición de eventos ---
EVENTO_LLEGADA = 'llegada'
EVENTO_FIN_REV = 'fin_revision'
EVENTO_FIN_REP = 'fin_reparacion'
EVENTO_FIN_CHEQUEO = 'fin_chequeo'
EVENTO_SALIDA = 'salida'

# --- Fases de reparación ---
TIPO_REP_LIGERA = 'ligera'
TIPO_REP_MEDIA = 'media'
TIPO_REP_COMPLEJA = 'compleja'
TIPOS_REP = [TIPO_REP_LIGERA, TIPO_REP_MEDIA, TIPO_REP_COMPLEJA]
PROB_REP = [0.5, 0.3, 0.2]  # Probabilidad de cada tipo de reparación

# --- Generadores de variables aleatorias ---
def tiempo_llegada(lam_llegadas):
    return np.random.exponential(1 / lam_llegadas)

def tiempo_revision():
    return np.random.exponential(REV_PREVIA_MEDIA)

def tipo_reparacion():
    return np.random.choice(TIPOS_REP, p=PROB_REP)

def tiempo_reparacion(tipo):
    if tipo == TIPO_REP_LIGERA:
        return np.random.normal(REP_LIGERA_MEDIA, REP_LIGERA_STD)
    elif tipo == TIPO_REP_MEDIA:
        return np.random.normal(REP_MEDIA_MEDIA, REP_MEDIA_STD)
    else:
        # Triangular manual
        return np.random.triangular(REP_COMPLEJA_MIN, REP_COMPLEJA_MODE, REP_COMPLEJA_MAX)

def tiempo_chequeo():
    return np.random.normal(CHEQUEO_MEDIA, CHEQUEO_STD)

def pasa_calidad(p_fallo):
    return np.random.rand() > p_fallo

# --- Estructura de evento ---
class Evento:
    def __init__(self, tiempo, contador, tipo, datos):
        self.tiempo = tiempo
        self.contador = contador
        self.tipo = tipo
        self.datos = datos
    def __lt__(self, other):
        return (self.tiempo, self.contador) < (other.tiempo, self.contador)

# --- Simulación principal ---
def simular(tiempo_max,
            lam_llegadas,
            p_fallo_calidad,
            n_servidores_rev,
            n_servidores_rep,
            n_servidores_cheq,
            semilla=None):

    if semilla is not None:
        np.random.seed(semilla) # Fija la semilla para esta réplica

    reloj = 0
    fel = []
    contador_eventos = 0
   
    # Servidores ocupados por fase
    servidores_ocupados_rev = 0
    servidores_ocupados_rep = 0
    servidores_ocupados_cheq = 0

    cola_revision = deque()
    cola_reparacion = deque()
    cola_chequeo = deque()

    estadisticas_vehiculos = defaultdict(list) # Para métricas por vehículo
    vehiculos_en_sistema = 0

    # Métricas para utilización de servidores por fase
    tiempo_servidores_ocupados_rev = 0
    tiempo_servidores_ocupados_rep = 0
    tiempo_servidores_ocupados_cheq = 0
    ultimo_evento_reloj = 0  # Para calcular el tiempo de ocupación entre eventos

    # Programar primer evento de llegada
    heapq.heappush(fel, Evento(tiempo_llegada(lam_llegadas), contador_eventos, EVENTO_LLEGADA, {}))
    contador_eventos += 1

    while fel and reloj < tiempo_max:
        evento = heapq.heappop(fel)

        # Actualizar tiempo de servidores ocupados por fase desde el último evento
        tiempo_transcurrido_desde_ultimo_evento = evento.tiempo - ultimo_evento_reloj
        if tiempo_transcurrido_desde_ultimo_evento > 0: # Asegurarse de que el tiempo avanzó
            tiempo_servidores_ocupados_rev += servidores_ocupados_rev * tiempo_transcurrido_desde_ultimo_evento
            tiempo_servidores_ocupados_rep += servidores_ocupados_rep * tiempo_transcurrido_desde_ultimo_evento
            tiempo_servidores_ocupados_cheq += servidores_ocupados_cheq * tiempo_transcurrido_desde_ultimo_evento
        ultimo_evento_reloj = evento.tiempo # Actualiza el último reloj antes de procesar el evento

        reloj = evento.tiempo

        if evento.tipo == EVENTO_LLEGADA:
            vehiculos_en_sistema += 1
            cola_revision.append({'llegada': reloj})
            heapq.heappush(fel, Evento(reloj + tiempo_llegada(lam_llegadas), contador_eventos, EVENTO_LLEGADA, {}))
            contador_eventos += 1
            # Intenta iniciar servicio en revisión
            if servidores_ocupados_rev < n_servidores_rev and cola_revision:
                servidores_ocupados_rev += 1
                datos = cola_revision.popleft()
                datos['inicio_revision'] = reloj
                heapq.heappush(fel, Evento(reloj + tiempo_revision(), contador_eventos, EVENTO_FIN_REV, datos))
                contador_eventos += 1

        elif evento.tipo == EVENTO_FIN_REV:
            tipo_rep = tipo_reparacion()
            evento.datos['tipo_rep'] = tipo_rep
            evento.datos['fin_revision'] = reloj
            cola_reparacion.append(evento.datos)
            servidores_ocupados_rev -= 1 # Un servidor de revisión se libera

            # Intenta iniciar un nuevo servicio en Revisión si hay cola
            if servidores_ocupados_rev < n_servidores_rev and cola_revision:
                servidores_ocupados_rev += 1
                datos = cola_revision.popleft()
                datos['inicio_revision'] = reloj
                heapq.heappush(fel, Evento(reloj + tiempo_revision(), contador_eventos, EVENTO_FIN_REV, datos))
                contador_eventos += 1

            # Intenta iniciar un servicio en Reparación (el vehículo que acaba de salir de revisión)
            if servidores_ocupados_rep < n_servidores_rep and cola_reparacion:
                servidores_ocupados_rep += 1
                datos = cola_reparacion.popleft()
                datos['inicio_reparacion'] = reloj
                t_rep = max(0, tiempo_reparacion(datos['tipo_rep']))
                heapq.heappush(fel, Evento(reloj + t_rep, contador_eventos, EVENTO_FIN_REP, datos))
                contador_eventos += 1

        elif evento.tipo == EVENTO_FIN_REP:
            evento.datos['fin_reparacion'] = reloj
            cola_chequeo.append(evento.datos)
            servidores_ocupados_rep -= 1 # Un servidor de reparación se libera

            # Intenta iniciar un nuevo servicio en Reparación si hay cola
            if servidores_ocupados_rep < n_servidores_rep and cola_reparacion:
                servidores_ocupados_rep += 1
                datos = cola_reparacion.popleft()
                datos['inicio_reparacion'] = reloj
                t_rep = max(0, tiempo_reparacion(datos['tipo_rep']))
                heapq.heappush(fel, Evento(reloj + t_rep, contador_eventos, EVENTO_FIN_REP, datos))
                contador_eventos += 1

            # Intenta iniciar un servicio en Chequeo (el vehículo que acaba de salir de reparación)
            if servidores_ocupados_cheq < n_servidores_cheq and cola_chequeo:
                servidores_ocupados_cheq += 1
                datos = cola_chequeo.popleft()
                datos['inicio_chequeo'] = reloj
                t_cheq = max(0, tiempo_chequeo())
                heapq.heappush(fel, Evento(reloj + t_cheq, contador_eventos, EVENTO_FIN_CHEQUEO, datos))
                contador_eventos += 1

        elif evento.tipo == EVENTO_FIN_CHEQUEO:
            evento.datos['fin_chequeo'] = reloj
            if pasa_calidad(p_fallo_calidad):
                heapq.heappush(fel, Evento(reloj, contador_eventos, EVENTO_SALIDA, evento.datos))
                contador_eventos += 1
            else:
                evento.datos['retrabajo'] = evento.datos.get('retrabajo', 0) + 1
                tipo_rep = tipo_reparacion() # Se genera un nuevo tipo de reparación para el retrabajo
                evento.datos['tipo_rep'] = tipo_rep
                cola_reparacion.append(evento.datos) # Vuelve a la cola de reparación
            servidores_ocupados_cheq -= 1 # Un servidor de chequeo se libera

            # Intenta iniciar un nuevo servicio en Chequeo si hay cola
            if servidores_ocupados_cheq < n_servidores_cheq and cola_chequeo:
                servidores_ocupados_cheq += 1
                datos = cola_chequeo.popleft()
                datos['inicio_chequeo'] = reloj
                t_cheq = max(0, tiempo_chequeo())
                heapq.heappush(fel, Evento(reloj + t_cheq, contador_eventos, EVENTO_FIN_CHEQUEO, datos))
                contador_eventos += 1

            # Intenta iniciar un servicio en Reparación (para retrabajo o nuevo)
            if servidores_ocupados_rep < n_servidores_rep and cola_reparacion:
                servidores_ocupados_rep += 1
                datos = cola_reparacion.popleft()
                datos['inicio_reparacion'] = reloj
                t_rep = max(0, tiempo_reparacion(datos['tipo_rep']))
                heapq.heappush(fel, Evento(reloj + t_rep, contador_eventos, EVENTO_FIN_REP, datos))
                contador_eventos += 1

        elif evento.tipo == EVENTO_SALIDA:
            evento.datos['salida'] = reloj
            estadisticas_vehiculos['tiempos_sistema'].append(evento.datos['salida'] - evento.datos['llegada'])
            estadisticas_vehiculos['retrabajos'].append(evento.datos.get('retrabajo', 0))
            estadisticas_vehiculos['tipo_rep'].append(evento.datos['tipo_rep'])
            vehiculos_en_sistema -= 1

    # Asegurar que el cálculo de tiempo_servidores_ocupados sea hasta el final de la simulación
    tiempo_transcurrido_final = reloj - ultimo_evento_reloj
    if tiempo_transcurrido_final > 0:
        tiempo_servidores_ocupados_rev += servidores_ocupados_rev * tiempo_transcurrido_final
        tiempo_servidores_ocupados_rep += servidores_ocupados_rep * tiempo_transcurrido_final
        tiempo_servidores_ocupados_cheq += servidores_ocupados_cheq * tiempo_transcurrido_final

    # --- Recopilación de Resultados de la Réplica ---
    resultados_replica = {}
   
    resultados_replica['vehiculos_procesados'] = len(estadisticas_vehiculos['tiempos_sistema'])
    
    if resultados_replica['vehiculos_procesados'] > 0:
        tiempos = np.array(estadisticas_vehiculos['tiempos_sistema'])
        resultados_replica['tiempo_sistema_promedio'] = np.mean(tiempos)
        resultados_replica['retrabajos_promedio'] = np.mean(estadisticas_vehiculos['retrabajos'])
    else: # Si no se procesaron vehículos, las métricas son 0 o NaN
        resultados_replica['tiempo_sistema_promedio'] = 0
        resultados_replica['retrabajos_promedio'] = 0

    # Utilización de servidores por fase
    # Asegurarse de no dividir por cero si la simulación no duró (ej. tiempo_max muy corto o sin llegadas)
    resultados_replica['utilizacion_rev'] = (tiempo_servidores_ocupados_rev / (n_servidores_rev * reloj)) * 100 if reloj > 0 and n_servidores_rev > 0 else 0
    resultados_replica['utilizacion_rep'] = (tiempo_servidores_ocupados_rep / (n_servidores_rep * reloj)) * 100 if reloj > 0 and n_servidores_rep > 0 else 0
    resultados_replica['utilizacion_cheq'] = (tiempo_servidores_ocupados_cheq / (n_servidores_cheq * reloj)) * 100 if reloj > 0 and n_servidores_cheq > 0 else 0
    
    return resultados_replica

# --- Función para ejecutar múltiples réplicas y calcular estadísticas generales ---
def ejecutar_escenario(nombre_escenario, params, num_replicas=NUM_REPLICAS):
    print(f"\n--- {nombre_escenario} ({num_replicas} Réplicas) ---")
   
    # Listas para almacenar los resultados de cada réplica
    vehiculos_procesados_replicas = []
    tiempo_sistema_promedio_replicas = []
    retrabajos_promedio_replicas = []
    utilizacion_rev_replicas = []
    utilizacion_rep_replicas = []
    utilizacion_cheq_replicas = []

    for i in range(num_replicas):
        # Usar una semilla diferente para cada réplica para asegurar independencia
        # La semilla es 'i' para que la secuencia de réplicas sea reproducible
        resultados_replica = simular(
            tiempo_max=TIEMPO_MAX,
            lam_llegadas=params.get('lam_llegadas', LAMBDA_LLEGADAS_BASE),
            p_fallo_calidad=params.get('p_fallo_calidad', P_FALLO_CALIDAD_BASE),
            n_servidores_rev=params.get('n_servidores_rev', N_SERVIDORES_REV_BASE),
            n_servidores_rep=params.get('n_servidores_rep', N_SERVIDORES_REP_BASE),
            n_servidores_cheq=params.get('n_servidores_cheq', N_SERVIDORES_CHEQ_BASE),
            semilla=i # Semilla diferente para cada réplica
        )
        vehiculos_procesados_replicas.append(resultados_replica['vehiculos_procesados'])
        tiempo_sistema_promedio_replicas.append(resultados_replica['tiempo_sistema_promedio'])
        retrabajos_promedio_replicas.append(resultados_replica['retrabajos_promedio'])
        utilizacion_rev_replicas.append(resultados_replica['utilizacion_rev'])
        utilizacion_rep_replicas.append(resultados_replica['utilizacion_rep'])
        utilizacion_cheq_replicas.append(resultados_replica['utilizacion_cheq'])

    # --- Cálculo de Estadísticas (Promedio e Intervalo de Confianza) ---
    def calcular_estadisticas(data):
        data = np.array(data)
        media = np.mean(data)
        std = np.std(data, ddof=1) # Desviación estándar muestral
        if NUM_REPLICAS > 1:
            error_estandar = std / np.sqrt(NUM_REPLICAS)
            z = 1.96 # Valor z para 95% de confianza (para N >= 30, se usa normal)
            margen_error = z * error_estandar
            ic_inferior = media - margen_error
            ic_superior = media + margen_error
        else: # Si solo hay una réplica, no se puede calcular IC
            ic_inferior, ic_superior = np.nan, np.nan # Not a Number
        return media, ic_inferior, ic_superior

    print("\n--- Resumen General ---")
    media_vehiculos, ic_vehiculos_inf, ic_vehiculos_sup = calcular_estadisticas(vehiculos_procesados_replicas)
    print(f"Vehículos procesados (Promedio): {media_vehiculos:.0f} (IC 95%: [{ic_vehiculos_inf:.0f}, {ic_vehiculos_sup:.0f}])")

    media_tiempo_sistema, ic_tiempo_sistema_inf, ic_tiempo_sistema_sup = calcular_estadisticas(tiempo_sistema_promedio_replicas)
    print(f"Tiempo promedio en sistema (Promedio): {media_tiempo_sistema:.2f} min (IC 95%: [{ic_tiempo_sistema_inf:.2f}, {ic_tiempo_sistema_sup:.2f}] min)")

    media_retrabajos, ic_retrabajos_inf, ic_retrabajos_sup = calcular_estadisticas(retrabajos_promedio_replicas)
    print(f"Retrabajos promedio por vehículo (Promedio): {media_retrabajos:.3f} (IC 95%: [{ic_retrabajos_inf:.3f}, {ic_retrabajos_sup:.3f}])")

    print("\n--- Utilización de Servidores por Fase (Promedio) ---")
    media_util_rev, ic_util_rev_inf, ic_util_rev_sup = calcular_estadisticas(utilizacion_rev_replicas)
    print(f"Revisión: {media_util_rev:.2f}% (IC 95%: [{ic_util_rev_inf:.2f}, {ic_util_rev_sup:.2f}]%)")

    media_util_rep, ic_util_rep_inf, ic_util_rep_sup = calcular_estadisticas(utilizacion_rep_replicas)
    print(f"Reparación: {media_util_rep:.2f}% (IC 95%: [{ic_util_rep_inf:.2f}, {ic_util_rep_sup:.2f}]%)")

    media_util_cheq, ic_util_cheq_inf, ic_util_cheq_sup = calcular_estadisticas(utilizacion_cheq_replicas)
    print(f"Chequeo: {media_util_cheq:.2f}% (IC 95%: [{ic_util_cheq_inf:.2f}, {ic_util_cheq_sup:.2f}]%)")

# --- Ejecución de Escenarios ---
if __name__ == "__main__":
    # Escenario 1: Simulación Base
    ejecutar_escenario(
        "Simulación Base",
        params={
            'lam_llegadas': LAMBDA_LLEGADAS_BASE,
            'p_fallo_calidad': P_FALLO_CALIDAD_BASE,
            'n_servidores_rev': N_SERVIDORES_REV_BASE,
            'n_servidores_rep': N_SERVIDORES_REP_BASE,
            'n_servidores_cheq': N_SERVIDORES_CHEQ_BASE
        }
    )

    # --- Análisis de Sensibilidad (Ejemplos) ---

    # Escenario 2: Mayor Tasa de Llegadas (lambda = 3/60)
    ejecutar_escenario(
        "Simulación con mayor tasa de llegadas (λ=3/60)",
        params={
            'lam_llegadas': 3/60, # Alteramos solo este parámetro
            'p_fallo_calidad': P_FALLO_CALIDAD_BASE,
            'n_servidores_rev': N_SERVIDORES_REV_BASE,
            'n_servidores_rep': N_SERVIDORES_REP_BASE,
            'n_servidores_cheq': N_SERVIDORES_CHEQ_BASE
        }
    )

    # Escenario 3: Menor Probabilidad de Fallo en Chequeo (p = 0.05)
    ejecutar_escenario(
        "Simulación con menor probabilidad de fallo en chequeo (p=0.05)",
        params={
            'lam_llegadas': LAMBDA_LLEGADAS_BASE,
            'p_fallo_calidad': 0.05, # Alteramos solo este parámetro
            'n_servidores_rev': N_SERVIDORES_REV_BASE,
            'n_servidores_rep': N_SERVIDORES_REP_BASE,
            'n_servidores_cheq': N_SERVIDORES_CHEQ_BASE
        }
    )

    # --- Ejemplo de Sistema Modificado (para tus conclusiones) ---
    # Podrías, por ejemplo, mover un mecánico de Reparación a Chequeo
    ejecutar_escenario(
        "Sistema Modificado: 1 mecánico menos en Reparación, 1 más en Chequeo",
        params={
            'lam_llegadas': LAMBDA_LLEGADAS_BASE,
            'p_fallo_calidad': P_FALLO_CALIDAD_BASE,
            'n_servidores_rev': N_SERVIDORES_REV_BASE,
            'n_servidores_rep': N_SERVIDORES_REP_BASE - 1, # Un mecánico menos
            'n_servidores_cheq': N_SERVIDORES_CHEQ_BASE + 1 # Un mecánico más
        }
    )