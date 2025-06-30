import numpy as np
import heapq
# from scipy.stats import norm, triang
from collections import deque, defaultdict

# Parámetros del sistema (puedes modificar para análisis de sensibilidad)
LAMBDA_LLEGADAS = 2/60  # vehículos por minuto
REV_PREVIA_MEDIA = 15
REP_LIGERA_MEDIA, REP_LIGERA_STD = 30, 5
REP_MEDIA_MEDIA, REP_MEDIA_STD = 60, 10
REP_COMPLEJA_MIN, REP_COMPLEJA_MODE, REP_COMPLEJA_MAX = 120, 280, 480
CHEQUEO_MEDIA, CHEQUEO_STD = 10, 2
P_FALLO_CALIDAD = 0.10

# Configuración de simulación
TIEMPO_MAX = 2920 * 60  # 2920 horas en minutos (1 año)
N_SERVIDORES = 3  # Ejemplo: 3 mecánicos

# Definición de eventos
EVENTO_LLEGADA = 'llegada'
EVENTO_FIN_REV = 'fin_revision'
EVENTO_FIN_REP = 'fin_reparacion'
EVENTO_FIN_CHEQUEO = 'fin_chequeo'
EVENTO_SALIDA = 'salida'

# Fases de reparación
TIPO_REP_LIGERA = 'ligera'
TIPO_REP_MEDIA = 'media'
TIPO_REP_COMPLEJA = 'compleja'
TIPOS_REP = [TIPO_REP_LIGERA, TIPO_REP_MEDIA, TIPO_REP_COMPLEJA]
PROB_REP = [0.5, 0.3, 0.2]  # Probabilidad de cada tipo de reparación

# Generadores de variables aleatorias
def tiempo_llegada():
    return np.random.exponential(1 / LAMBDA_LLEGADAS)

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

def pasa_calidad():
    return np.random.rand() > P_FALLO_CALIDAD

# Estructura de evento: (tiempo, contador, tipo_evento, datos)
class Evento:
    def __init__(self, tiempo, contador, tipo, datos):
        self.tiempo = tiempo
        self.contador = contador
        self.tipo = tipo
        self.datos = datos
    def __lt__(self, other):
        return (self.tiempo, self.contador) < (other.tiempo, other.contador)

# Simulación principal
def simular(tiempo_max=TIEMPO_MAX, n_servidores=N_SERVIDORES, semilla=None):
    if semilla is not None:
        np.random.seed(semilla)
    reloj = 0
    fel = []
    contador_eventos = 0
    servidores_libres = n_servidores
    cola_revision = deque()
    cola_reparacion = deque()
    cola_chequeo = deque()
    estadisticas = defaultdict(list)
    vehiculos_en_sistema = 0

    # Programar primer evento de llegada
    heapq.heappush(fel, Evento(tiempo_llegada(), contador_eventos, EVENTO_LLEGADA, {}))
    contador_eventos += 1

    while fel and reloj < tiempo_max:
        evento = heapq.heappop(fel)
        reloj = evento.tiempo

        if evento.tipo == EVENTO_LLEGADA:
            # Llegada de vehículo
            vehiculos_en_sistema += 1
            cola_revision.append({'llegada': reloj})
            # Programar siguiente llegada
            heapq.heappush(fel, Evento(reloj + tiempo_llegada(), contador_eventos, EVENTO_LLEGADA, {}))
            contador_eventos += 1
            # Si hay servidor libre, iniciar revisión
            if servidores_libres > 0:
                servidores_libres -= 1
                datos = cola_revision.popleft()
                datos['inicio_revision'] = reloj
                heapq.heappush(fel, Evento(reloj + tiempo_revision(), contador_eventos, EVENTO_FIN_REV, datos))
                contador_eventos += 1

        elif evento.tipo == EVENTO_FIN_REV:
            # Fin de revisión, decidir tipo de reparación
            tipo_rep = tipo_reparacion()
            evento.datos['tipo_rep'] = tipo_rep
            evento.datos['fin_revision'] = reloj
            cola_reparacion.append(evento.datos)
            # Si hay servidor libre, iniciar reparación
            if servidores_libres > 0:
                servidores_libres -= 1
                datos = cola_reparacion.popleft()
                datos['inicio_reparacion'] = reloj
                t_rep = max(0, tiempo_reparacion(datos['tipo_rep']))
                heapq.heappush(fel, Evento(reloj + t_rep, contador_eventos, EVENTO_FIN_REP, datos))
                contador_eventos += 1
            # Liberar servidor de revisión
            servidores_libres += 1
            # Si hay más en cola de revisión, iniciar siguiente
            if cola_revision and servidores_libres > 0:
                servidores_libres -= 1
                datos = cola_revision.popleft()
                datos['inicio_revision'] = reloj
                heapq.heappush(fel, Evento(reloj + tiempo_revision(), contador_eventos, EVENTO_FIN_REV, datos))
                contador_eventos += 1

        elif evento.tipo == EVENTO_FIN_REP:
            # Fin de reparación, pasar a chequeo
            evento.datos['fin_reparacion'] = reloj
            cola_chequeo.append(evento.datos)
            # Si hay servidor libre, iniciar chequeo
            if servidores_libres > 0:
                servidores_libres -= 1
                datos = cola_chequeo.popleft()
                datos['inicio_chequeo'] = reloj
                t_cheq = max(0, tiempo_chequeo())
                heapq.heappush(fel, Evento(reloj + t_cheq, contador_eventos, EVENTO_FIN_CHEQUEO, datos))
                contador_eventos += 1
            # Liberar servidor de reparación
            servidores_libres += 1
            # Si hay más en cola de reparación, iniciar siguiente
            if cola_reparacion and servidores_libres > 0:
                servidores_libres -= 1
                datos = cola_reparacion.popleft()
                datos['inicio_reparacion'] = reloj
                t_rep = max(0, tiempo_reparacion(datos['tipo_rep']))
                heapq.heappush(fel, Evento(reloj + t_rep, contador_eventos, EVENTO_FIN_REP, datos))
                contador_eventos += 1

        elif evento.tipo == EVENTO_FIN_CHEQUEO:
            # Fin de chequeo, decidir si sale o vuelve a reparación
            evento.datos['fin_chequeo'] = reloj
            if pasa_calidad():
                # Sale del sistema
                heapq.heappush(fel, Evento(reloj, contador_eventos, EVENTO_SALIDA, evento.datos))
                contador_eventos += 1
            else:
                # Vuelve a reparación (retrabajo, tipo aleatorio)
                evento.datos['retrabajo'] = evento.datos.get('retrabajo', 0) + 1
                tipo_rep = tipo_reparacion()
                evento.datos['tipo_rep'] = tipo_rep
                cola_reparacion.append(evento.datos)
                if servidores_libres > 0:
                    servidores_libres -= 1
                    datos = cola_reparacion.popleft()
                    datos['inicio_reparacion'] = reloj
                    t_rep = max(0, tiempo_reparacion(datos['tipo_rep']))
                    heapq.heappush(fel, Evento(reloj + t_rep, contador_eventos, EVENTO_FIN_REP, datos))
                    contador_eventos += 1
            # Liberar servidor de chequeo
            servidores_libres += 1
            # Si hay más en cola de chequeo, iniciar siguiente
            if cola_chequeo and servidores_libres > 0:
                servidores_libres -= 1
                datos = cola_chequeo.popleft()
                datos['inicio_chequeo'] = reloj
                t_cheq = max(0, tiempo_chequeo())
                heapq.heappush(fel, Evento(reloj + t_cheq, contador_eventos, EVENTO_FIN_CHEQUEO, datos))
                contador_eventos += 1

        elif evento.tipo == EVENTO_SALIDA:
            # Salida del sistema, recolectar estadísticas
            evento.datos['salida'] = reloj
            estadisticas['tiempos_sistema'].append(evento.datos['salida'] - evento.datos['llegada'])
            estadisticas['retrabajos'].append(evento.datos.get('retrabajo', 0))
            estadisticas['tipo_rep'].append(evento.datos['tipo_rep'])
            vehiculos_en_sistema -= 1

    # Análisis de resultados
    tiempos = np.array(estadisticas['tiempos_sistema'])
    media = np.mean(tiempos)
    std = np.std(tiempos, ddof=1)
    n = len(tiempos)
    conf = 0.95
    # z = norm.ppf(1 - (1 - conf) / 2)
    z = 1.96  # Aproximación para 95% confianza
    error = z * std / np.sqrt(n)
    print(f"Vehículos procesados: {n}")
    print(f"Tiempo promedio en sistema: {media:.2f} min")
    print(f"Intervalo de confianza 95%: [{media - error:.2f}, {media + error:.2f}] min")
    print(f"Retrabajos promedio por vehículo: {np.mean(estadisticas['retrabajos']):.3f}")
    print(f"Distribución de tipos de reparación: {dict(zip(*np.unique(estadisticas['tipo_rep'], return_counts=True)))}")
    return estadisticas

# Ejemplo de análisis de sensibilidad
if __name__ == "__main__":
    print("Simulación base:")
    simular()
    print("\nSimulación con mayor tasa de llegadas (λ=3/60):")
    LAMBDA_LLEGADAS = 3/60
    simular()
    print("\nSimulación con menor probabilidad de fallo en chequeo (p=0.05):")
    LAMBDA_LLEGADAS = 2/60
    P_FALLO_CALIDAD = 0.05
    simular()
