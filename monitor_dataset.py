import pandas as pd, numpy as np, time, random, logging
from scipy.stats import ks_2samp

# Configuración de logs
logging.basicConfig(filename="monitor.log", level=logging.INFO)

# Cargar dataset
df = pd.read_csv("src/dataset_predictive_collection.csv")

# -------------------------------
# Funciones de métricas
# -------------------------------
def calculate_metrics(df):
    # Accuracy simulado: proporción de acción "call"
    acc = (df['accion_recomendada'] == 'call').mean()
    latency = random.randint(100,600)   # latencia simulada en ms
    error_rate = random.uniform(0.0,0.1) # tasa de error simulada
    return {"accuracy": acc, "latency_ms": latency, "error_rate": error_rate}

def detect_drift(df, column, reference):
    stat, p = ks_2samp(df[column], reference)
    return p < 0.05

# -------------------------------
# Runbooks simulados
# -------------------------------
def rollback_model():
    logging.error("⚠️ Rollback ejecutado → modelo anterior restaurado.")

def scale_resources():
    logging.warning("⚠️ Escalando recursos → latencia normalizada.")

def retrain_model():
    logging.error("⚠️ Data drift detectado → reentrenar modelo.")

# -------------------------------
# Loop de monitoreo
# -------------------------------
def monitor_loop(df):
    # Referencia inicial para drift
    ref = np.random.choice(df['monto_factura'], size=100)
    while True:
        m = calculate_metrics(df)
        logging.info(f"Métricas: {m}")

        # Incidente 1: accuracy bajo
        if m["accuracy"] < 0.7:
            rollback_model()
            logging.info("✅ Accuracy restaurado a >70% tras rollback.")

        # Incidente 2: latencia elevada
        if m["latency_ms"] > 500:
            scale_resources()
            logging.info("✅ Latencia normalizada a <200ms tras escalamiento.")

        # Incidente 3: error rate elevado
        if m["error_rate"] > 0.05:
            logging.warning("⚠️ Error rate elevado detectado.")

        # Incidente 4: data drift
        if detect_drift(df, "monto_factura", ref):
            retrain_model()
            logging.info("✅ Modelo actualizado tras reentrenamiento.")

        time.sleep(5)  # espera 5 segundos entre ciclos

# Ejecutar monitoreo
monitor_loop(df)
