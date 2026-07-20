from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging
import os
import time

# --- Crear la carpeta logs si no existe ---
if not os.path.exists('logs'):
    os.makedirs('logs')

# --- Configuración de Logs idéntica al maestro ---
logging.basicConfig(
    filename="logs/predictions.log", 
    level=logging.INFO,
    format='%(asctime)s,%(msecs)03d | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# --- Inicializar FastAPI con títulos personalizados ---
app = FastAPI(
    title="API de Cuentas por Cobrar",
    description="Interfaz interactiva para predicción de riesgo y recuperación basada en Machine Learning",
    version="1.0.0"
)

# --- Cargar modelos ---
try:
    classification_model = joblib.load("classification_model.pkl")
    regression_model = joblib.load("regression_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    logging.info("API iniciada | model_version=1.0.0 | model_loaded=true | columns_loaded=true | categories_loaded=true")
except Exception as e:
    logging.error(f"Error al cargar modelos: {e}")
    classification_model, regression_model, label_encoder = None, None, None

# --- Estructura de datos que recibirá la API ---
class DataInput(BaseModel):
    monto_factura: float
    dias_atraso: int
    historial_pago: int
    frecuencia_disputas: int

# -------------------------------
# Endpoint de clasificación (riesgo)
# -------------------------------
@app.post("/predict_risk", summary="Predice el nivel de riesgo del cliente")
def predict_risk(data: DataInput):
    start = time.time()
    try:
        X = [[data.monto_factura, data.dias_atraso, data.historial_pago, data.frecuencia_disputas]]
        pred = classification_model.predict(X)
        result = label_encoder.inverse_transform(pred)[0]
        response = {"riesgo_predicho": result}
        logging.info(f"Predicción exitosa: {response}")
        return response
    except Exception as e:
        logging.error(f"Error en predicción de riesgo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        end = time.time()
        logging.info(f"Tiempo de respuesta /predict_risk: {end-start:.3f} segundos")

# -------------------------------
# Endpoint de regresión (recuperación esperada)
# -------------------------------
@app.post("/predict_recovery", summary="Predice el monto de recuperación esperada")
def predict_recovery(data: DataInput):
    start = time.time()
    try:
        X = [[data.monto_factura, data.dias_atraso, data.historial_pago, data.frecuencia_disputas]]
        pred = regression_model.predict(X)
        result = float(pred[0])
        response = {"recuperacion_esperada": result}
        logging.info(f"Predicción exitosa: {response}")
        return response
    except Exception as e:
        logging.error(f"Error en predicción de recuperación: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        end = time.time()
        logging.info(f"Tiempo de respuesta /predict_recovery: {end-start:.3f} segundos")

# --- Arrancar servidor con Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) # <- Cambiado a 8000