import os
import mlflow

# 1. Configurar la URI de rastreo al servidor local
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)

# 2. Asegurar el experimento objetivo: "Default"
mlflow.set_experiment("Default")

print("Iniciando registro de traza GenAI en MLflow...")

# 3. Usar mlflow.start_span para crear una traza manual válida
with mlflow.start_span(name="openai_chat_completion_mock") as span:
    # Registramos las entradas y atributos en el span de la traza
    span.set_inputs({"model": "gpt-4o-mini", "user_message": "Hello!"})
    
    resultado_simulado = "¡Hola! Conexión simulada con éxito para la traza de MLflow."
    
    # Registramos las salidas
    span.set_outputs({"assistant_response": resultado_simulado})

print("¡Traza enviada con éxito! Revisa tu MLflow UI en la pestaña de GenAI.")