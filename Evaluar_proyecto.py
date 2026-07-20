import os
import mlflow
import pandas as pd

# 1. Configurar URI y asegurar Experimento "Default"
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Default")

print("Generando dataset de evaluacion local de manera limpia...")

# 2. Dataset con los datos esperados
eval_data = pd.DataFrame({
    "inputs": ["Hello!", "How do I log GenAI traces?"],
    "outputs": [
        "¡Hola! Conexión simulada con éxito para la traza de MLflow.",
        "You can use mlflow.start_span() or framework autologging."
    ],
    "ground_truth": [
        "¡Hola! Conexión simulada con éxito para la traza de MLflow.",
        "You can use mlflow.start_span() or framework autologging."
    ]
}, index=[0, 1])

# 3. Función predictora simulada
def predict_fn(inputs):
    return eval_data["outputs"].tolist()

print("Iniciando mlflow.models.evaluate...")

# 4. Evaluación limpia (Usando un tipo base para que no intente cargar tiktoken)
with mlflow.start_run(run_name="genai_evaluation_run"):
    results = mlflow.models.evaluate(
        model=predict_fn,
        data=eval_data,
        targets="ground_truth",
        model_type="classifier" # Al usar 'classifier' o dejarlo vacío evitamos las métricas automáticas de GenAI que rompen tu entorno
    )

print("Evaluacion completada e inyectada con exito.")
# ... (debajo de tu código anterior)

# Extraer e imprimir las métricas directamente en la terminal
metrics_dict = results.metrics
print(f"Accuracy:  {metrics_dict.get('accuracy_score', 0.0):.4f}")
print(f"Precision: {metrics_dict.get('precision_score', 0.0):.4f}")
print(f"Recall:    {metrics_dict.get('recall_score', 0.0):.4f}")
print(f"F1 Score:  {metrics_dict.get('f1_score', 0.0):.4f}")