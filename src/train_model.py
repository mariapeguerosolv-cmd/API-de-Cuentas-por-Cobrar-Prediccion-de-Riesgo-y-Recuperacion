import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)

import joblib
import mlflow
mlflow.set_experiment("AR_Model_Training")
import mlflow.sklearn


# 1. CARGA DE DATOS
df = pd.read_csv("src/dataset_predictive_collection.csv")

print("=== Primeras filas del dataset ===")
print(df.head())
print("\n=== Estadísticas descriptivas ===")
print(df.describe())

# -------------------------------
# CLASIFICACIÓN (riesgo)
# -------------------------------
X_class = df[["monto_factura", "dias_atraso", "historial_pago", "frecuencia_disputas"]]
y_class = df["riesgo"]

le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class_encoded, test_size=0.2, random_state=42, stratify=y_class_encoded
)

with mlflow.start_run(run_name="Clasificación_vs_Regresión"):

    # Entrenar modelo clasificación
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_class, y_train_class)
    y_pred_class = log_reg.predict(X_test_class)

    acc = accuracy_score(y_test_class, y_pred_class)
    f1 = f1_score(y_test_class, y_pred_class, average="weighted")

    print("\n=== Clasificación: Logistic Regression ===")
    print("Accuracy:", acc)
    print("F1-Score:", f1)
    print("\nReporte de clasificación:\n", classification_report(y_test_class, y_pred_class))

    mlflow.log_metric("Accuracy_classification", acc)
    mlflow.log_metric("F1_classification", f1)
    mlflow.sklearn.log_model(log_reg, "classification_model")

    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test_class, y_pred_class), annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Matriz de Confusión - Clasificación")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Guardar modelo y codificador de etiquetas
    joblib.dump(log_reg, "classification_model.pkl")
    joblib.dump(le, "label_encoder.pkl")

    # -------------------------------
    # REGRESIÓN (recuperación esperada)
    # -------------------------------
    X_reg = df[["monto_factura", "dias_atraso", "historial_pago", "frecuencia_disputas"]]
    y_reg = df["recuperacion_esperada"]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = lin_reg.predict(X_test_reg)

    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = mse**0.5
    r2 = r2_score(y_test_reg, y_pred_reg)

    print("\n=== Regresión Lineal ===")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")

    cv_scores = cross_val_score(lin_reg, X_reg, y_reg, cv=5, scoring="r2")
    print(f"Cross-Validation R² promedio: {cv_scores.mean():.4f}")

    mlflow.log_metric("MSE_regression", mse)
    mlflow.log_metric("RMSE_regression", rmse)
    mlflow.log_metric("R2_regression", r2)
    mlflow.sklearn.log_model(lin_reg, "regression_model")
    mlflow.log_metric("CV_R2_regression", cv_scores.mean())

    plt.figure(figsize=(6,6))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
    plt.xlabel("Valores reales")
    plt.ylabel("Predicciones")
    plt.title("Regresión Lineal - Valores reales vs predichos")
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], "r--")
    plt.savefig("regresion_dispersion.png")
    plt.close()

    errores = y_test_reg - y_pred_reg
    plt.figure(figsize=(6,4))
    sns.histplot(errores, bins=20, kde=True)
    plt.title("Distribución de errores (residuos)")
    plt.xlabel("Error")
    plt.ylabel("Frecuencia")
    plt.savefig("regresion_errores.png")
    plt.close()

    joblib.dump(lin_reg, "regression_model.pkl")

    # -------------------------------
    # TABLA COMPARATIVA FINAL
    # -------------------------------
    comparacion = pd.DataFrame({
        "Modelo": ["Clasificación (Logistic Regression)", "Regresión Lineal"],
        "Métrica Principal": ["F1-Score", "R²"],
        "Valor": [f1, r2]
    })

    print("\n=== Comparación Final ===")
    print(comparacion)

    comparacion.to_csv("comparacion_modelos.csv", index=False)
    mlflow.log_artifact("comparacion_modelos.csv")

print("\nModelos guardados correctamente.")
print("MLflow registró el entrenamiento exitosamente.")