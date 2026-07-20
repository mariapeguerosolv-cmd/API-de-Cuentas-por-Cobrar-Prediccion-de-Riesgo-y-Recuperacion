# API-de-Cuentas-por-Cobrar-Prediccion-de-Riesgo-y-Recuperacion

**Proyecto:** API de Cuentas por Cobrar — Predicción de Riesgo y Recuperación

**Entorno de Ejecución:** FastAPI (`[http://127.0.0.1:8000](http://127.0.0.1:8000)`) & MLflow Monitor (`[http://127.0.0.1:5001](http://127.0.0.1:5001)`)

## 1. Documento Técnico

### A. Definición de Indicadores (Métricas, Logs, Trazas)

Para garantizar la estabilidad y audibilidad del sistema, se implementa una estrategia de monitoreo multinivel:

-   **Métricas del Modelo:** Evaluadas y registradas mediante MLflow en cada ciclo.
    
    -   _Clasificación (Logistic Regression):_ **F1-Score Base de 0.6893**, Accuracy de 0.70, Precision de 0.69 y Recall de 0.70.
        
    -   _Regresión (Linear Regression):_ **$R^2$ de 0.5715** y un RMSE de 5,784.31.
        
-   **Métricas Operativas (Golden Signals):**
    
    -   _Latencia de predicción:_ Tiempo que le toma al modelo procesar un registro. En producción promedia **0.001 a 0.005 segundos** por petición según los logs de FastAPI.
        
    -   _Throughput:_ Cantidad de solicitudes procesadas de forma concurrente por Uvicorn.
        
-   **Logs Estructurados:** Almacenados localmente en `logs\predictions.log` con formato estandarizado mediante marcas de tiempo y niveles de severidad (`INFO`, `WARNING`, `ERROR`). Registran inicialización de artefactos (`model_loaded=true`) y el cuerpo de las predicciones.
    
-   **Trazas:** Monitoreo del ciclo completo del flujo de la petición: `HTTP POST` $\rightarrow$ `FastAPI Endpoint` $\rightarrow$ `Inferencia en Array bidimensional (X)` $\rightarrow$ `JSON Response` junto con el tiempo exacto de ejecución en milisegundos.

<img width="1223" height="802" alt="image" src="https://github.com/user-attachments/assets/20ceb408-1c46-4029-8c32-3a81abab4fa9" />

Imagen 1: interfaz de API de FastAPI.

### B. Diseño de Alertas Inteligentes

Configuración basada en los Acuerdos de Nivel de Servicio (SLO) y los presupuestos de error (_Error Budgets_), priorizando incidentes según su impacto crítico en el negocio:

-   **Alerta de Degradación de Modelo (Crítica):** Se dispara si el Accuracy acumulado disminuye por debajo del **70%** ($< 0.70$) o el $R^2$ cae por debajo de **0.50**. Agota inmediatamente el presupuesto de error de precisión del negocio.
    
-   **Alerta de Degradación de Infraestructura (Warning):** Se activa si la latencia promedio supera los **500 ms** en una ventana de 5 minutos.
    
-   **Alerta de Errores del Servidor (Crítica):** Se dispara si la tasa de errores HTTP 5xx supera el **5%** de las peticiones concurrentes, indicando fallos en la carga del backend o corrupción de los archivos `.pkl`.
    

### C. Descripción de Dashboards

El monitoreo visual y operativo se centraliza mediante dos tableros estratégicos:

1.  **Dashboard de Experimentos y Evaluación (MLflow UI):** Utilizado por el equipo de Ciencia de Datos para auditar el rendimiento y registrar los artefactos (`classification_model.pkl` y `regression_model.pkl`). Muestra gráficos en tiempo real de curvas de aprendizaje, matrices de confusión e importancia de variables.
    
2.  **Dashboard Operativo (Métricas de Servicio):** Diseñado sobre el estándar de _Golden Signals_ para el equipo de MLOps/DevOps. Monitorea la salud de la API en el puerto activo (`8000`), registrando los códigos de estado HTTP (200 OK, 404 Not Found) e identificando sobrecargas en el hilo de procesamiento.

<img width="1506" height="257" alt="image" src="https://github.com/user-attachments/assets/d572a6a0-d539-47ae-a5da-abd391eefcac" />


Imagen 2: Tabla de MLflow que resume la corrida genai_evaluation_run

# D. Runbooks de Respuesta a Incidentes

| **Incidente**                              | **Pasos de Respuesta**                                                                                                                                       | **Responsable**        | **Resultado Esperado**                                      |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-------------------------------------------------------------|
| Accuracy Bajo (<70%)                        | 1. Detectar caída en MLflow UI.<br>2. Notificar vía webhook a MLOps.<br>3. Ejecutar script de Rollback a versión estable.<br>4. Validar métricas.             | Ingeniero MLOps        | Restauración del Accuracy a ≥70%.                          |
| Latencia Elevada (>500 ms)                  | 1. Identificar cuello de botella en Uvicorn.<br>2. Escalar recursos (CPU/RAM) en la plataforma de hosting (Render).<br>3. Validar tiempos de respuesta.      | Ingeniero DevOps       | Latencia normalizada a <200 ms.                            |
| Error de Socket / Puerto Ocupado (WinError 10048) | 1. Detectar colisión en puerto 5000.<br>2. Matar proceso fantasma de Flask vía PowerShell (Stop-Process) o redirigir dinámicamente al Puerto 8000 en app.py. | Soporte / SysAdmin     | API activa y escuchando peticiones correctamente.           |
| Data Drift Detectado                        | 1. Identificar desviación estadística en variables clave.<br>2. Disparar pipeline de reentrenamiento.<br>3. Validar y promover nuevo modelo.                 | Científico de Datos    | Despliegue de versión actualizada y alineada al mercado.    |


### E. Estrategias de Detección de Data Drift

Para evitar la obsolescencia del modelo en producción, se diseña un esquema de monitoreo continuo sobre las variables de negocio de Cuentas por Cobrar (`monto_factura`, `dias_atraso`, `historial_pago`, `frecuencia_disputas`):

-   **Frecuencia de revisión:** Automatizada de forma quincenal mediante tareas programadas.
    
-   **Pruebas Estadísticas:** Implementación de la prueba **Kolmogorov-Smirnov (KS-test)** sobre la distribución continua de la variable `monto_factura` y divergencia de Kullback-Leibler (KL) para las frecuencias categóricas.
    
-   **Criterio de Drift:** Si el p-valor de la prueba KS es inferior a 0.05 ($p < 0.05$) o el Índice de Estabilidad de Población (PSI) supera 0.25, el sistema declara oficialmente un estado de _Data Drift_ y genera una alerta automatizada.
    

### F. Documentación de Respuesta Automatizada

-   **Rollback Automático:** Ante la degradación confirmada de métricas en producción, el sistema cambia los punteros de los archivos cargados (`joblib.load`) hacia la última versión etiquetada como estable (`v1.0.0`) dentro del registro de MLflow, mitigando impactos financieros.
    
-   **Reentrenamiento Programado:** Al confirmarse un estado de _Data Drift_, se dispara automáticamente un pipeline en segundo plano que recolecta los datos sintéticos e históricos recientes, reajusta los hiperparámetros de la regresión y clasificación, y genera un nuevo reporte comparativo sin detener el servicio.
    
-   **Escalamiento de Recursos:** En picos de tráfico que afecten la latencia, el entorno se reconfigura dinámicamente incrementando los workers de Uvicorn y los límites de computo.
    

## 2. Evidencia de Implementación

### A. Registro y Carga de Artefactos (MLflow Logs)

Los modelos fueron entrenados, evaluados y serializados de manera correcta bajo la ruta del proyecto actual:

=== Regresión Lineal ===
MSE: 33458257.80 | RMSE: 5784.31 | R²: 0.5715
Cross-Validation R² promedio: 0.5922
Modelos guardados correctamente.
MLflow registró el entrenamiento exitosamente.

### B. Logs Operativos de la API (FastAPI Telemetry)

Captura real del archivo de auditoría interna `logs\predictions.log`, cumpliendo estrictamente con la estructura limpia solicitada para producción:

2026-07-19 13:25:34,131 | INFO | API iniciada | model_version=1.0.0 | model_loaded=true | columns_loaded=true | categories_loaded=true
2026-07-19 13:27:22,338 | INFO | Predicción exitosa: {'riesgo_predicho': 'high'}
2026-07-19 13:27:22,338 | INFO | Tiempo de respuesta /predict_risk: 0.009 segundos
2026-07-19 13:27:48,188 | INFO | Predicción exitosa: {'recuperacion_esperada': 31075.472461863534}
2026-07-19 13:27:48,188 | INFO | Tiempo de respuesta /predict_recovery: 0.003 segundos
2026-07-19 13:32:21,367 | INFO | Predicción exitosa: {'riesgo_predicho': 'low'}
2026-07-19 13:32:21,367 | INFO | Tiempo de respuesta /predict_risk: 0.002 segundos
2026-07-19 13:32:36,481 | INFO | Predicción exitosa: {'recuperacion_esperada': 26893.946715732054}
2026-07-19 13:32:36,481 | INFO | Tiempo de respuesta /predict_recovery: 0.005 segundos

## 3. Simulación de Incidentes

### Incidente 1: Conflicto de Red por Puerto Ocupado (`WinError 10048`)

-   **Evidencia:** El sistema arrojó un fallo crítico al intentar inicializar el microservicio en la dirección local por defecto: `ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 5000)`.
    
-   **Ejecución del Runbook:** Se procedió al aislamiento del proceso bloqueante redirigiendo la infraestructura web al **Puerto 8000** (`uvicorn.run(app, host="127.0.0.1", port=8000)`).
    
-   **Resultado:** Servidor activo con éxito: `INFO: Uvicorn running on [http://127.0.0.1:8000](http://127.0.0.1:8000) (Press CTRL+C to quit)`.
    

### Incidente 2: Evaluación con Métricas Saturadas (`GenAI Evaluation Run`)

-   **Evidencia:** En la pestaña `Model metrics` de MLflow UI, las gráficas de `accuracy_score`, `f1_score`, `recall_score`, `true_negatives` y `true_positives` registraron de forma lineal un valor estático de **1.00**.
    
-   **Ejecución del Runbook / Diagnóstico:** Un valor perfecto de 1.00 levantó una bandera de auditoría por posible _Overfitting_ (sobreajuste) o _Data Leakage_ debido a un volumen reducido de muestras de validación sintéticas en el script de prueba (`example_count: 2.00`).
    
-   **Resultado:** Se documenta la necesidad de ampliar el conjunto de test con datos reales desbalanceados para estabilizar las curvas de métricas en producción por debajo del 100% teórico.
    
<img width="1531" height="771" alt="image" src="https://github.com/user-attachments/assets/07ab223b-8ab5-4306-b3fe-695d0894a0d5" />

<img width="1532" height="272" alt="image" src="https://github.com/user-attachments/assets/d73a7445-ff79-4e32-adc7-02a695d27b02" />

Imagen 3: Graficas de barras mostrando incidente de Sobreajuste (Overfitting)

## 4. Preguntas de Utilidad y Ciclo de Vida del Modelo

-   **¿Cómo sabemos si el modelo sigue siendo de utilidad para el negocio?** El modelo es útil mientras genere un retorno de inversión positivo al mitigar la cartera vencida. Si el costo de reentrenamiento constante iguala o supera los beneficios económicos de las cuentas recuperadas, el ciclo de vida de este software ha concluido y se debe diseñar una nueva arquitectura desde cero, tal como lo especifica la norma de gobernanza ISO 42001.
