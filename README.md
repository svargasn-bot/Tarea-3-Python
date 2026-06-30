# Tarea 3: Análisis, Desarrollo y Refinamiento
## Proyecto: Impacto de la Inteligencia Artificial Generativa en Estudiantes

Este proyecto implementa el ciclo de vida completo de ciencia de datos para modelar y predecir el impacto de la Inteligencia Artificial Generativa en el rendimiento académico final de los estudiantes (`Post_Semester_GPA`), utilizando técnicas de regresión supervisada y regularización.

---

## Estructura del Proyecto

El repositorio está estructurado siguiendo la plantilla recomendada por el profesor para proyectos de datos:

```text
mi-proyecto-datos/
├── data/                      # Carpeta principal de datos
│   ├── raw/                   # Datos originales (inmutables): ai_student_impact_dataset.csv
│   ├── interim/               # Datos en proceso de limpieza (no utilizados)
│   └── processed/             # Datos finales listos para modelamiento: ai_student_impact_procesado.csv
├── notebooks/                 # Archivos Jupyter Notebook (.ipynb) para experimentación
│   ├── 01_eda.ipynb           # Análisis Exploratorio de Datos y Selección de Variables
│   ├── 02_preparation.ipynb   # Split Train/Test, Entrenamiento, CV y Regularización
│   └── 03_visualization.ipynb # Comparación de Modelos, Visualización de Resultados y Conclusiones
├── src/                       # Código fuente de Python (.py) reusable y modularizado
│   ├── __init__.py            # Inicializador de paquete Python
│   ├── preparation.py         # Funciones de validación de split y construcción de Pipelines
│   ├── features.py            # Selección de variables de interés
│   └── plots.py               # Funciones de visualización (EDA, predicciones y métricas)
├── models/                    # Modelos entrenados guardados como archivos .pkl serializados
│   ├── baseline_model.pkl
│   ├── multiple_linear_model.pkl
│   ├── polynomial_linear_model.pkl
│   ├── ridge_model.pkl
│   ├── lasso_model.pkl
│   └── optimized_best_model.pkl
├── outputs/                   # Salidas generadas por el proyecto
│   ├── figures/               # Gráficos de salida (PNG/PDF)
│   └── reports/               # Reportes y CSVs de métricas finales
├── config/                    # Directorio para configuración (parámetros/rutas)
├── .gitignore                 # Archivos que Git no debe rastrear (datos, modelos, entorno virtual)
├── requirements.txt           # Dependencias necesarias para ejecutar el proyecto
└── README.md                  # Documentación principal del proyecto
```

---

## Ejecución del Proyecto

Para reproducir el flujo completo del proyecto, ejecute los notebooks en el siguiente orden secuencial:

1. **`notebooks/01_eda.ipynb`**:
   - Ingiere el dataset original inmutable desde `data/raw/`.
   - Realiza un análisis exploratorio (EDA) de tipos, nulos, distribuciones y relaciones categóricas/numéricas.
   - Aplica pruebas de hipótesis (ANOVA, Levene) y correlación de Pearson para seleccionar variables críticas.
   - Guarda el conjunto de datos filtrado en `data/processed/ai_student_impact_procesado.csv`.

2. **`notebooks/02_preparation.ipynb`**:
   - Carga el dataset procesado y realiza la división en entrenamiento (Train) y prueba (Test).
   - Valida estadísticamente que no haya sesgos en el split (medias, rangos y varianzas).
   - Entrena los modelos: Baseline (Simple), Lineal Múltiple, Polinomial (Grado 2), Ridge y Lasso.
   - Evalúa predicciones out-of-fold mediante `cross_val_predict()`.
   - Optimiza hiperparámetros ($\alpha$ y grado polinomial) con `GridSearchCV`.
   - Serializa y guarda todos los modelos en `models/`.

3. **`notebooks/03_visualization.ipynb`**:
   - Carga los modelos entrenados desde `models/` y evalúa sus predicciones finales en el conjunto de prueba independiente ($Test$).
   - Genera y guarda la tabla comparativa de métricas en `outputs/reports/metrics_comparison.csv`.
   - Guarda los gráficos comparativos y de diagnóstico final en `outputs/figures/`.

---

## Resultados y Conclusiones

Los modelos de regresión lineal múltiple y polinomial de grado 2 demostraron una excelente capacidad predictiva sobre el rendimiento académico posterior (`Post_Semester_GPA`):

- **Baseline (Simple)**: $R^2 \approx 0.859$, RMSE $\approx 0.184$
- **Regresión Lineal Múltiple**: $R^2 \approx 0.887$, RMSE $\approx 0.165$
- **Regresión Polinomial (Grado 2)**: $R^2 \approx 0.891$, RMSE $\approx 0.162$
- **Modelo Ridge Ganador ($\alpha=10.0$)**: $R^2 \approx 0.892$, RMSE $\approx 0.161$

La sintonización automatizada mediante `GridSearchCV` identificó que un polinomio de segundo grado con regularización Ridge ligera logra la mejor capacidad de generalización en el conjunto de prueba, controlando eficazmente el riesgo de sobreajuste.
