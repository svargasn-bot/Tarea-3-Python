from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def _ensure_parent_dir(filepath):
    if filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def plot_eda_regplots(df, filepath=None):
    """
    Dibuja gráficos de regresión lineal simple (regplot) para GPA previo,
    horas semanales de GenAI y horas de estudio tradicional contra el GPA posterior.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    df_sample = df.sample(min(2000, len(df)), random_state=42)
    
    # GPA Previo vs GPA Posterior
    sns.regplot(data=df_sample, x='Pre_Semester_GPA', y='Post_Semester_GPA', 
                ax=axes[0], scatter_kws={'alpha': 0.15}, line_kws={'color': 'red'})
    axes[0].set_title('GPA Previo vs GPA Posterior')
    
    # Horas de IA vs GPA Posterior
    sns.regplot(data=df_sample, x='Weekly_GenAI_Hours', y='Post_Semester_GPA', 
                ax=axes[1], scatter_kws={'alpha': 0.15}, line_kws={'color': 'red'})
    axes[1].set_title('Horas Semanales de IA vs GPA Posterior')
    
    # Horas de Estudio Tradicional vs GPA Posterior
    sns.regplot(data=df_sample, x='Traditional_Study_Hours', y='Post_Semester_GPA', 
                ax=axes[2], scatter_kws={'alpha': 0.15}, line_kws={'color': 'red'})
    axes[2].set_title('Estudio Tradicional vs GPA Posterior')
    
    plt.tight_layout()
    if filepath:
        _ensure_parent_dir(filepath)
        plt.savefig(filepath, dpi=300)
    plt.show()

def plot_eda_boxplots(df, filepath=None):
    """
    Dibuja boxplots para Habilidad de Prompting, Carrera (Major) y
    Caso de Uso Principal contra el GPA posterior.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Habilidad de Prompting vs GPA Posterior
    sns.boxplot(data=df, x='Prompt_Engineering_Skill', y='Post_Semester_GPA', 
                ax=axes[0], order=['Beginner', 'Intermediate', 'Advanced'])
    axes[0].set_title('Prompting Skill vs GPA')
    
    # Categoría de Carrera (Major) vs GPA Posterior
    sns.boxplot(data=df, x='Major_Category', y='Post_Semester_GPA', ax=axes[1])
    axes[1].set_title('Carrera (Major) vs GPA')
    axes[1].tick_params(axis='x', rotation=30)
    
    # Caso de Uso Principal vs GPA Posterior
    sns.boxplot(data=df, x='Primary_Use_Case', y='Post_Semester_GPA', ax=axes[2])
    axes[2].set_title('Caso de Uso Principal vs GPA')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if filepath:
        _ensure_parent_dir(filepath)
        plt.savefig(filepath, dpi=300)
    plt.show()

def plot_pivot_heatmap(df, filepath=None):
    """
    Genera un mapa de calor sobre el promedio de horas semanales de IA por carrera y caso de uso.
    """
    pivot = df.pivot_table(index='Major_Category', columns='Primary_Use_Case', 
                           values='Weekly_GenAI_Hours', aggfunc='mean')
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Uso de IA (Horas) por Carrera y Caso de Uso')
    plt.tight_layout()
    if filepath:
        _ensure_parent_dir(filepath)
        plt.savefig(filepath, dpi=300)
    plt.show()

def plot_correlation_matrix(df, columns, filepath=None):
    """
    Dibuja la matriz de correlación de Pearson para variables numéricas.
    """
    corr_matrix = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', vmin=-1, vmax=1)
    plt.title('Matriz de Correlación entre Variables Numéricas')
    plt.tight_layout()
    if filepath:
        _ensure_parent_dir(filepath)
        plt.savefig(filepath, dpi=300)
    plt.show()

def plot_residuals_distribution(y_test, y_pred_multi, y_pred_poly, filepath=None):
    """
    Grafica la densidad KDE de la distribución de residuos para los modelos
    de regresión lineal múltiple y regresión polinomial.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    residuos_multi = y_test - y_pred_multi
    residuos_poly = y_test - y_pred_poly
    
    sns.kdeplot(residuos_multi, label='Lineal Múltiple', ax=axes[0], fill=True, alpha=0.3)
    sns.kdeplot(residuos_poly, label='Polinomial Grado 2', ax=axes[0], fill=True, alpha=0.3)
    axes[0].set_title('Distribución de Residuos (Errores)')
    axes[0].legend()
    
    axes[1].scatter(y_test, y_pred_poly, alpha=0.1, color='purple', label='Polinomial')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel('Valores Reales (Post_Semester_GPA)')
    axes[1].set_ylabel('Valores Predichos')
    axes[1].set_title('Predicción vs Realidad (Polinomial Grado 2)')
    axes[1].legend()
    
    plt.tight_layout()
    if filepath:
        _ensure_parent_dir(filepath)
        plt.savefig(filepath, dpi=300)
    plt.show()

def plot_predictions_vs_real(y_test, y_pred, title, filepath=None):
    """
    Grafica los valores reales vs predichos con una línea de referencia y = x.
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.1, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(title)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.tight_layout()
    if filepath:
        _ensure_parent_dir(filepath)
        plt.savefig(filepath, dpi=300)
    plt.show()

def plot_cv_predictions(y_train, y_cv_base, y_cv_multi, y_cv_poly, 
                        r2_base, rmse_base, r2_multi, rmse_multi, r2_poly, rmse_poly, 
                        filepath=None):
    """
    Muestra un panel de 3 gráficos con valores reales vs predicciones out-of-fold para
    Baseline, Lineal Múltiple y Polinomial G2.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Baseline
    axes[0].scatter(y_train, y_cv_base, alpha=0.08, color='blue')
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0].set_title(f'Baseline (CV)\nR²: {r2_base:.4f} | RMSE: {rmse_base:.4f}')
    axes[0].set_xlabel('Valores Reales (GPA)')
    axes[0].set_ylabel('Predicciones CV')
    
    # Múltiple
    axes[1].scatter(y_train, y_cv_multi, alpha=0.08, color='orange')
    axes[1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[1].set_title(f'L. Múltiple (CV)\nR²: {r2_multi:.4f} | RMSE: {rmse_multi:.4f}')
    axes[1].set_xlabel('Valores Reales (GPA)')
    axes[1].set_ylabel('Predicciones CV')
    
    # Polinomial
    axes[2].scatter(y_train, y_cv_poly, alpha=0.08, color='purple')
    axes[2].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[2].set_title(f'Polinomial G2 (CV)\nR²: {r2_poly:.4f} | RMSE: {rmse_poly:.4f}')
    axes[2].set_xlabel('Valores Reales (GPA)')
    axes[2].set_ylabel('Predicciones CV')
    
    plt.suptitle('Diagnóstico de Predicciones Out-of-Fold (Validación Cruzada)', fontsize=14, y=1.03)
    plt.tight_layout()
    if filepath:
        _ensure_parent_dir(filepath)
        plt.savefig(filepath, dpi=300)
    plt.show()

def plot_model_comparison(df_comparacion, filepath=None):
    """
    Dibuja un gráfico de barras comparando el coeficiente R² en Test de todos los modelos.
    """
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_comparacion, x='R² Test', y='Modelo', hue='Modelo', legend=False, palette='viridis')
    plt.xlim(df_comparacion['R² Test'].min() - 0.05, df_comparacion['R² Test'].max() + 0.02)
    plt.title('Comparación del Coeficiente de Determinación (R²) en Test')
    plt.xlabel('R² (Mayor es mejor)')
    plt.ylabel('Modelo')
    plt.tight_layout()
    if filepath:
        _ensure_parent_dir(filepath)
        plt.savefig(filepath, dpi=300)
    plt.show()
