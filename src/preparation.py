import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def validate_split(y_train, y_test):
    """
    Realiza una batería de pruebas estadísticas para validar si los conjuntos de
    entrenamiento y prueba representan la misma distribución (homogeneidad).
    """
    print('=' * 80)
    print('BATERÍA DE PRUEBAS ESTADÍSTICAS PARA VALIDAR EL SPLIT')
    print('=' * 80)
    
    # 1. Test t de Welch (Comparación de medias)
    t_stat, p_val_t = stats.ttest_ind(y_train, y_test, equal_var=False)
    # 2. Test U de Mann-Whitney (Comparación de rangos/medianas)
    u_stat, p_val_u = stats.mannwhitneyu(y_train, y_test, alternative='two-sided')
    # 3. Test de Levene (Comparación de varianzas)
    lev_stat, p_val_l = stats.levene(y_train, y_test, center='median')
    
    print(f'1. Test t de Welch       | p-valor: {p_val_t:.4f} (t={t_stat:.2f})')
    print(f'2. Test U Mann-Whitney   | p-valor: {p_val_u:.4f} (U={u_stat:.2f})')
    print(f'3. Test de Levene        | p-valor: {p_val_l:.4f} (lev={lev_stat:.2f})')
    print('-' * 80)
    
    aprobadas = sum([p_val_t > 0.05, p_val_u > 0.05, p_val_l > 0.05])
    if aprobadas == 3:
        print('VEREDICTO: SPLIT EXITOSO Y REPRESENTATIVO (3/3).')
    else:
        print(f'VEREDICTO: PRECAUCIÓN. El split pasó {aprobadas}/3 criterios. Hay ligeras asimetrías.')
    print('=' * 80)
    
    return {
        'ttest_p': p_val_t,
        'mannwhitney_p': p_val_u,
        'levene_p': p_val_l,
        'aprobadas': aprobadas
    }

def create_preprocessor(variables_numericas, variables_categoricas, poly_degree=None):
    """
    Crea un ColumnTransformer que estandariza las variables numéricas
    (con opción a expansión polinomial) y aplica One-Hot Encoding a las categóricas.
    """
    if poly_degree is not None:
        num_transformer = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False))
        ])
    else:
        num_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, variables_numericas),
        ('cat', OneHotEncoder(drop='first'), variables_categoricas)
    ])
    
    return preprocessor
