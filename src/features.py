import pandas as pd

def select_features(df):
    """
    Selecciona las variables predictoras clave y la variable objetivo
    según el análisis estadístico del Módulo 1.
    """
    variables_modelo = [
        'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours',
        'Anxiety_Level_During_Exams', 'Tool_Diversity', 'Perceived_AI_Dependency',
        'Major_Category', 'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Paid_Subscription',
        'Post_Semester_GPA'
    ]
    
    # Nos aseguramos de retornar una copia limpia
    return df[variables_modelo].copy()
