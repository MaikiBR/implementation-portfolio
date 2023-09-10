"""
Momento de Retroalimentación: Módulo 2 Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. 
(Portafolio Implementación) [semana 5]

Miguel Ángel Bermea Rodríguez | A01411671

Algoritmo: Logistic Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Lectura de dataset (proveniente de Kaggle)
data = pd.read_csv('data.csv', index_col = 0)

# Eliminar la columna 'Unnamed: 32' - Tiene valores faltantes
data = data.drop('Unnamed: 32', axis = 1)

"""
Eliminar columnas innecesarias
"""
# Eliminar columnas "worst"
cols = ['radius_worst', 
        'texture_worst', 
        'perimeter_worst', 
        'area_worst', 
        'smoothness_worst', 
        'compactness_worst', 
        'concavity_worst',
        'concave points_worst', 
        'symmetry_worst', 
        'fractal_dimension_worst']

data = data.drop(cols, axis = 1)

# Eliminar columnas relacionadas con "perimeter" y "area"
cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']

data = data.drop(cols, axis = 1)

# Eliminar columnas relacionadas con "concavity" y "concave points"
cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']

data = data.drop(cols, axis = 1)

# Visualizar columnas restantes
print(data.columns)

# Visualización de datos
X = data.drop(['diagnosis'], axis = 1)
print("Data X: ", X)

y = data['diagnosis']
y = y.map({'M': 1, 'B': 0})
print("Data y: ", y)

# Separación de datos en datos de entrenamiento, datos de validación  y datos de prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalización de datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Entrenamiento de modelo de regresión logística (con framework)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Cálculo de precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100

print(f'Accuracy: {accuracy_percentage:.2f}%')

# Cálculo de pérdida de registro (log loss) en los datos de validación
y_val_pred_proba = model.predict_proba(X_val)[:, 1]  # Probabilidad de la clase positiva
logloss = log_loss(y_val, y_val_pred_proba)
print(f'Log Loss en los datos de validación: {logloss:.4f}')
