"""
Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. 
(Portafolio Implementación) [semana 3]

Miguel Ángel Bermea Rodríguez | A01411671

Algoritmo: Logistic Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, learning_rate = 0.1, max_iter = 100, validation_freq = 10, patience = 5):
        """
        Inicializa el modelo de regresión logística con early stopping.

        :param learning_rate: Tasa de aprendizaje para el descenso de gradiente.
        :param max_iter: Número máximo de iteraciones permitidas.
        :param validation_freq: Frecuencia de validación para early stopping.
        :param patience: Paciencia para early stopping (número de iteraciones sin mejora).
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.validation_freq = validation_freq
        self.patience = patience
        self.weights = None
        self.bias = None
        self.losses = [] # Variable para almacenar valor de pérdida en cada iteración

    def sigmoid(self, z):
        # Mapeo de cada dato a un valor entre 0 y 1
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, X_val, y_val):
        """
        Entrena el modelo de regresión logística con early stopping.

        :param X: Datos de entrenamiento.
        :param y: Etiquetas de entrenamiento.
        :param X_val: Datos de validación.
        :param y_val: Etiquetas de validación.
        """

        # Número de samples y features en el dataset
        n_samples, n_features = X.shape

        # Inicialización de pesos y sesgo
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Variables para early stopping
        best_loss = np.inf
        best_weights = self.weights
        best_bias = self.bias
        best_iter = 0
        no_improvement_count = 0

        # Gradient descent para actualizar pesos y sesgo
        for i in range(self.max_iter):
            # Modelo lineal
            linear_model = np.dot(X, self.weights) + self.bias

            # Función sigmoide para obtener probabilidades predichas
            y_predicted = self.sigmoid(linear_model)

            # Cálculo de gradientes para pesos y sesgo
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) # Pesos
            db = (1 / n_samples) * np.sum(y_predicted - y) # Sesgo

            # Actualizar pesos y sesgo con descenso de gradiente
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Early stopping 
            if i % self.validation_freq == 0:
                y_val_predicted = self.sigmoid(np.dot(X_val, self.weights) + self.bias)

                # Cálculo de pérdida
                loss = -np.mean(y_val * np.log(y_val_predicted) + (1 - y_val) * np.log(1 - y_val_predicted))

                # Reducción del error en cada iteración
                print(f'Iteration {i + 1}: Loss = {loss:.6f}')

                self.losses.append(loss)

                if loss < best_loss:
                    best_loss = loss
                    best_weights = self.weights
                    best_bias = self.bias
                    best_iter = i
                    no_improvement_count = 0
                else:
                    # Aumenta el contador en caso de que no haya habido mejoría
                    no_improvement_count += 1

                if no_improvement_count >= self.patience:
                    # Se detiene si el contador sobrepasa la paciencia establecida (limite de iteraciones sin mejoría)
                    print(f'Early stopping on iteration {i}, with validation loss {best_loss}')
                    break

                # Restaurar los mejores pesos y bias encontrados durante el entrenamiento
                self.weights = best_weights
                self.bias = best_bias


    def predict(self, X):
        # Modelo lineal
        linear_model = np.dot(X, self.weights) + self.bias

        # Función sigmoide para obtener probabilidades predichas -> Labels binarios
        y_predicted_cls = [1 if i > 0.5 else 0 for i in self.sigmoid(linear_model)]
        return y_predicted_cls
    
    def accuracy(self, X, y):
        # Cálculo de precisión de las prediciones del modelo (dado un dataset)
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

# Lectura de dataset (proveniente de Kaggle)
data = pd.read_csv('data.csv', index_col = 0)

# Head de data
# data.head()

# Summary de data
# data.info()

# Eliminar columna 'Unnamed: 32' - Tiene valores faltantes
data = data.drop('Unnamed: 32', axis = 1)

# Tipos de datos de columnas
# data.dtypes

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

# Aplicación de modelo en datos de entrenamiento + datos de validación y datos de prueba
model = LogisticRegression()
model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
accuracy = model.accuracy(X_test, y_test)
accuracy_percentage = accuracy * 100

print(f'\nAccuracy: {accuracy_percentage}%')

"""
En caso de obtener accuracy: 100% -> Modelo predijo todas las labels del dataset de prueba.

Observaciones:
- Esto no significa que el modelo siempre va a hacer predicciones perfectas en datos nuevos.
- Ya que la accuracy depende de las samples de entrenamiento y de prueba propocionadas.
- Al igual que los parametros de tasa de aprendizaje y número de iteraciones aplicado al Gradient Descent.
"""

# Plot de progreso de entrenamiento reduciendo el error
plt.plot(model.losses)
plt.title("Progreso de entrenamiento reduciendo el error")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()