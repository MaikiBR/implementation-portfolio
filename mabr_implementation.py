"""
Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. 
(Portafolio Implementación) [semana 3]

Miguel Ángel Bermea Rodríguez | A01411671

Algoritmo: Logistic Regression
"""

import numpy as np
from sklearn.datasets import load_iris;
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, learning_rate = 0.1, max_iter = 100):
        # Tasa de aprendizaje y número máximo de iteraciones para el gradient descent
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def sigmoid(self, z):
        # Mapeo de cada dato a un valor entre 0 y 1
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # Número de samples y features en el dataset
        n_samples, n_features = X.shape

        # Inicialización de pesos y sesgo
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent para actualizar pesos y sesgo
        for _ in range(self.max_iter):
            # Modelo lineal
            linear_model = np.dot(X, self.weights) + self.bias

            # Función sigmoide para obtener probabilidades predichas
            y_predicted = self.sigmoid(linear_model)

            # Gradients para pesos y sesgo
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) # Pesos
            db = (1 / n_samples) * np.sum(y_predicted - y) # Sesgo

            # Actualizar pesos y sesgo utilizando los gradients correspondientes
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Modelo lineal
        linear_model = np.dot(X, self.weights) + self.bias

        # Función sigmoide para obtener probabilidades predichas
        y_predicted = self.sigmoid(linear_model)

        # Probabilidades predichas -> Labels binarios
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls
    
    def accuracy(self, X, y):
        # Cálculo de precisión de las prediciones del modelo (dado un dataset)
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = model.accuracy(X_test, y_test)
accuracy_percentage = accuracy * 100

"""
En caso de obtener accuracy: 100% -> Modelo predijo todas las labels del dataset de prueba.

Observaciones:
- Esto no significa que el modelo siempre va a hacer predicciones perfectas en datos nuevos.
- Ya que la accuracy depende de las samples de entrenamiento y de prueba propocionadas.
- Al igual que los parametros de tasa de aprendizaje y número de iteraciones aplicado al Gradient Descent.
"""

print(f'Accuracy: {accuracy_percentage}%')
