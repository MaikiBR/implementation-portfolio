# Archivos en el repositorio

* **mabr_implementation.py** | *Archivo de implementación de modelo de regresión logística sin framework*
* **mabr_implementation_framework.py** | *Archivo de implementación de modelo de regresión logística con framework*
* **data.csv** | *Dataset utilizado | Breast Cancer Wisconsin (Diagnostic)*
  
**Fuente del dataset:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?datasetId=180&searchQuery=logistic+

**Implementacion seleccionada para análisis:** Modelo de regresión logística sin framework

# Separación y evaluación del modelo con un conjunto de prueba y un conjunto de validación (Train/Test/Validation).

El modelo de regresión logística se entrenó y evaluó utilizando tres conjuntos de datos: entrenamiento (Train), validación (Validation) y prueba (Test). Los datos se dividieron de la siguiente manera:

- **Datos de entrenamiento:** 60% del conjunto de datos original.
- **Datos de validación:** 20% del conjunto de datos original.
- **Datos de prueba:** 20% del conjunto de datos original.

# Diagnóstico y explicación el grado de bias o sesgo: bajo medio alto 

El grado de bias (sesgo) de un modelo se refiere a su capacidad para capturar las relaciones subyacentes en los datos. En este caso, el modelo de regresión logística parece tener un **bajo sesgo**, ya que logra una precisión en los datos de prueba de aproximadamente 94.74%. Esto indica que el modelo puede capturar las relaciones entre las características de entrada (features) y las etiquetas (labels) de manera efectiva.

# Diagnóstico y explicación el grado de varianza: bajo medio alto

El grado de varianza de un modelo se refiere a su capacidad para generalizar a nuevos datos no vistos. Un modelo con alta varianza puede sobreajustarse a los datos de entrenamiento y tener un desempeño deficiente en datos de prueba. En este caso, el modelo parece tener una **varianza moderada**, ya que la precisión en los datos de prueba es coherente con la precisión en los datos de validación. No se observa un alto grado de sobreajuste.

# Diagnóstico y explicación el nivel de ajuste del modelo: underfitt fitt overfitt

El nivel de ajuste de un modelo se refiere a si el modelo está subajustado (underfitting), adecuadamente ajustado (fitting) o sobreajustado (overfitting). En este caso, el modelo parece estar **adecuadamente ajustado**, ya que logra una buena precisión tanto en los datos de entrenamiento como en los datos de prueba. Esto se refleja en la gráfica de progreso de entrenamiento, donde la pérdida disminuye de manera constante durante las iteraciones, sin signos de sobreajuste o subajuste extremo.

<a href="https://ibb.co/fGfT9NT"><img src="https://i.ibb.co/BKbYG6Y/Progreso-de-entrenamiento-reduciendo-el-error.png" alt="Progreso-de-entrenamiento-reduciendo-el-error" border="0"></a>

# Mejora del desempeño del modelo

Para mejorar aún más el desempeño del modelo, se pueden explorar las siguientes estrategias:

- **Regularización:** Se puede implementar la regularización L1 o L2 en el modelo para evitar el posible sobreajuste. Esto ayudaría a controlar el grado de varianza y mejorar la generalización del modelo.

- **Ajuste de Hiperparámetros:** Se pueden ajustar los hiperparámetros del modelo, como la tasa de aprendizaje (*learning_rate*), el número máximo de iteraciones (*max_iter*) y la paciencia (patience) para el early stopping. Experimentar con diferentes valores de estos hiperparámetros puede ayudar a encontrar una configuración que mejore el desempeño. **[Ya se utilizó]**

- **Selección de características:** Realizar un análisis de importancia de características para identificar cuáles tienen un mayor impacto en las predicciones y considerar la eliminación de características irrelevantes o altamente correlacionadas. **[Ya se utilizó] Pero en caso de ser necesario se puede mostrar la matriz de correlación**
