## **Proyecto de Predicción de Supervivencia y Clasificación de Imágenes de Gladiadores**

Este repositorio contiene scripts para el procesamiento de datos, entrenamiento de modelos de Machine Learning (clasificación y clustering), y una aplicación Streamlit para predecir la supervivencia de gladiadores y clasificar imágenes.

### **Descripción del Proyecto**

El objetivo de este proyecto es explorar y predecir la supervivencia de gladiadores basándose en diversas características, así como clasificar imágenes para identificar si contienen gladiadores. Se utilizan técnicas de Machine Learning supervisado y no supervisado para lograr estos objetivos.

### **Estructura del Repositorio**

La estructura del proyecto sigue una convención común para proyectos de Machine Learning:

REPO_GLADIADORES_ML/
├── data/
│   ├── raw/                  # Datos brutos originales (ej. gladiator_data.csv)
│   ├── processed/            # Datos limpios y preprocesados
│   ├── train/                # Datos de entrenamiento
│   ├── test/                 # Datos de prueba
│   ├── imagenes_gladiadores/ # Conjunto de datos de imágenes para VGG16 (estructurado por clases)
│   └── predict/              # Imágenes para pruebas de predicción manual
├── models/                   # Modelos entrenados (RFC, VGG16, KMeans, Scaler)
├── app_streamlit/            # Aplicación Streamlit
│   └── app.py
│   └── image_34e09b.png      # Imagen del logo/banner de la app
├── .gitignore                # Archivos y directorios a ignorar por Git
├── dataprocessing.py         # Script para el procesamiento de datos
├── training.py               # Script para la división de datos de entrenamiento/prueba
├── evaluation.py             # Script para la evaluación de múltiples modelos de clasificación
├── main.py                   # Script principal para el modelo de clasificación (Random Forest)
├── main_images.py            # Script para el entrenamiento y evaluación del modelo VGG16
├── 03_Entrenamiento_Evaluacion_kmeans_NO_SUPERVISADO.ipynb # Cuaderno Jupyter para K-Means
└── README.md                 # Este archivo

## **Scripts y su Funcionalidad**

A continuación, se describe la función de cada script principal en el repositorio:

<span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">dataprocessing.py</span>
Este script es responsable de la preparación inicial de los datos. Realiza las siguientes tareas:

Carga el archivo de datos brutos <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">(gladiator_data.csv)</span>.

Realiza limpieza de datos, manejo de valores nulos y transformaciones de características.

Crea nuevas características (feature engineering), como <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">WinLossRatio, BMI</span>, y productos de interacción entre características.

Codifica variables categóricas utilizando <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">pd.get_dummies</span>.

Genera y visualiza matrices de correlación para entender las relaciones entre las características y la variable objetivo <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">Survived</span>.

Selecciona un subconjunto final de características relevantes.

Guarda el DataFrame procesado en <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">../data/processed/gladiador_data_procesado.csv</span>.

<span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">training.py</span>
Este script se encarga de preparar los conjuntos de datos para el entrenamiento y la prueba:

Carga los datos procesados.

Divide el conjunto de datos en características <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">(X)</span> y la variable objetivo <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">(y)</span>.

Realiza la división de los datos en conjuntos de entrenamiento y prueba (X_train, X_test, y_train, y_test).

Guarda los conjuntos de entrenamiento y prueba resultantes en archivos CSV separados en las carpetas ../data/train/ y ../data/test/ respectivamente.

<span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">evaluation.py</span>
Este script se enfoca en la evaluación comparativa y la optimización de varios modelos de clasificación:

Carga los datos procesados.

Define y entrena múltiples modelos de clasificación (Bagging Classifier, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Regresión Logística, Árbol de Decisión).

Evalúa el rendimiento inicial de cada modelo utilizando validación cruzada (K-Fold).

Realiza la hiperparametrización utilizando <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">GridSearchCV</span> para encontrar los mejores parámetros para el modelo Random Forest.

Define un <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">Pipeline</span> para encapsular el escalado y el modelo, y realiza una búsqueda de hiperparámetros más exhaustiva sobre diferentes modelos y escaladores.

Imprime el mejor score y el mejor estimador encontrado por <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">GridSearchCV</span>.

Guarda el mejor modelo (pipeline) entrenado en <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">../models/best_gladiator_survival_model.pkl</span>.

<span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">main.py</span>
Este script es la parte principal para la predicción de supervivencia de gladiadores:

Carga el modelo <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">RandomForestClassifier</span> previamente entrenado.

Realiza predicciones sobre el conjunto de prueba.

Calcula y muestra métricas de clasificación clave: precisión (accuracy), precisión (precision), sensibilidad (recall), puntuación F1 y ROC AUC.

Genera visualizaciones importantes:

Matriz de Confusión: Para evaluar el rendimiento del modelo en términos de verdaderos/falsos positivos/negativos.

Curva ROC y AUC: Para visualizar la capacidad del modelo para distinguir entre clases.

Importancia de las Características: Un gráfico de barras que muestra la influencia de cada característica en las predicciones del modelo.

Distribución de Probabilidades Predichas: Histograma de las probabilidades de supervivencia por clase real.

Guarda el modelo <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">rfc_model_final.pkl</span> en la carpeta <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">../models/</span>.

<span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">main_images.py</span>
Este script se dedica al entrenamiento y evaluación de un modelo de clasificación de imágenes:

Utiliza una arquitectura VGG16 pre-entrenada para Transfer Learning.

Carga un conjunto de datos de imágenes de gladiadores desde <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">../data/imagenes_gladiadores</span>.

Aplica técnicas de aumento de datos <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">(RandomFlip, RandomRotation, RandomZoom, RandomContrast)</span> para mejorar la robustez del modelo.

Define y compila el modelo VGG16 con capas personalizadas para la clasificación binaria.

Entrena el modelo con <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">EarlyStopping</span> para prevenir el sobreajuste.

Evalúa el rendimiento final del modelo en el conjunto de validación.

Genera gráficos del historial de entrenamiento (precisión y pérdida a lo largo de las épocas).

Guarda el modelo VGG16 entrenado como <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">mi_modelo_VGG16.keras</span> en la carpeta <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">models/</span>.

Incluye una función para preprocesar una sola imagen y realizar una predicción manual, mostrando el resultado y la imagen.

<span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">03_Entrenamiento_Evaluacion_kmeans_NO_SUPERVISADO.ipynb</span>
Este cuaderno de Jupyter explora el clustering no supervisado utilizando el algoritmo K-Means:

Carga los datos procesados de gladiadores.

Aplica el algoritmo K-Means para agrupar los datos en 2 clústeres.

Realiza el escalado de características <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">(StandardScaler)</span> antes de aplicar K-Means, lo cual es crucial para algoritmos basados en distancia.

Calcula y muestra el Coeficiente de Silueta para evaluar la calidad del clustering.

Interpreta los clústeres mostrando los valores promedio de cada característica dentro de cada grupo, lo que permite caracterizar qué tipo de gladiadores pertenecen a cada clúster.

Visualiza la distribución de los clústeres con un gráfico de pastel y un gráfico de dispersión (Wins vs. Public Favor) coloreado por clúster y estilo por supervivencia.

Guarda el modelo K-Means y el escalador utilizado en la carpeta <span style="background-color: #FFFF00; padding: 2px 4px; border-radius: 3px;">../models/</span>.

## **Requisitos**
* Python 3.8+

**Autor**: Yolanda Pérez San Segundo
**Fecha**: Julio 2025