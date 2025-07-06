Proyecto de Predicción de Supervivencia y Clasificación de Imágenes de Gladiadores
Este repositorio contiene scripts para el procesamiento de datos, entrenamiento de modelos de Machine Learning (clasificación y clustering), y una aplicación Streamlit para predecir la supervivencia de gladiadores y clasificar imágenes.

Descripción del Proyecto
El objetivo de este proyecto es explorar y predecir la supervivencia de gladiadores basándose en diversas características, así como clasificar imágenes para identificar si contienen gladiadores. Se utilizan técnicas de Machine Learning supervisado y no supervisado para lograr estos objetivos.

Estructura del Repositorio
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

Scripts y su Funcionalidad
A continuación, se describe la función de cada script principal en el repositorio:

dataprocessing.py
Este script es responsable de la preparación inicial de los datos. Realiza las siguientes tareas:

Carga el archivo de datos brutos (gladiator_data.csv).

Realiza limpieza de datos, manejo de valores nulos y transformaciones de características.

Crea nuevas características (feature engineering), como WinLossRatio, BMI, y productos de interacción entre características.

Codifica variables categóricas utilizando pd.get_dummies.

Genera y visualiza matrices de correlación para entender las relaciones entre las características y la variable objetivo Survived.

Selecciona un subconjunto final de características relevantes.

Guarda el DataFrame procesado en ../data/processed/gladiador_data_procesado.csv.

training.py
Este script se encarga de preparar los conjuntos de datos para el entrenamiento y la prueba:

Carga los datos procesados.

Divide el conjunto de datos en características (X) y la variable objetivo (y).

Realiza la división de los datos en conjuntos de entrenamiento y prueba (X_train, X_test, y_train, y_test).

Guarda los conjuntos de entrenamiento y prueba resultantes en archivos CSV separados en las carpetas ../data/train/ y ../data/test/ respectivamente.

evaluation.py
Este script se enfoca en la evaluación comparativa y la optimización de varios modelos de clasificación:

Carga los datos procesados.

Define y entrena múltiples modelos de clasificación (Bagging Classifier, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Regresión Logística, Árbol de Decisión).

Evalúa el rendimiento inicial de cada modelo utilizando validación cruzada (K-Fold).

Realiza la hiperparametrización utilizando GridSearchCV para encontrar los mejores parámetros para el modelo Random Forest.

Define un Pipeline para encapsular el escalado y el modelo, y realiza una búsqueda de hiperparámetros más exhaustiva sobre diferentes modelos y escaladores.

Imprime el mejor score y el mejor estimador encontrado por GridSearchCV.

Guarda el mejor modelo (pipeline) entrenado en ../models/best_gladiator_survival_model.pkl.

main.py
Este script es la parte principal para la predicción de supervivencia de gladiadores:

Carga el modelo RandomForestClassifier previamente entrenado.

Realiza predicciones sobre el conjunto de prueba.

Calcula y muestra métricas de clasificación clave: precisión (accuracy), precisión (precision), sensibilidad (recall), puntuación F1 y ROC AUC.

Genera visualizaciones importantes:

Matriz de Confusión: Para evaluar el rendimiento del modelo en términos de verdaderos/falsos positivos/negativos.

Curva ROC y AUC: Para visualizar la capacidad del modelo para distinguir entre clases.

Importancia de las Características: Un gráfico de barras que muestra la influencia de cada característica en las predicciones del modelo.

Distribución de Probabilidades Predichas: Histograma de las probabilidades de supervivencia por clase real.

Guarda el modelo rfc_model_final.pkl en la carpeta ../models/.

main_images.py
Este script se dedica al entrenamiento y evaluación de un modelo de clasificación de imágenes:

Utiliza una arquitectura VGG16 pre-entrenada para Transfer Learning.

Carga un conjunto de datos de imágenes de gladiadores desde ../data/imagenes_gladiadores.

Aplica técnicas de aumento de datos (RandomFlip, RandomRotation, RandomZoom, RandomContrast) para mejorar la robustez del modelo.

Define y compila el modelo VGG16 con capas personalizadas para la clasificación binaria.

Entrena el modelo con EarlyStopping para prevenir el sobreajuste.

Evalúa el rendimiento final del modelo en el conjunto de validación.

Genera gráficos del historial de entrenamiento (precisión y pérdida a lo largo de las épocas).

Guarda el modelo VGG16 entrenado como mi_modelo_VGG16.keras en la carpeta models/.

Incluye una función para preprocesar una sola imagen y realizar una predicción manual, mostrando el resultado y la imagen.

03_Entrenamiento_Evaluacion_kmeans_NO_SUPERVISADO.ipynb
Este cuaderno de Jupyter explora el clustering no supervisado utilizando el algoritmo K-Means:

Carga los datos procesados de gladiadores.

Aplica el algoritmo K-Means para agrupar los datos en 2 clústeres.

Realiza el escalado de características (StandardScaler) antes de aplicar K-Means, lo cual es crucial para algoritmos basados en distancia.

Calcula y muestra el Coeficiente de Silueta para evaluar la calidad del clustering.

Interpreta los clústeres mostrando los valores promedio de cada característica dentro de cada grupo, lo que permite caracterizar qué tipo de gladiadores pertenecen a cada clúster.

Visualiza la distribución de los clústeres con un gráfico de pastel y un gráfico de dispersión (Wins vs. Public Favor) coloreado por clúster y estilo por supervivencia.

Guarda el modelo K-Means y el escalador utilizado en la carpeta ../models/.

Requisitos
Python 3.8+

Autor: Yolanda Pérez San Segundo
Fecha: Julio 2025