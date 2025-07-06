import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from skimage.io import imread 
from skimage.transform import resize 
import matplotlib.pyplot as plt
import numpy as np
import os

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = r'..\\data\imagenes_gladiadores' 
NUM_CLASSES = 1 # Para clasificación binaria

#Cargo el Dataset de Imágenes 
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred", 
    label_mode='binary', 
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred", 
    label_mode='binary', 
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
print(f"Clases encontradas: {train_ds.class_names}")

#Optimizo el rendimiento de la carga de datos
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#Capas de Preprocesamiento y Aumento de Datos 
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomContrast(0.2)
])

#Para VGG16:
preprocess_input = tf.keras.applications.vgg16.preprocess_input

#Definición del Modelo 
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

for layer in base_model.layers:
    layer.trainable = False

inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)

x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
x = Dropout(0.6)(x)

predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=predictions)

#Compilación del Modelo
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
print("Modelo CNN (VGG16) construido y compilado correctamente para clasificación binaria.")

#Entrenamiento del Modelo (con Early Stopping)
EPOCHS = 20
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

print(f"\nIniciando el entrenamiento del modelo por un máximo de {EPOCHS} épocas con Early Stopping...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[early_stopping]
)
print("\nEntrenamiento completado (o detenido por Early Stopping).")

#Evaluación Final del Modelo
print("\nEvaluando el modelo en el conjunto de validación (como conjunto de prueba)...")
loss, accuracy = model.evaluate(val_ds)
print(f"Pérdida en el conjunto de prueba: {loss:.4f}")
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

#Graficar el historial de entrenamiento
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Defino la ruta y el nombre del archivo donde quiero guardar el modelo
output_directory = 'models'
model_filename = 'mi_modelo_VGG16.keras' # Nombre recomendado para modelos Keras
full_model_path = os.path.join(output_directory, model_filename)

#Me de que la carpeta de destino exista
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

#Guardo el modelo
try:
    model.save(full_model_path)
    print(f"Modelo guardado exitosamente en: {full_model_path}")
except Exception as e:
    print(f"Error al guardar el modelo: {e}")

CLASS_NAMES = ['No_Gladiador', 'Gladiador']

#FUNCIÓN NECESARIA PARA EL PREPROCESAMIENTO DE UNA SOLA IMAGEN
def preprocess_single_image(image_path, target_size, normalize_func=None):

    # Cargo la imagen
    img = imread(image_path)

    # Redimensiono la imagen
    # resize devuelve float en [0, 1] si el tipo de dato original era int.
    # Si la imagen es en escala de grises y se espera RGB, es necesario convertirla.
    if len(img.shape) == 2: # Si es escala de grises
        img = np.stack([img, img, img], axis=-1) # Convertir a RGB duplicando canales
    elif img.shape[2] == 4: # Si es RGBA
        img = img[:, :, :3] # Elimina el canal alfa

    # Asegurarse de que sea de tipo float antes de redimensionar si es necesario
    img_resized = resize(img, target_size)

    # Convertir a array de numpy y expandir dimensiones para el batch (1, height, width, channels)
    img_array = np.expand_dims(img_resized, axis=0)

    # Aplicar normalización específica de la arquitectura si se proporciona
    if normalize_func:
        # Asegúrate de que el tipo de dato sea compatible con la función de preprocesamiento
        # Las funciones de preprocesamiento de Keras a menudo esperan floats
        img_array = normalize_func(img_array * 255.0) # Multiplicar por 255 para llevar a 0-255 antes de preprocesar
    else:
        # Si no hay función de normalización específica, escalar a [0, 1] si los píxeles no están ya en ese rango
        if img_array.max() > 1.0:
            img_array /= 255.0

    return img_array

# Realizo la Predicción con una Imagen Introducida Manualmente 
print("\n--- ¡Listo para hacer una predicción con una nueva imagen! ---")
image_path_input = image_path_to_predict = r'..\\data\predict\gladiador.jpg'


try:
    # Preprocesar la imagen de entrada
    # Usamos preprocess_input de VGG16 (tf.keras.applications.vgg16.preprocess_input) que está definida en tu código principal.
    processed_image = preprocess_single_image(image_path_input, IMAGE_SIZE, normalize_func=preprocess_input)

    # Realizo la predicción utilizando el modelo cargado
    predictions = model.predict(processed_image)

    # Interpreto los resultados de la predicción
    # Para NUM_CLASSES = 1 y activación 'sigmoid', la salida es una probabilidad única.
    # Si la probabilidad es > 0.5, se considera de la clase 1, de lo contrario clase 0.
    # El índice de la clase predicha se obtiene así:
    predicted_class_index = (predictions[0] > 0.5).astype(int)[0] # Para salida sigmoide binaria
    confidence = predictions[0][0] # La probabilidad directa de la clase positiva (1)

    print("\n--- Resultado de la Predicción para la Nueva Imagen ---")
    print(f"El modelo predice que la imagen pertenece a la: **{CLASS_NAMES[predicted_class_index]}**")
    print(f"Con una confianza del: {confidence:.2%}")
    print(f"Probabilidad de la clase positiva: {predictions[0][0]:.4f}") # Mostrar solo la probabilidad de la clase positiva

    # Visualizao la imagen de entrada y su predicción 
    plt.figure(figsize=(10, 8))
    img_display = imread(image_path_input)
    plt.imshow(img_display)
    plt.title(f"Imagen de Entrada: {os.path.basename(image_path_input)}\nPredicción: {CLASS_NAMES[predicted_class_index]} (Confianza: {confidence:.2%})", fontsize=14)
    plt.axis('off')

except FileNotFoundError as e:
    print(f"\nError: {e}")
    print("Por favor, verifica que la ruta de la imagen que has introducido es correcta y que el archivo existe.")
except Exception as e:
    print(f"\nOcurrió un error inesperado durante el preprocesamiento o la predicción: {e}")
    print("Asegúrate de que la imagen es un formato compatible (ej. .jpg, .png) y no está corrupta.")

    

