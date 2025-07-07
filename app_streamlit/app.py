import streamlit as st
import pickle
import pandas as pd
import base64
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
import tensorflow as tf # Import TensorFlow for preprocess_input

# --- Funciones para el Fondo ---
# @st.cache_data
# def get_base64(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

def set_background(color):
    """
    Establece el color de fondo de la aplicación Streamlit.
    Acepta un nombre de color CSS o un código hexadecimal (ej. '#CD853F').
    """
    page_bg_img = f'''
    <style>
    .stApp {{
        background-color: {color};
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Configuración de la página
st.set_page_config(layout="centered")

# Llama a la función para establecer el fondo con el color deseado
set_background("#D8EB71") # Un color marrón anaranjado (Peru)

# Título de la aplicación
st.title('¿El Gladiador Sobrevivirá? Predicción de Supervivencia y Clasificación de Imágenes')
st.write('Introduce las características del gladiador para predecir si vivirá o morirá en la arena, o sube una imagen para clasificarla como gladiador.')

# --- Imagen Centrada en la Parte Superior (NUEVO) ---
# Puedes reemplazar esta URL con la URL de tu imagen real.
# Asegúrate de que la imagen sea accesible públicamente si no está en tu proyecto local.
image_url = "fondo_streamlit.png" # Ejemplo: Color marrón oscuro con texto blanco

# Para centrar la imagen, usamos columnas de Streamlit
col1, col2, col3 = st.columns([1, 4, 1]) # Proporciones de las columnas para centrar

with col2: # Coloca la imagen en la columna central
    st.image(image_url, caption='Imagen representativa de un gladiador', use_container_width=True)
st.markdown("---") # Una línea divisoria para separar la imagen del contenido siguiente

# --- Cargar el modelo de Random Forest (existente) ---
@st.cache_resource
def load_trained_rfc_model():
    model_path = "../models/rfc_model_final.pkl" # Asegúrate de que esta ruta sea correcta
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.sidebar.success(f"Modelo de Supervivencia cargado correctamente desde: {model_path}")
        return model
    except FileNotFoundError:
        st.error(f"Error: El archivo del modelo de Random Forest '{model_path}' no fue encontrado. Asegúrate de que la ruta sea correcta y el archivo exista.")
        st.stop() # Detiene la ejecución de la app si el modelo no se encuentra
    except Exception as e:
        st.error(f"Error al cargar el modelo de Random Forest: {e}")
        st.stop() # Detiene la ejecución si hay otro error al cargar

# Llama a la función para cargar el modelo de Random Forest
rfc_model = load_trained_rfc_model()

# --- Cargar el modelo VGG16 para clasificación de imágenes (nuevo) ---
@st.cache_resource
def load_vgg16_model():
    # Asegúrate de que esta ruta sea correcta para tu modelo VGG16
    vgg16_model_path = "../models/mi_modelo_VGG16.keras" # Actualizado para coincidir con el script de entrenamiento
    try:
        # Cargar el modelo VGG16 pre-entrenado y fine-tuned
        model = load_model(vgg16_model_path)
        st.sidebar.success(f"Modelo VGG16 cargado correctamente desde: {vgg16_model_path}")
        return model
    except FileNotFoundError:
        st.error(f"Error: El archivo del modelo VGG16 '{vgg16_model_path}' no fue encontrado. Asegúrate de que la ruta sea correcta y el archivo exista.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo VGG16: {e}")
        st.stop()

# Llama a la función para cargar el modelo VGG16
vgg16_model = load_vgg16_model()

# --- Formulario de Entrada para Predicción de Supervivencia (existente) ---
st.header('Predicción de Supervivencia del Gladiador')
st.markdown("---")

with st.form("gladiator_prediction_form"):
    wins = st.number_input(
        "Victorias (Wins)",
        min_value=0,
        max_value=25,
        value=10,
        step=1,
        help="Número total de victorias del gladiador."
    )

    public_favor = st.slider(
        "Favor Público (entre 0 y 1)",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01,
        help="Nivel de favor del público, donde 0 es muy desfavorable y 1 es muy favorable."
    )

    allegiance_network_strong_options = {
        "No (0.0)": 0.0,
        "Sí (1.0)": 1.0
    }
    allegiance_network_strong_display = st.radio(
        "¿Red de Lealtad Fuerte?",
        list(allegiance_network_strong_options.keys()),
        index=1,
        help="Indica si el gladiador tiene una red de lealtad fuerte."
    )
    allegiance_network_strong = allegiance_network_strong_options[allegiance_network_strong_display]

    submitted = st.form_submit_button("Predecir Supervivencia")

# --- Lógica de Predicción de Supervivencia (se ejecuta solo si el botón de enviar es presionado) ---
if submitted:
    input_data = pd.DataFrame([[wins, public_favor, allegiance_network_strong]],
                              columns=['Wins', 'Public Favor', 'Allegiance Network_Strong'])

    try:
        prediction = rfc_model.predict(input_data)[0]
        prediction_proba = rfc_model.predict_proba(input_data)[0]

        st.subheader('Resultado de la Predicción de Supervivencia:')

        if prediction == 1:
            st.success('¡El gladiador **VIVIRÁ**!')
            st.write(f'Probabilidad de sobrevivir: **{prediction_proba[1]:.2f}**')
        else:
            st.error('El gladiador **MORIRÁ**.')
            st.write(f'Probabilidad de sobrevivir: **{prediction_proba[1]:.2f}**')

        st.markdown("---")
        st.write('**Nota:** Esta predicción se basa en el modelo de Random Forest y los datos de entrada proporcionados.')

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción de supervivencia: {e}")
        st.write("Por favor, verifica que los datos de entrada sean válidos y que el modelo esté cargado correctamente.")

# --- Sección para Clasificación de Imágenes (nuevo) ---
st.header('Clasificación de Imágenes de Gladiadores')
st.markdown("---")
st.write("Sube una imagen para determinar si contiene un gladiador.")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, caption='Imagen Subida.', use_container_width=True)
    st.write("")
    st.write("Clasificando...")

    try:
        # Preprocesar la imagen para el modelo VGG16
        # VGG16 espera imágenes de 224x224 píxeles
        # Usamos tf.keras.preprocessing.image.load_img y img_to_array
        img = image.load_img(io.BytesIO(uploaded_file.read()), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Añadir una dimensión de lote

        # Aplicar el preprocesamiento específico de VGG16
        # Es crucial usar la misma función de preprocesamiento que se usó durante el entrenamiento
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

        # Realizar la predicción
        prediction_vgg16 = vgg16_model.predict(img_array)
        # Asumiendo que la salida es una probabilidad única para la clase 'gladiador' (NUM_CLASSES = 1, sigmoid)
        gladiator_probability = prediction_vgg16[0][0]

        st.subheader('Resultado de la Clasificación de Imagen:')

        # Umbral para decidir si es gladiador o no (puedes ajustarlo)
        threshold = 0.5
        if gladiator_probability > threshold:
            st.success(f'¡Esta imagen **ES** un gladiador! (Confianza: {gladiator_probability:.2f})')
        else:
            st.error(f'Esta imagen **NO ES** un gladiador. (Confianza: {1 - gladiator_probability:.2f} de no ser gladiador)')


    except Exception as e:
        st.error(f"Ocurrió un error durante la clasificación de la imagen: {e}")
        st.write("Asegúrate de que el archivo subido sea una imagen válida y que el modelo VGG16 esté configurado correctamente.")

st.sidebar.markdown("---")
st.sidebar.markdown("Aplicación creada con Streamlit.")


