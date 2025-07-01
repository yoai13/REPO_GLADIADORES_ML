import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.title('Modelo de predicción')

with st.form('Formulario:'):
    victorias = st.number_input ('Wins', max_value=15)


    boton = st.form_submit_button('Enviar')

if boton:

    st.subheader('valores:')
    st.write('Wins', victorias)






