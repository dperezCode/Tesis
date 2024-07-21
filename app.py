import cv2
import streamlit as st
from ultralytics import YOLO
import os
import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image
from io import BytesIO
from sqlalchemy import create_engine, text
import pandas as pd

import matplotlib.pyplot as plt

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

CARPETA_CARGA = 'uploads'

# Configuración de la conexión
def get_connection_string():
    secrets = st.secrets["connections"]["mysql"]
    return f"mysql+mysqlconnector://{secrets['username']}:{secrets['password']}@{secrets['host']}:{secrets['port']}/{secrets['database']}?charset={secrets['query']['charset']}"

# Función para insertar datos en la tabla 'plagas'
def insert_data(engine, plaga):
    query = text("INSERT INTO plagas (name) VALUES (:plaga)")
    with engine.connect() as conn:
        try:
            conn.execute(query, {"plaga": plaga})
            conn.commit()  # Confirmar la transacción
            st.write(f"Plaga '{plaga}' insertada correctamente.")
        except Exception as e:
            st.write(f"Error al insertar plaga '{plaga}': {e}")
            conn.rollback()  # Revertir la transacción en caso de error

# Realizar consulta y obtener datos
def fetch_data(engine):
    query = "SELECT name, COUNT(*) as count FROM plagas GROUP BY name"
    df = pd.read_sql(query, engine)
    return df

# Configuración de la página
st.set_page_config(page_title="letsGoThesis", page_icon="📚", layout="wide")


st.markdown(
    "<h1 style='text-align: center;'>SISTEMA WEB INTELIGENTE PARA LA DETECCIÓN DE AGENTES PLAGA EN LA PRODUCCIÓN DE TOMATE 🍅</h1>", 
    unsafe_allow_html=True
)

def extraer_cont_detecciones(cont_detecciones_str):
    cont_detecciones = {}
    coincidencias = re.findall(r'(\d+)\s+(\w+)', cont_detecciones_str)
    for cont, nombre_clase in coincidencias:
        cont_detecciones[nombre_clase] = int(cont)
    return cont_detecciones

def generar_informe_pdf(ruta_archivo, cont_detecciones, tabla_porcentajes):
    buffer_pdf = BytesIO()
    pdf = canvas.Canvas(buffer_pdf, pagesize=letter)
    ancho_pagina, alto_pagina = pdf._pagesize

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(ancho_pagina / 2, alto_pagina - 30, "Informe de Detección de Tuta absoluta y Bactericera Cockerelli")

    ancho_imagen = 320
    alto_imagen = 320
    x_imagen = (ancho_pagina - ancho_imagen) / 2
    y_imagen = alto_pagina - 30 - 50 - alto_imagen

    pdf.drawImage(ruta_archivo, x_imagen, y_imagen, width=ancho_imagen, height=alto_imagen)

    pdf.setFont("Helvetica", 12)
    pdf.drawString(x_imagen, y_imagen - 30, "Detecciones por Clase:")

    altura_fila = 10
    x = x_imagen
    y = y_imagen - 20

    for nombre_clase, cont in cont_detecciones.items():
        pdf.drawString(x, y, f"{nombre_clase}: {cont} detecciones - {tabla_porcentajes[nombre_clase]}")
        y -= altura_fila

    pdf.save()

    st.download_button(
        label="Descargar Informe PDF",
        data=buffer_pdf.getvalue(),
        file_name="informe_deteccion.pdf",
        key="pdf_report",
    )

def generar_frames(ruta_imagen, contenedor_detecciones, engine):
    modelo = YOLO("tomate2.pt")
    imagen = cv2.imread(ruta_imagen)
    imagen = cv2.resize(imagen, (640, 640))
    resultados = modelo.predict(imagen, conf=0.2)
    resultado = resultados[0]
    imagen_cv = cv2.cvtColor(resultado.plot()[:, :, ::-1], cv2.COLOR_RGB2BGR)
    cont_detecciones_str = resultado.verbose()
    cont_detecciones = extraer_cont_detecciones(cont_detecciones_str)

    st.write("Detecciones encontradas:", cont_detecciones)  # Agregar mensaje de depuración

    for plaga in cont_detecciones.keys():
        st.write(f"Intentando insertar la plaga: {plaga}")  # Agregar mensaje de depuración
        insert_data(engine, plaga)

    cont_total = sum(cont_detecciones.values())
    st.image(imagen_cv, channels="BGR", use_column_width=True)
    contenedor_detecciones.text("Detecciones por Clase:")

    tabla_porcentajes = {
        nombre_clase: f"{(cont / cont_total) * 100:.2f}%" for nombre_clase, cont in cont_detecciones.items()
    }

    contenedor_detecciones.write("Porcentajes Relativos:")
    contenedor_detecciones.write(tabla_porcentajes)

    generar_informe_pdf(ruta_imagen, cont_detecciones, tabla_porcentajes)

def generar_video_frames(ruta_video, contenedor_detecciones, engine):
    modelo = YOLO("tomate2.pt")
    cap = cv2.VideoCapture(ruta_video)
    st.warning("Mostrando frames del video:")
    contenedor_frames = st.empty()
    cont_total = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    generar_informe = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 640))
        resultados = modelo.predict(frame, conf=0.2)
        resultado = resultados[0]
        imagen_cv = cv2.cvtColor(resultado.plot()[:, :, ::-1], cv2.COLOR_RGB2BGR)
        cont_detecciones_str = resultado.verbose()
        cont_detecciones = extraer_cont_detecciones(cont_detecciones_str)

        contenedor_frames.image(imagen_cv, channels="BGR", use_column_width=True)
        contenedor_detecciones.text("Detecciones por Clase:")

        for nombre_clase, cont in cont_detecciones.items():
            cont_total[nombre_clase] = cont_total.get(nombre_clase, 0) + cont

        cont_total_global = sum(cont_total.values())

        tabla_porcentajes_global = {
            nombre_clase: f"{(cont / cont_total_global) * 100:.2f}%" for nombre_clase, cont in cont_total.items()
        }

        contenedor_detecciones.write("Porcentajes Relativos:")
        contenedor_detecciones.write(tabla_porcentajes_global)

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == total_frames - 1 and not generar_informe:
            boton_key = f"boton_informe_pdf_{cap.get(cv2.CAP_PROP_POS_FRAMES)}"
            if st.button("Generar Informe PDF", key=boton_key):
                generar_informe = True
                break

    if generar_informe:
        generar_informe_pdf(ruta_video, cont_total, tabla_porcentajes_global)
        st.stop()

# Main

# Mostrar gráfico de barras
def mostrar_grafico_barras(df):
    plt.figure(figsize=(10, 5))
    bars =plt.bar(df['name'], df['count'], color='red')
    plt.xlabel('Plaga')
    plt.ylabel('Cantidad')
    plt.title('Cantidad de Plagas Detectadas')
    total = df['count'].sum()
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total) * 100
        plt.annotate(f'{percentage:.2f}%', 
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')
    st.pyplot(plt)

def main():
     
    connection_string = get_connection_string()
    engine = create_engine(connection_string)
    
    opcion = st.radio("Seleccionar Procesamiento", ["Imagen"])

    archivo_cargado = st.file_uploader(f"Elegir y subir {opcion.lower()}", type=["jpg", "jpeg", "png"])

    if archivo_cargado is not None:
        ruta_archivo = os.path.join(CARPETA_CARGA, archivo_cargado.name)
        with open(ruta_archivo, "wb") as f:
            f.write(archivo_cargado.getvalue())

        contenedor_detecciones = st.sidebar.empty()

        if opcion == "Imagen":
            st.image(ruta_archivo, channels="BGR", use_column_width=True)
            st.warning("Mostrando imagen procesada por YOLO:")
            generar_frames(ruta_archivo, contenedor_detecciones, engine)
        elif opcion == "Video":
            generar_video_frames(ruta_archivo, contenedor_detecciones, engine)
    
    imagen_fondo = st.columns(1)[0]
    with imagen_fondo:
        image = Image.open('imagenes/fondoTesis.png')
        st.image(image, use_column_width=True)

    
    # Mostrar datos de la tabla 'plagas'
   
    st.write("Plagas registradas")
    df = fetch_data(engine)
    if not df.empty:
        st.write(df)
        mostrar_grafico_barras(df)
    else:
        st.write("No se encontraron datos en la tabla 'plagas'.")

if __name__ == "__main__":
    main()


# Función para enviar correo electrónico
def enviar_correo(destinatario, asunto, mensaje):
    remitente = "perezeusebiodavila@gmail.com"  # Reemplaza con tu correo de Gmail
    contrasena = "qjie vezi sbxb ibkt"  # Reemplaza con tu contraseña de Gmail

    # Crear el mensaje
    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto
    msg.attach(MIMEText(mensaje, 'plain'))

    try:
        # Conectarse al servidor de Gmail
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(remitente, contrasena)
        
        # Enviar el correo
        servidor.sendmail(remitente, destinatario, msg.as_string())
        servidor.close()
        
        return True
    except Exception as e:
        print(f"Error al enviar el correo: {e}")
        return False

# Crear el formulario en Streamlit

with st.container():
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h1 style='text-align: start;'>Escríbemos 📝</h1>", unsafe_allow_html=True)
        st.write("""
            Si tienes alguna duda sobre el funcionamiento o desarrollo del sistema, 
            por favor, deja tu mensaje en el formulario a continuación. 
            Nos aseguraremos de responderte a la brevedad posible.
            Tu feedback es muy importante para nosotros. 📝
            """)
        image = Image.open("imagenes/secretaria.png") 
        # image = resize_image(image)
        st.image(image, use_column_width=True)

    with col2:  
        st.title("Formulario de Contacto")  
        nombre = st.text_input("Nombre")
        email = st.text_input("Correo Electrónico")
        asunto = st.text_input("Asunto")
        mensaje = st.text_area("Mensaje")

        if st.button("Enviar"):
            if nombre and email and asunto and mensaje:
                cuerpo_mensaje = f"Nombre: {nombre}\nCorreo Electrónico: {email}\n\nMensaje:\n{mensaje}"
                if enviar_correo("perezeusebiodavila@gmail.com", asunto, cuerpo_mensaje):
                    st.success("¡Correo enviado exitosamente!")
                else:
                    st.error("Error al enviar el correo.")
            else:
                st.warning("Por favor, completa todos los campos del formulario.")


        