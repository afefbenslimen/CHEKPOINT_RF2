import os

import cv2
import streamlit as st

# Instructions
st.title("Application de Détection 👀 de Visages")
st.write("""
Cette application utilise l'algorithme Viola-Jones pour détecter les visages à partir de votre webcam. Voici comment l'utiliser :
1. Appuyez sur le bouton 'Démarrer la détection des visages ⚡⚡⚡' pour commencer à détecter les visages à partir de votre webcam.
2. Utilisez les contrôles pour ajuster les paramètres de détection 🌱🌱🌱comme la couleur des rectangles, le nombre minimal de voisins et le facteur d'échelle.
3. Si vous souhaitez enregistrer les images avec les visages détectés, cochez la case '📫📫Enregistrer les images'.
4. Appuyez sur le bouton 'Arrêter la détection 👋' pour arrêter la détection des visages.
""")

# Chargement du classificateur de visages
face_cascade = cv2.CascadeClassifier(
    r'C:\Users\benslimane\OneDrive - VITALAIT\Bureau\ion af\Data sience\reconnaisance facial\haarcascade_frontalface_default.xml')

# Variable globale pour contrôler la boucle de détection
is_running = False


# Fonction pour détecter les visages
def detect_faces(rectangle_color, save_images, min_neighbors, scale_factor):
    global is_running
    is_running = True
    counter = 0

    # Initialisation de la webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("Erreur : Impossible d'ouvrir la webcam.")
        return

    frame_placeholder = st.empty()

    while is_running:
        # Lecture des images depuis la webcam
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur : Impossible de capturer l'image.")
            break

        # Conversion des images en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Détection des visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Conversion de la couleur de l'hexadécimal au format BGR
        bgr_color = tuple(int(rectangle_color[i:i + 2], 16) for i in (1, 3, 5))

        # Dessiner des rectangles autour des visages détectés
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)

        # Affichage des images
        frame_placeholder.image(frame, channels="BGR")

        # Enregistrement des images avec les visages détectés si activé
        if save_images:
            counter += 1
            # Obtenir le chemin du projet
            project_path = os.path.dirname(os.path.abspath(__file__))
            # Créer le dossier 'images' s'il n'existe pas
            images_folder = os.path.join(project_path, 'images')
            os.makedirs(images_folder, exist_ok=True)
            # Enregistrer l'image avec le chemin approprié
            save_path = os.path.join(images_folder, f"image_{counter}.jpg")
            cv2.imwrite(save_path, frame)
            st.write(f"Image enregistrée : {save_path}")

    # Libérer la webcam
    cap.release()


def app():
    # Interface de l'application Streamlit
    st.title("Détection de Visages avec l'Algorithme Viola-Jones")
    st.write(
        "Appuyez sur le bouton 'Démarrer la détection des visages' pour commencer à détecter les visages à partir de votre webcam.")
    st.write("Appuyez sur le bouton 'Arrêter la détection' pour arrêter la détection des visages.")

    # Choix de la couleur pour les rectangles
    rectangle_color = st.color_picker("Choisissez la couleur des rectangles", "#00FF00")

    # Option pour enregistrer les images
    save_images = st.checkbox("Enregistrer les images")

    # Ajustement des paramètres minNeighbors et scaleFactor
    min_neighbors = st.slider("minNeighbors", 1, 10, 5)
    scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3)

    # Bouton pour démarrer la détection des visages
    if st.button("Démarrer la détection des visages"):
        # Appeler la fonction detect_faces avec les paramètres choisis
        detect_faces(rectangle_color, save_images, min_neighbors, scale_factor)

    # Bouton pour arrêter la détection des visages
    if st.button("Arrêter la détection"):
        global is_running
        is_running = False


app()
