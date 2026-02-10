import streamlit as st
import os
from data_manager import DataManager
from preparator_view import PreparatorView
from doctor_view import DoctorView

# Configuration de la page
st.set_page_config(
    page_title="Analyse X-Ray Pneumonie",
    page_icon="🫁",
    layout="wide"
)

# Initialisation de la session state
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()
else:
    # Vérifier que l'instance a les nouvelles méthodes (si le code a été mis à jour)
    if not hasattr(st.session_state.data_manager, 'start_treatment'):
        st.session_state.data_manager = DataManager()

if 'current_user_role' not in st.session_state:
    st.session_state.current_user_role = None

if 'current_user_name' not in st.session_state:
    st.session_state.current_user_name = None

def main():
    st.title("🫁 Système d'Analyse de Radiographies Thoraciques pour la détection de la pneumonie")
    
    # Sélection du rôle
    if st.session_state.current_user_role is None:
        st.sidebar.title("Connexion")
        user_role = st.sidebar.selectbox(
            "Sélectionnez votre rôle",
            ["Préparateur", "Médecin"]
        )
        user_name = st.sidebar.text_input("Nom d'utilisateur", value="")
        
        if st.sidebar.button("Se connecter"):
            if user_name:
                st.session_state.current_user_role = user_role
                st.session_state.current_user_name = user_name
                st.rerun()
            else:
                st.sidebar.error("Veuillez entrer un nom d'utilisateur")
    else:
        # Affichage du rôle actuel
        st.sidebar.title("Session")
        st.sidebar.write(f"**Rôle:** {st.session_state.current_user_role}")
        st.sidebar.write(f"**Utilisateur:** {st.session_state.current_user_name}")
        
        if st.sidebar.button("Se déconnecter"):
            st.session_state.current_user_role = None
            st.session_state.current_user_name = None
            st.rerun()
        
        # Navigation selon le rôle
        if st.session_state.current_user_role == "Préparateur":
            PreparatorView().render()
        elif st.session_state.current_user_role == "Médecin":
            DoctorView().render()

if __name__ == "__main__":
    main()

