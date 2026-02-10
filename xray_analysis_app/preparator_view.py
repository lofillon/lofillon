import streamlit as st
import os
from dicom_importer import DICOMImporter
from model_interface import ModelInterface
import pandas as pd
from datetime import datetime, date
from PIL import Image
import uuid

class PreparatorView:
    """Vue pour le rôle Préparateur"""
    
    def __init__(self):
        self.data_manager = st.session_state.data_manager
        self.dicom_importer = DICOMImporter()
        self.model_interface = ModelInterface()
    
    def render(self):
        st.header("👨‍💼 Vue Préparateur")
        
        # Navigation par onglets
        tab1, tab2, tab3, tab4 = st.tabs([
            "📥 Import DICOM",
            "🤖 Analyse Modèle",
            "📊 Visualisation & Filtrage",
            "✅ Validation & Envoi"
        ])
        
        with tab1:
            self._render_import_tab()
        
        with tab2:
            self._render_analysis_tab()
        
        with tab3:
            self._render_visualization_tab()
        
        with tab4:
            self._render_validation_tab()
    
    def _render_import_tab(self):
        """Onglet d'import DICOM et images simples"""
        # Créer deux colonnes pour les deux types d'import
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Import Fichiers DICOM")
            
            st.info("""
            **Instructions:**
            - Sélectionnez un ou plusieurs fichiers DICOM (.dcm)
            - Chaque fichier sera analysé et lié à un patient via le PatientID
            - Les métadonnées DICOM seront extraites automatiquement
            """)
            
            uploaded_dicom_files = st.file_uploader(
                "Sélectionnez les fichiers DICOM",
                type=['dcm', 'dicom'],
                accept_multiple_files=True,
                key="dicom_uploader"
            )
            
            if uploaded_dicom_files:
                if st.button("Importer les fichiers DICOM", type="primary", key="import_dicom"):
                    self._import_files(uploaded_dicom_files)
        
        with col2:
            st.subheader("🖼️ Import Images Simples")
            
            st.info("""
            **Instructions:**
            - Sélectionnez une ou plusieurs images (PNG, JPG, JPEG)
            - Entrez l'ID patient manuellement pour chaque image
            - Les métadonnées de base seront créées automatiquement
            """)
            
            # Utiliser une clé basée sur le temps pour réinitialiser le file_uploader
            if 'image_uploader_key' not in st.session_state:
                st.session_state.image_uploader_key = 0
            
            uploaded_image_files = st.file_uploader(
                "Sélectionnez les images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key=f"image_uploader_{st.session_state.image_uploader_key}"
            )
            
            if uploaded_image_files:
                # Formulaire pour les métadonnées patient
                with st.form("simple_image_form"):
                    st.write("**Spécifiez l'ID patient pour chaque image :**")
                    
                    # Créer un champ patient_id pour chaque image
                    patient_ids = {}
                    patient_metadata = {}
                    
                    for i, uploaded_file in enumerate(uploaded_image_files):
                        st.write(f"**Image {i+1}:** {uploaded_file.name}")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            patient_id = st.text_input(
                                f"ID Patient pour {uploaded_file.name}",
                                value="",
                                help="ID unique du patient (obligatoire)",
                                key=f"patient_id_{i}_{st.session_state.image_uploader_key}"
                            )
                            patient_ids[uploaded_file.name] = patient_id
                        
                        with col2:
                            patient_sex = st.selectbox(
                                "Sexe",
                                ['', 'M', 'F'],
                                key=f"patient_sex_{i}_{st.session_state.image_uploader_key}"
                            )
                            patient_metadata[uploaded_file.name] = {
                                'sex': patient_sex,
                                'age': 0,
                                'exam_date': datetime.now().date()
                            }
                        
                        # Âge et date d'examen (optionnel, peut être partagé)
                        if i == 0:  # Afficher seulement pour la première image
                            col_a, col_b = st.columns(2)
                            with col_a:
                                patient_age = st.number_input(
                                    "Âge",
                                    min_value=0,
                                    max_value=150,
                                    value=0,
                                    key=f"patient_age_{st.session_state.image_uploader_key}"
                                )
                            with col_b:
                                exam_date = st.date_input(
                                    "Date d'examen",
                                    value=datetime.now().date(),
                                    key=f"exam_date_{st.session_state.image_uploader_key}"
                                )
                            
                            # Appliquer à toutes les images
                            for img_name in patient_ids.keys():
                                patient_metadata[img_name]['age'] = patient_age
                                patient_metadata[img_name]['exam_date'] = exam_date
                    
                    submitted = st.form_submit_button("Importer les images", type="primary")
                    
                    if submitted:
                        # Vérifier que tous les patient_id sont remplis
                        missing_ids = [name for name, pid in patient_ids.items() if not pid]
                        if missing_ids:
                            st.error(f"⚠️ L'ID Patient est obligatoire pour : {', '.join(missing_ids)}")
                        else:
                            self._import_simple_images(
                                uploaded_image_files,
                                patient_ids,
                                patient_metadata
                            )
                            # Réinitialiser le file_uploader en changeant la clé
                            st.session_state.image_uploader_key += 1
                            # Ne pas faire de rerun pour éviter le refresh
                            st.success("✅ Import terminé. Vous pouvez continuer à importer d'autres images.")
        
        # Afficher les fichiers récemment importés
        st.divider()
        st.subheader("📋 Fichiers importés récemment")
        images = self.data_manager.get_all_images()
        if images:
            recent_images = sorted(images, key=lambda x: x.get('created_at', ''), reverse=True)[:10]
            df_recent = pd.DataFrame([{
                'ID Image': img['id'],
                'ID Patient': img.get('patient_id', 'N/A'),
                'Date Examen': img.get('exam_date', 'N/A'),
                'Type': img.get('import_type', 'DICOM'),
                'Statut': img.get('status', 'pending')
            } for img in recent_images])
            st.dataframe(df_recent, use_container_width=True)
        else:
            st.info("Aucun fichier importé pour le moment")
    
    def _import_files(self, uploaded_files):
        """Importe les fichiers DICOM"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Sauvegarder temporairement les fichiers
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        # Importer les fichiers
        results = self.dicom_importer.import_batch(file_paths)
        
        success_count = 0
        error_count = 0
        
        for i, result in enumerate(results):
            progress = (i + 1) / len(results)
            progress_bar.progress(progress)
            
            if result['success']:
                # Ajouter le patient
                patient_id = result['metadata']['patient_id']
                patient_metadata = {
                    'sex': result['metadata'].get('sex', ''),
                    'age': result['metadata'].get('age', ''),
                    'institution_name': result['metadata'].get('institution_name', ''),
                    'station_name': result['metadata'].get('station_name', '')
                }
                self.data_manager.add_patient(patient_id, patient_metadata)
                
                # Ajouter l'image
                image_data = {
                    'patient_id': patient_id,
                    'file_path': result['file_path'],
                    'image_path': result['image_path'],
                    'exam_date': result['metadata'].get('exam_date_formatted', ''),
                    'exam_time': result['metadata'].get('exam_time', ''),
                    'modality': result['metadata'].get('modality', ''),
                    'body_part': result['metadata'].get('body_part', ''),
                    'patient_position': result['metadata'].get('patient_position', ''),
                    'view_position': result['metadata'].get('view_position', ''),
                    'study_description': result['metadata'].get('study_description', '')
                }
                self.data_manager.add_image(image_data)
                success_count += 1
            else:
                error_count += 1
                st.warning(f"Erreur pour {os.path.basename(result['file_path'])}: {result['error']}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Nettoyer les fichiers temporaires
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except:
                pass
        
        if success_count > 0:
            st.success(f"✅ {success_count} fichier(s) importé(s) avec succès")
        if error_count > 0:
            st.error(f"❌ {error_count} fichier(s) en erreur")
        
        # Ne pas faire de rerun pour éviter le refresh
    
    def _import_simple_images(self, uploaded_images, patient_ids, patient_metadata_dict):
        """Importe des images simples (PNG, JPG, etc.) avec un patient_id par image"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Créer le répertoire pour les images
        images_dir = "data/images"
        os.makedirs(images_dir, exist_ok=True)
        
        success_count = 0
        error_count = 0
        patients_imported = set()
        
        for i, uploaded_image in enumerate(uploaded_images):
            progress = (i + 1) / len(uploaded_images)
            progress_bar.progress(progress)
            status_text.text(f"Import de l'image {i+1}/{len(uploaded_images)}: {uploaded_image.name}")
            
            try:
                # Récupérer le patient_id pour cette image
                patient_id = patient_ids.get(uploaded_image.name, '')
                if not patient_id:
                    error_count += 1
                    st.warning(f"⚠️ ID Patient manquant pour {uploaded_image.name}")
                    continue
                
                # Récupérer les métadonnées pour cette image
                metadata = patient_metadata_dict.get(uploaded_image.name, {})
                patient_sex = metadata.get('sex', '')
                patient_age = metadata.get('age', 0)
                exam_date = metadata.get('exam_date', datetime.now().date())
                
                # Ajouter le patient si pas déjà fait
                if patient_id not in patients_imported:
                    patient_meta = {
                        'sex': patient_sex if patient_sex else '',
                        'age': f"{patient_age:03d}Y" if patient_age > 0 else '',
                        'institution_name': '',
                        'station_name': ''
                    }
                    self.data_manager.add_patient(patient_id, patient_meta)
                    patients_imported.add(patient_id)
                
                # Lire l'image
                image = Image.open(uploaded_image)
                
                # Convertir en RGB si nécessaire
                if image.mode != 'RGB' and image.mode != 'L':
                    image = image.convert('RGB')
                
                # Générer un nom de fichier unique
                file_extension = os.path.splitext(uploaded_image.name)[1] or '.png'
                unique_id = str(uuid.uuid4())[:8]
                image_filename = f"{patient_id}_{unique_id}{file_extension}"
                image_path = os.path.join(images_dir, image_filename)
                
                # Redimensionner si trop grande (max 2048px)
                max_size = 2048
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Sauvegarder l'image
                image.save(image_path, format='PNG' if file_extension.lower() == '.png' else 'JPEG')
                
                # Créer les métadonnées de l'image
                image_data = {
                    'patient_id': patient_id,
                    'file_path': uploaded_image.name,
                    'image_path': image_path,
                    'exam_date': exam_date.strftime("%Y-%m-%d") if isinstance(exam_date, date) else str(exam_date),
                    'exam_time': datetime.now().strftime("%H%M%S"),
                    'modality': 'CR',  # Computed Radiography
                    'body_part': 'CHEST',
                    'patient_position': '',
                    'view_position': '',
                    'study_description': 'CHEST',
                    'import_type': 'Simple Image'  # Pour distinguer des DICOM
                }
                
                self.data_manager.add_image(image_data)
                success_count += 1
                
            except Exception as e:
                error_count += 1
                st.warning(f"Erreur pour {uploaded_image.name}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        if success_count > 0:
            st.success(f"✅ {success_count} image(s) importée(s) avec succès")
            if len(patients_imported) > 1:
                st.info(f"📋 {len(patients_imported)} patient(s) différent(s) : {', '.join(sorted(patients_imported))}")
        if error_count > 0:
            st.error(f"❌ {error_count} image(s) en erreur")
    
    def _render_analysis_tab(self):
        """Onglet d'analyse par le modèle"""
        st.subheader("Lancement de l'Analyse par le Modèle")
        
        # Sélection des images à analyser
        images = self.data_manager.get_all_images()
        pending_images = [img for img in images if img.get('status') == 'pending']
        
        if not pending_images:
            st.info("Aucune image en attente d'analyse")
            return
        
        st.write(f"**{len(pending_images)} image(s) en attente d'analyse**")
        
        # Filtrer par date ou patient
        col1, col2 = st.columns(2)
        with col1:
            filter_date = st.date_input("Filtrer par date d'examen")
        with col2:
            filter_patient = st.text_input("Filtrer par ID Patient")
        
        filtered_images = pending_images
        if filter_date:
            filter_date_str = filter_date.strftime("%Y-%m-%d")
            filtered_images = [img for img in filtered_images if img.get('exam_date') == filter_date_str]
        if filter_patient:
            filtered_images = [img for img in filtered_images if filter_patient.lower() in img.get('patient_id', '').lower()]
        
        if filtered_images:
            st.write(f"**{len(filtered_images)} image(s) sélectionnée(s)**")
            
            # Afficher la liste
            df_pending = pd.DataFrame([{
                'ID Image': img['id'],
                'ID Patient': img.get('patient_id', 'N/A'),
                'Date Examen': img.get('exam_date', 'N/A'),
                'Statut': img.get('status', 'pending')
            } for img in filtered_images])
            st.dataframe(df_pending, use_container_width=True)
            
            if st.button("🚀 Lancer l'analyse sur les images sélectionnées", type="primary"):
                self._run_model_analysis(filtered_images)
        else:
            st.info("Aucune image ne correspond aux filtres")
        
        # Afficher les statuts d'analyse
        st.subheader("Statut des Analyses")
        all_images = self.data_manager.get_all_images()
        status_counts = {}
        for img in all_images:
            status = img.get('status', 'pending')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("En attente", status_counts.get('pending', 0))
        col2.metric("En cours", status_counts.get('processing', 0))
        col3.metric("Terminé", status_counts.get('completed', 0))
        col4.metric("Erreur", status_counts.get('failed', 0))
        
        # Liste des erreurs
        failed_images = [img for img in all_images if img.get('status') == 'failed']
        if failed_images:
            st.subheader("Images en Erreur")
            df_failed = pd.DataFrame([{
                'ID Image': img['id'],
                'ID Patient': img.get('patient_id', 'N/A'),
                'Erreur': img.get('error', 'Erreur inconnue')
            } for img in failed_images])
            st.dataframe(df_failed, use_container_width=True)
    
    def _run_model_analysis(self, images):
        """Lance l'analyse du modèle sur les images"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, image in enumerate(images):
            progress = (i + 1) / len(images)
            progress_bar.progress(progress)
            status_text.text(f"Analyse de l'image {i+1}/{len(images)}: {image['id']}")
            
            # Mettre à jour le statut
            self.data_manager.update_image_status(image['id'], 'processing')
            
            # Lancer la prédiction
            image_path = image.get('image_path')
            if image_path and os.path.exists(image_path):
                try:
                    prediction = self.model_interface.predict(image_path)
                    
                    # Sauvegarder la prédiction
                    self.data_manager.add_prediction({
                        'image_id': image['id'],
                        'patient_id': image.get('patient_id'),
                        'label': prediction['label'],
                        'confidence': prediction['confidence']
                    })
                    
                    # Mettre à jour le statut
                    self.data_manager.update_image_status(image['id'], 'completed')
                except Exception as e:
                    self.data_manager.update_image_status(image['id'], 'failed', str(e))
            else:
                self.data_manager.update_image_status(image['id'], 'failed', "Image non trouvée")
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Analyse terminée pour {len(images)} image(s)")
        st.rerun()
    
    def _render_visualization_tab(self):
        """Onglet de visualisation et filtrage"""
        st.subheader("Visualisation et Filtrage des Classifications")
        
        df = self.data_manager.get_dataframe_for_preparator()
        
        if df.empty:
            st.info("Aucune donnée à afficher")
            return
        
        # Filtres
        st.subheader("Filtres")
        col1, col2 = st.columns(2)
        
        with col1:
            filter_prediction = st.selectbox(
                "Filtrer par prédiction",
                ['Tous', 'sain', 'malade', 'En attente']
            )
        
        with col2:
            filter_annotation = st.selectbox(
                "Filtrer par annotation",
                ['Tous', 'Annoté', 'Non annoté']
            )
        
        # Appliquer les filtres
        filtered_df = df.copy()
        
        if filter_prediction != 'Tous':
            filtered_df = filtered_df[filtered_df['Prédiction Modèle'] == filter_prediction]
        
        if filter_annotation != 'Tous':
            if filter_annotation == 'Annoté':
                filtered_df = filtered_df[filtered_df['Annotation Préparateur'] != 'Non annoté']
            else:
                filtered_df = filtered_df[filtered_df['Annotation Préparateur'] == 'Non annoté']
        
        # Mise en évidence visuelle
        st.subheader("Résultats")
        st.write(f"**{len(filtered_df)} résultat(s) trouvé(s)**")
        
        # Trier
        sort_option = st.selectbox(
            "Trier par",
            ['Date Examen', 'ID Patient', 'Prédiction Modèle']
        )
        
        if sort_option == 'Date Examen':
            filtered_df = filtered_df.sort_values('Date Examen', ascending=False)
        elif sort_option == 'ID Patient':
            filtered_df = filtered_df.sort_values('ID Patient')
        else:
            filtered_df = filtered_df.sort_values('Prédiction Modèle')
        
        # Afficher le tableau sans mise en évidence, texte en blanc
        styled_df = filtered_df.style.set_properties(
            subset=filtered_df.columns,
            **{'color': '#ffffff'}  # Texte blanc pour toutes les lignes
        )
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    def _render_validation_tab(self):
        """Onglet de validation et envoi"""
        st.subheader("Validation et Envoi au Médecin")
        
        # Sélection d'un patient pour annotation
        images = self.data_manager.get_all_images()
        completed_images = [img for img in images if img.get('status') == 'completed']
        
        if not completed_images:
            st.info("Aucune image analysée disponible pour validation")
            return
        
        st.write("**Sélectionnez un patient pour annoter et valider**")
        
        # Grouper par patient
        patients_dict = {}
        for img in completed_images:
            patient_id = img.get('patient_id')
            if patient_id not in patients_dict:
                patients_dict[patient_id] = []
            patients_dict[patient_id].append(img)
        
        selected_patient = st.selectbox(
            "Sélectionner un patient",
            list(patients_dict.keys())
        )
        
        if selected_patient:
            patient_images = patients_dict[selected_patient]
            
            # Afficher les informations du patient
            patient = self.data_manager.get_patient_by_id(selected_patient)
            if patient:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**ID Patient:** {selected_patient}")
                with col2:
                    st.write(f"**Sexe:** {patient.get('metadata', {}).get('sex', 'N/A')}")
                with col3:
                    st.write(f"**Âge:** {patient.get('metadata', {}).get('age', 'N/A')}")
            
            # Pour chaque image du patient
            for img in patient_images:
                with st.expander(f"Image {img['id']} - {img.get('exam_date', 'N/A')}"):
                    self._render_image_annotation(img)
        
        # Section d'envoi au médecin
        st.subheader("Envoi au Médecin")
        
        # Vérifier quels patients sont annotés
        annotated_patients = []
        unannotated_patients = []
        
        for patient_id in patients_dict.keys():
            if self.data_manager.is_patient_annotated(patient_id):
                annotated_patients.append(patient_id)
            else:
                unannotated_patients.append(patient_id)
        
        if unannotated_patients:
            st.warning(f"⚠️ {len(unannotated_patients)} patient(s) non annoté(s). Tous les patients doivent être annotés avant l'envoi.")
            st.write("**Patients non annotés:**")
            for pid in unannotated_patients:
                st.write(f"- {pid}")
        
        if annotated_patients:
            st.success(f"✅ {len(annotated_patients)} patient(s) annoté(s) et prêt(s) pour l'envoi")
            
            # Sélection des patients à envoyer
            selected_for_send = st.multiselect(
                "Sélectionner les patients à envoyer au médecin",
                annotated_patients,
                default=annotated_patients
            )
            
            if st.button("📤 Envoyer la liste au médecin", type="primary"):
                # Récupérer toutes les images des patients sélectionnés
                image_ids_to_send = []
                for patient_id in selected_for_send:
                    for img in patients_dict[patient_id]:
                        image_ids_to_send.append(img['id'])
                
                self.data_manager.mark_batch_for_review(
                    image_ids_to_send,
                    st.session_state.current_user_name
                )
                
                st.success(f"✅ {len(selected_for_send)} patient(s) envoyé(s) au médecin pour revue")
                st.rerun()
    
    def _render_image_annotation(self, image):
        """Affiche l'interface d'annotation pour une image"""
        # Afficher l'image
        image_path = image.get('image_path')
        if image_path and os.path.exists(image_path):
            st.image(image_path, caption=f"Image {image['id']}", use_container_width=True)
        else:
            st.warning("Image non disponible")
        
        # Afficher la prédiction du modèle
        prediction = self.data_manager.get_prediction_by_image(image['id'])
        if prediction:
            st.write(f"**Prédiction Modèle:** {prediction.get('label', 'N/A')}")
        
        # Récupérer l'annotation existante
        annotation = self.data_manager.get_annotation_by_image(image['id'])
        
        # Formulaire d'annotation
        with st.form(f"annotation_form_{image['id']}"):
            st.subheader("Annotation")
            
            # Label
            current_label = annotation.get('label', prediction.get('label', 'sain')) if annotation else prediction.get('label', 'sain')
            label = st.radio(
                "Classification",
                ['sain', 'malade'],
                index=0 if current_label == 'sain' else 1
            )
            
            # Confiance
            current_confidence = annotation.get('confidence', prediction.get('confidence', 0.5)) if annotation else prediction.get('confidence', 0.5)
            confidence = st.slider(
                "Confiance",
                min_value=0.0,
                max_value=1.0,
                value=float(current_confidence),
                step=0.01
            )
            
            # Notes
            current_notes = annotation.get('notes', '') if annotation else ''
            notes = st.text_area("Notes", value=current_notes)
            
            # Informations supplémentaires
            st.subheader("Informations Complémentaires")
            
            current_info = annotation.get('additional_info', {}) if annotation else {}
            
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Âge", min_value=0, max_value=150, value=int(current_info.get('age', 0)) if current_info.get('age') else 0)
                symptoms = st.text_input("Symptômes", value=current_info.get('symptoms', ''))
                comorbidities = st.text_input("Comorbidités", value=current_info.get('comorbidities', ''))
            
            with col2:
                spo2 = st.number_input("SpO₂ (%)", min_value=0, max_value=100, value=int(current_info.get('spo2', 0)) if current_info.get('spo2') else 0)
                temperature = st.number_input("Température (°C)", min_value=30.0, max_value=45.0, value=float(current_info.get('temperature', 37.0)) if current_info.get('temperature') else 37.0)
                crp = st.number_input("CRP (mg/L)", min_value=0.0, value=float(current_info.get('crp', 0.0)) if current_info.get('crp') else 0.0)
            
            image_quality = st.selectbox(
                "Qualité d'image",
                ['Excellente', 'Bonne', 'Moyenne', 'Mauvaise'],
                index=['Excellente', 'Bonne', 'Moyenne', 'Mauvaise'].index(current_info.get('image_quality', 'Bonne'))
            )
            
            urgency = st.selectbox(
                "Urgence clinique",
                ['Normale', 'Élevée', 'Critique'],
                index=['Normale', 'Élevée', 'Critique'].index(current_info.get('urgency', 'Normale'))
            )
            
            additional_info = {
                'age': age if age > 0 else None,
                'symptoms': symptoms,
                'comorbidities': comorbidities,
                'spo2': spo2 if spo2 > 0 else None,
                'temperature': temperature,
                'crp': crp,
                'image_quality': image_quality,
                'urgency': urgency
            }
            
            submitted = st.form_submit_button("💾 Enregistrer l'annotation", type="primary")
            
            if submitted:
                if annotation:
                    # Mettre à jour l'annotation existante
                    self.data_manager.update_annotation(
                        image['id'],
                        st.session_state.current_user_name,
                        {
                            'label': label,
                            'confidence': confidence,
                            'notes': notes,
                            'additional_info': additional_info,
                            'user_role': 'Préparateur'
                        }
                    )
                else:
                    # Créer une nouvelle annotation
                    self.data_manager.add_annotation({
                        'image_id': image['id'],
                        'patient_id': image.get('patient_id'),
                        'label': label,
                        'confidence': confidence,
                        'notes': notes,
                        'additional_info': additional_info,
                        'user_name': st.session_state.current_user_name,
                        'user_role': 'Préparateur'
                    })
                
                st.success("✅ Annotation enregistrée")
                st.rerun()
        
        # Afficher l'historique des versions
        if annotation:
            st.subheader("Historique des Versions")
            st.write(f"**Version actuelle:** {annotation.get('version', 1)}")
            st.write(f"**Dernière modification:** {annotation.get('created_at', 'N/A')}")
            st.write(f"**Par:** {annotation.get('user_name', 'N/A')}")

