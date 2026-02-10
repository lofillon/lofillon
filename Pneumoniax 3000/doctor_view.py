import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
import shutil
import zipfile
from pathlib import Path

class DoctorView:
    """Vue pour le rôle Médecin"""
    
    def __init__(self):
        self.data_manager = st.session_state.data_manager
    
    def render(self):
        st.header("👨‍⚕️ Vue Médecin")
        
        # Navigation par onglets
        tab1, tab2, tab3, tab4 = st.tabs([
            "✅ Validation des patients",
            "📋 Suivi du Traitement",
            "📝 Finalisation du dossier",
            "📊 Résultats, Export & Historique"
        ])
        
        with tab1:
            self._render_patient_list_tab()
        
        with tab2:
            self._render_treatment_followup_tab()
        
        with tab3:
            self._render_finalization_tab()
        
        with tab4:
            self._render_results_tab()
    
    def _render_patient_list_tab(self):
        """Onglet de validation des patients"""
        st.subheader("Validation des Patients")
        
        # Récupérer les images en attente de revue
        df = self.data_manager.get_dataframe_for_doctor()
        
        if df.empty:
            st.info("Aucun patient en attente de revue médicale")
            return
        
        st.write(f"**{len(df)} patient(s) en attente de revue**")
        
        # Statistiques
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Patients malades", len(df[df['Classification Préparateur'] == 'malade']))
        with col2:
            st.metric("Patients sains", len(df[df['Classification Préparateur'] == 'sain']))
        
        # Filtres
        st.subheader("Filtres")
        col1, col2 = st.columns(2)
        
        with col1:
            filter_classification = st.selectbox(
                "Filtrer par classification",
                ['Tous', 'malade', 'sain']
            )
        
        with col2:
            filter_priority = st.selectbox(
                "Filtrer par priorité",
                ['Toutes', 'Haute (malades)', 'Basse (sains)']
            )
        
        # Appliquer les filtres
        filtered_df = df.copy()
        
        if filter_classification != 'Tous':
            filtered_df = filtered_df[filtered_df['Classification Préparateur'] == filter_classification]
        
        if filter_priority != 'Toutes':
            if filter_priority == 'Haute (malades)':
                filtered_df = filtered_df[filtered_df['Priorité'] == 2]
            else:
                filtered_df = filtered_df[filtered_df['Priorité'] == 1]
        
        # Afficher le tableau sans mise en évidence, texte en blanc
        styled_df = filtered_df.style.set_properties(
            subset=filtered_df.columns,
            **{'color': '#ffffff'}  # Texte blanc pour toutes les lignes
        )
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Sélection d'un patient pour revue détaillée
        st.subheader("Revue Détaillée")
        
        if not filtered_df.empty:
            selected_image_id = st.selectbox(
                "Sélectionner un patient pour revue",
                filtered_df['ID Image'].tolist()
            )
            
            if selected_image_id:
                self._render_patient_detail(selected_image_id)
        else:
            st.info("Aucun patient ne correspond aux filtres")
    
    def _render_patient_detail(self, image_id):
        """Affiche la vue détaillée d'un patient"""
        image = self.data_manager.get_image(image_id)
        if not image:
            st.error("Image non trouvée")
            return
        
        patient_id = image.get('patient_id')
        patient = self.data_manager.get_patient_by_id(patient_id)
        prediction = self.data_manager.get_prediction_by_image(image_id)
        annotation = self.data_manager.get_annotation_by_image(image_id)
        
        # Informations du patient
        st.subheader(f"Patient: {patient_id}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**Date Examen:** {image.get('exam_date', 'N/A')}")
        with col2:
            if patient:
                st.write(f"**Sexe:** {patient.get('metadata', {}).get('sex', 'N/A')}")
        with col3:
            if patient:
                st.write(f"**Âge:** {patient.get('metadata', {}).get('age', 'N/A')}")
        with col4:
            st.write(f"**Position:** {image.get('patient_position', 'N/A')}")
        
        # Afficher l'image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Radiographie")
            image_path = image.get('image_path')
            if image_path and os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
            else:
                st.warning("Image non disponible")
        
        with col2:
            st.subheader("Informations")
            
            # Prédiction du modèle
            if prediction:
                st.write("**Prédiction Modèle:**")
                if prediction.get('label') == 'malade':
                    st.error(f"🔴 {prediction.get('label')}")
                else:
                    st.success(f"🟢 {prediction.get('label')}")
            
            # Classification du préparateur
            if annotation and annotation.get('user_role') == 'Préparateur':
                st.write("**Classification Préparateur:**")
                if annotation.get('label') == 'malade':
                    st.error(f"🔴 {annotation.get('label')}")
                else:
                    st.success(f"🟢 {annotation.get('label')}")
                st.write(f"**Par:** {annotation.get('user_name', 'N/A')}")
                st.write(f"**Version:** {annotation.get('version', 1)}")
                
                if annotation.get('notes'):
                    st.write("**Notes Préparateur:**")
                    st.info(annotation.get('notes'))
            
            # Informations supplémentaires
            if annotation and annotation.get('additional_info'):
                info = annotation.get('additional_info', {})
                st.write("**Informations Complémentaires:**")
                
                if info.get('symptoms'):
                    st.write(f"**Symptômes:** {info.get('symptoms')}")
                if info.get('comorbidities'):
                    st.write(f"**Comorbidités:** {info.get('comorbidities')}")
                if info.get('spo2'):
                    st.write(f"**SpO₂:** {info.get('spo2')}%")
                if info.get('temperature'):
                    st.write(f"**Température:** {info.get('temperature')}°C")
                if info.get('crp'):
                    st.write(f"**CRP:** {info.get('crp')} mg/L")
                if info.get('image_quality'):
                    st.write(f"**Qualité image:** {info.get('image_quality')}")
                if info.get('urgency'):
                    urgency_color = {
                        'Normale': '🟢',
                        'Élevée': '🟡',
                        'Critique': '🔴'
                    }
                    st.write(f"**Urgence:** {urgency_color.get(info.get('urgency'), '')} {info.get('urgency')}")
        
        # Section pour démarrer le traitement
        st.divider()
        st.subheader("💊 Démarrer le Traitement")
        
        # Vérifier si un traitement existe déjà
        existing_treatment = annotation.get('additional_info', {}).get('treatment') if annotation else None
        
        if existing_treatment:
            st.info(f"**Traitement en cours:** {existing_treatment.get('action_type', 'N/A').title()} - Statut: {existing_treatment.get('status', 'N/A')}")
            st.write(f"**Démarré le:** {existing_treatment.get('started_at', 'N/A')}")
            if existing_treatment.get('details', {}).get('notes'):
                st.write(f"**Notes:** {existing_treatment.get('details', {}).get('notes')}")
        else:
            # Formulaire pour démarrer le traitement directement
            with st.form(f"start_treatment_form_{image_id}"):
                st.subheader("Type d'Action")
                
                action_type = st.radio(
                    "Sélectionner le type d'action",
                    ['prescription', 'examens', 'hospitalisation', 'orientation'],
                    help="Choisissez l'action à entreprendre pour ce patient"
                )
                
                # Détails selon le type d'action
                details = {}
                
                if action_type == 'prescription':
                    st.subheader("Détails de la Prescription")
                    details['medication'] = st.text_input("Médicament(s)", placeholder="Ex: Amoxicilline 500mg", key=f"med_{image_id}")
                    details['dosage'] = st.text_input("Posologie", placeholder="Ex: 3x par jour pendant 7 jours", key=f"dosage_{image_id}")
                    details['duration'] = st.number_input("Durée (jours)", min_value=1, value=7, key=f"dur_{image_id}")
                
                elif action_type == 'examens':
                    st.subheader("Examens Complémentaires")
                    exam_types = st.multiselect(
                        "Type d'examen(s)",
                        ['Scanner thoracique', 'Prise de sang', 'ECG', 'Échographie', 'Autre'],
                        key=f"exam_{image_id}"
                    )
                    details['exam_types'] = exam_types
                    if 'Autre' in exam_types:
                        details['other_exam'] = st.text_input("Préciser l'autre examen", key=f"other_exam_{image_id}")
                    details['urgency'] = st.selectbox(
                        "Urgence",
                        ['Normale', 'Urgente', 'Très urgente'],
                        key=f"urgency_{image_id}"
                    )
                
                elif action_type == 'hospitalisation':
                    st.subheader("Détails d'Hospitalisation")
                    details['department'] = st.selectbox(
                        "Service",
                        ['Médecine', 'Soins intensifs', 'Urgences', 'Pneumologie', 'Autre'],
                        key=f"dept_{image_id}"
                    )
                    details['reason'] = st.text_area("Motif d'hospitalisation", key=f"reason_hosp_{image_id}")
                    details['estimated_duration'] = st.number_input("Durée estimée (jours)", min_value=1, value=3, key=f"est_dur_{image_id}")
                
                elif action_type == 'orientation':
                    st.subheader("Orientation")
                    details['destination'] = st.selectbox(
                        "Orienter vers",
                        ['Médecin spécialiste', 'Service hospitalier', 'Soins à domicile', 'Suivi ambulatoire', 'Autre'],
                        key=f"dest_{image_id}"
                    )
                    details['reason'] = st.text_area("Motif d'orientation", key=f"reason_orient_{image_id}")
                
                notes = st.text_area("Notes complémentaires", placeholder="Ajoutez des notes sur le traitement...", key=f"notes_treat_{image_id}")
                if notes:
                    details['notes'] = notes
                
                submitted = st.form_submit_button("✅ Démarrer le Traitement", type="primary")
                
                if submitted:
                    self.data_manager.start_treatment(
                        image_id,
                        st.session_state.current_user_name,
                        action_type,
                        details
                    )
                    st.success(f"✅ Traitement démarré pour le patient {patient_id}")
                    st.rerun()
        
        # Historique des modifications
        st.subheader("Historique des Modifications")
        audit_log = self.data_manager.get_audit_log(image_id)
        
        if audit_log:
            for entry in sorted(audit_log, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]:
                st.write(f"**{entry.get('timestamp', 'N/A')}** - {entry.get('user_name', 'N/A')}")
                st.write(f"Action: {entry.get('action', 'N/A')}")
                details = entry.get('details', {})
                if 'old_label' in details and 'new_label' in details:
                    st.write(f"Changement: {details.get('old_label')} → {details.get('new_label')}")
                st.divider()
        else:
            st.info("Aucun historique disponible")
    
    def _render_treatment_tab(self):
        """Onglet pour démarrer le traitement"""
        st.subheader("💊 Démarrer un Traitement")
        
        # Récupérer les patients validés par le médecin (pas encore en traitement)
        images = self.data_manager.get_all_images()
        medical_annotations = {a.get('image_id'): a for a in self.data_manager.get_all_annotations() 
                              if a.get('user_role') == 'Médecin'}
        
        validated_patients = []
        for img in images:
            if img['id'] in medical_annotations:
                ann = medical_annotations[img['id']]
                # Vérifier si pas déjà en traitement
                if not ann.get('additional_info', {}).get('treatment'):
                    validated_patients.append({
                        'image': img,
                        'annotation': ann
                    })
        
        if not validated_patients:
            st.info("Aucun patient validé disponible pour démarrer un traitement")
        else:
            st.write(f"**{len(validated_patients)} patient(s) validé(s) disponible(s)**")
            
            # Sélection d'un patient
            patient_options = {
                f"{p['image'].get('patient_id', 'N/A')} - {p['image'].get('exam_date', 'N/A')}": p['image']['id']
                for p in validated_patients
            }
            
            # Si on vient de la vue détaillée, pré-sélectionner le patient
            default_index = 0
            if 'start_treatment_for' in st.session_state and st.session_state.start_treatment_for:
                target_id = st.session_state.start_treatment_for
                for idx, (label, img_id) in enumerate(patient_options.items()):
                    if img_id == target_id:
                        default_index = idx
                        break
                # Nettoyer la variable de session après utilisation
                del st.session_state.start_treatment_for
            
            selected_patient_label = st.selectbox(
                "Sélectionner un patient pour démarrer le traitement",
                list(patient_options.keys()),
                index=default_index
            )
            
            if selected_patient_label:
                selected_image_id = patient_options[selected_patient_label]
                selected_patient = next(p for p in validated_patients if p['image']['id'] == selected_image_id)
                
                # Afficher les informations du patient
                image = selected_patient['image']
                annotation = selected_patient['annotation']
                patient_id = image.get('patient_id')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ID Patient:** {patient_id}")
                    st.write(f"**Date Examen:** {image.get('exam_date', 'N/A')}")
                with col2:
                    st.write(f"**Diagnostic:** {annotation.get('label', 'N/A')}")
                    st.write(f"**Validé par:** {annotation.get('user_name', 'N/A')}")
                
                # Formulaire pour démarrer le traitement
                with st.form("start_treatment_form"):
                    st.subheader("Type d'Action")
                    
                    action_type = st.radio(
                        "Sélectionner le type d'action",
                        ['prescription', 'examens', 'hospitalisation', 'orientation'],
                        help="Choisissez l'action à entreprendre pour ce patient"
                    )
                    
                    # Détails selon le type d'action
                    details = {}
                    
                    if action_type == 'prescription':
                        st.subheader("Détails de la Prescription")
                        details['medication'] = st.text_input("Médicament(s)", placeholder="Ex: Amoxicilline 500mg")
                        details['dosage'] = st.text_input("Posologie", placeholder="Ex: 3x par jour pendant 7 jours")
                        details['duration'] = st.number_input("Durée (jours)", min_value=1, value=7)
                    
                    elif action_type == 'examens':
                        st.subheader("Examens Complémentaires")
                        exam_types = st.multiselect(
                            "Type d'examen(s)",
                            ['Scanner thoracique', 'Prise de sang', 'ECG', 'Échographie', 'Autre']
                        )
                        details['exam_types'] = exam_types
                        if 'Autre' in exam_types:
                            details['other_exam'] = st.text_input("Préciser l'autre examen")
                        details['urgency'] = st.selectbox(
                            "Urgence",
                            ['Normale', 'Urgente', 'Très urgente']
                        )
                    
                    elif action_type == 'hospitalisation':
                        st.subheader("Détails d'Hospitalisation")
                        details['department'] = st.selectbox(
                            "Service",
                            ['Médecine', 'Soins intensifs', 'Urgences', 'Pneumologie', 'Autre']
                        )
                        details['reason'] = st.text_area("Motif d'hospitalisation")
                        details['estimated_duration'] = st.number_input("Durée estimée (jours)", min_value=1, value=3)
                    
                    elif action_type == 'orientation':
                        st.subheader("Orientation")
                        details['destination'] = st.selectbox(
                            "Orienter vers",
                            ['Médecin spécialiste', 'Service hospitalier', 'Soins à domicile', 'Suivi ambulatoire', 'Autre']
                        )
                        details['reason'] = st.text_area("Motif d'orientation")
                    
                    notes = st.text_area("Notes complémentaires", placeholder="Ajoutez des notes sur le traitement...")
                    if notes:
                        details['notes'] = notes
                    
                    submitted = st.form_submit_button("✅ Démarrer le Traitement", type="primary")
                    
                    if submitted:
                        self.data_manager.start_treatment(
                            selected_image_id,
                            st.session_state.current_user_name,
                            action_type,
                            details
                        )
                        st.success(f"✅ Traitement démarré pour le patient {patient_id}")
                        st.rerun()
    
    def _render_treatment_followup_tab(self):
        """Onglet pour suivre les patients en traitement"""
        st.subheader("📋 Suivi du Traitement")
        
        patients_in_treatment = self.data_manager.get_patients_in_treatment()
        
        if not patients_in_treatment:
            st.info("Aucun patient en traitement pour le moment")
        else:
            st.write(f"**{len(patients_in_treatment)} patient(s) en traitement**")
            
            # Statistiques
            col1, col2, col3 = st.columns(3)
            with col1:
                en_traitement = len([p for p in patients_in_treatment if p['treatment'].get('status') == 'en_traitement'])
                st.metric("En traitement", en_traitement)
            with col2:
                en_attente = len([p for p in patients_in_treatment if p['treatment'].get('status') == 'en_attente_examens'])
                st.metric("En attente d'examens", en_attente)
            with col3:
                hospitalises = len([p for p in patients_in_treatment if p['treatment'].get('status') == 'hospitalise'])
                st.metric("Hospitalisés", hospitalises)
            
            # Liste des patients
            for patient_data in patients_in_treatment:
                image = patient_data['image']
                annotation = patient_data['annotation']
                treatment = patient_data['treatment']
                
                with st.expander(f"Patient {image.get('patient_id', 'N/A')} - {treatment.get('action_type', 'N/A').title()} - {treatment.get('status', 'N/A')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**ID Patient:** {image.get('patient_id', 'N/A')}")
                        st.write(f"**Date Examen:** {image.get('exam_date', 'N/A')}")
                        st.write(f"**Diagnostic:** {annotation.get('label', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Type d'action:** {treatment.get('action_type', 'N/A').title()}")
                        st.write(f"**Statut:** {treatment.get('status', 'N/A')}")
                        st.write(f"**Démarré le:** {treatment.get('started_at', 'N/A')}")
                        st.write(f"**Par:** {treatment.get('started_by', 'N/A')}")
                    
                    # Afficher les détails selon le type
                    st.subheader("Détails du Traitement")
                    details = treatment.get('details', {})
                    
                    if treatment.get('action_type') == 'prescription':
                        st.write(f"**Médicament:** {details.get('medication', 'N/A')}")
                        st.write(f"**Posologie:** {details.get('dosage', 'N/A')}")
                        st.write(f"**Durée:** {details.get('duration', 'N/A')} jours")
                    
                    elif treatment.get('action_type') == 'examens':
                        st.write(f"**Examens demandés:** {', '.join(details.get('exam_types', []))}")
                        if details.get('other_exam'):
                            st.write(f"**Autre examen:** {details.get('other_exam')}")
                        st.write(f"**Urgence:** {details.get('urgency', 'N/A')}")
                    
                    elif treatment.get('action_type') == 'hospitalisation':
                        st.write(f"**Service:** {details.get('department', 'N/A')}")
                        st.write(f"**Motif:** {details.get('reason', 'N/A')}")
                        st.write(f"**Durée estimée:** {details.get('estimated_duration', 'N/A')} jours")
                    
                    elif treatment.get('action_type') == 'orientation':
                        st.write(f"**Destination:** {details.get('destination', 'N/A')}")
                        st.write(f"**Motif:** {details.get('reason', 'N/A')}")
                    
                    if details.get('notes'):
                        st.write(f"**Notes:** {details.get('notes')}")
                    
                    # Vérifier si le patient est déjà validé
                    medical_annotations = [a for a in self.data_manager.get_all_annotations() 
                                         if a.get('image_id') == image['id'] and a.get('user_role') == 'Médecin']
                    is_validated = len(medical_annotations) > 0
                    
                    if is_validated:
                        latest_medical = max(medical_annotations, key=lambda x: x.get('version', 0))
                        st.success(f"✅ **Patient validé** - Diagnostic: {latest_medical.get('label', 'N/A')} (Version {latest_medical.get('version', 1)})")
                        st.write(f"**Validé par:** {latest_medical.get('user_name', 'N/A')}")
                        st.write(f"**Date de validation:** {latest_medical.get('created_at', 'N/A')}")
                        if latest_medical.get('notes'):
                            st.write(f"**Commentaire clinique:** {latest_medical.get('notes')}")
                    
                    # Afficher l'image si disponible
                    image_path = image.get('image_path')
                    if image_path and os.path.exists(image_path):
                        st.subheader("Radiographie")
                        st.image(image_path, use_container_width=True)
                    
                    # Mise à jour du statut
                    st.divider()
                    st.subheader("Mettre à jour le Statut")
                    
                    with st.form(f"update_status_form_{image['id']}"):
                        # Options de statut sans "termine"
                        status_options = ['en_traitement', 'en_attente_examens', 'hospitalise']
                        current_status = treatment.get('status', 'en_traitement')
                        
                        # Si le statut actuel est "termine", ne pas l'afficher dans les options
                        if current_status == 'termine':
                            st.info("✅ Ce traitement est terminé. Allez dans l'onglet '📝 Finalisation du dossier' pour consigner le verdict final.")
                            new_status = 'termine'  # Garder le statut actuel
                        else:
                            new_status = st.selectbox(
                                "Nouveau statut",
                                status_options,
                                index=status_options.index(current_status) if current_status in status_options else 0,
                                key=f"status_{image['id']}"
                            )
                        
                        status_notes = st.text_area(
                            "Notes sur le changement de statut",
                            key=f"status_notes_{image['id']}"
                        )
                        
                        if st.form_submit_button("💾 Mettre à jour le Statut", type="primary"):
                            self.data_manager.update_treatment_status(
                                image['id'],
                                st.session_state.current_user_name,
                                new_status,
                                status_notes
                            )
                            st.success("✅ Statut mis à jour")
                            st.rerun()
                    
                    # Bouton pour envoyer vers finalisation
                    st.divider()
                    st.subheader("Finalisation")
                    
                    if treatment.get('status') == 'termine':
                        st.info("✅ Ce traitement a déjà été envoyé pour finalisation. Allez dans l'onglet '📝 Finalisation du dossier' pour consigner le verdict final.")
                    else:
                        if st.button("📝 Envoyer pour Finalisation", type="primary", key=f"finalize_{image['id']}"):
                            # Mettre le statut à "termine" pour déplacer vers l'onglet de finalisation
                            self.data_manager.update_treatment_status(
                                image['id'],
                                st.session_state.current_user_name,
                                'termine',
                                "Traitement terminé - Envoyé pour finalisation"
                            )
                            st.success("✅ Dossier envoyé pour finalisation. Allez dans l'onglet '📝 Finalisation du dossier' pour consigner le verdict final.")
                            st.rerun()
    
    def _render_finalization_tab(self):
        """Onglet pour finaliser les dossiers des patients avec traitement terminé"""
        st.subheader("📝 Finalisation du Dossier")
        
        completed_patients = self.data_manager.get_patients_with_completed_treatment()
        
        if not completed_patients:
            st.info("Aucun patient avec traitement terminé pour le moment")
        else:
            st.write(f"**{len(completed_patients)} patient(s) avec traitement terminé**")
            
            # Liste des patients
            for patient_data in completed_patients:
                image = patient_data['image']
                annotation = patient_data['annotation']
                treatment = patient_data['treatment']
                
                # Vérifier si le verdict final a déjà été consigné
                additional_info = annotation.get('additional_info', {})
                has_final_verdict = additional_info.get('ground_truth') is not None
                
                with st.expander(f"Patient {image.get('patient_id', 'N/A')} - {treatment.get('action_type', 'N/A').title()} - {'✅ Verdict consigné' if has_final_verdict else '⏳ En attente de verdict'}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**ID Patient:** {image.get('patient_id', 'N/A')}")
                        st.write(f"**Date Examen:** {image.get('exam_date', 'N/A')}")
                        st.write(f"**Diagnostic initial:** {annotation.get('label', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Type d'action:** {treatment.get('action_type', 'N/A').title()}")
                        st.write(f"**Statut:** {treatment.get('status', 'N/A')}")
                        st.write(f"**Terminé le:** {treatment.get('updated_at', treatment.get('started_at', 'N/A'))}")
                    
                    # Afficher les détails du traitement
                    st.subheader("Détails du Traitement")
                    details = treatment.get('details', {})
                    
                    if treatment.get('action_type') == 'prescription':
                        st.write(f"**Médicament:** {details.get('medication', 'N/A')}")
                        st.write(f"**Posologie:** {details.get('dosage', 'N/A')}")
                        st.write(f"**Durée:** {details.get('duration', 'N/A')} jours")
                    
                    elif treatment.get('action_type') == 'examens':
                        st.write(f"**Examens demandés:** {', '.join(details.get('exam_types', []))}")
                        if details.get('other_exam'):
                            st.write(f"**Autre examen:** {details.get('other_exam')}")
                        st.write(f"**Urgence:** {details.get('urgency', 'N/A')}")
                    
                    elif treatment.get('action_type') == 'hospitalisation':
                        st.write(f"**Service:** {details.get('department', 'N/A')}")
                        st.write(f"**Motif:** {details.get('reason', 'N/A')}")
                        st.write(f"**Durée estimée:** {details.get('estimated_duration', 'N/A')} jours")
                    
                    elif treatment.get('action_type') == 'orientation':
                        st.write(f"**Destination:** {details.get('destination', 'N/A')}")
                        st.write(f"**Motif:** {details.get('reason', 'N/A')}")
                    
                    if details.get('notes'):
                        st.write(f"**Notes:** {details.get('notes')}")
                    
                    # Afficher l'image si disponible
                    image_path = image.get('image_path')
                    if image_path and os.path.exists(image_path):
                        st.subheader("Radiographie")
                        st.image(image_path, use_container_width=True)
                    
                    # Formulaire pour consigner le verdict final
                    st.divider()
                    st.subheader("Verdict Final du Traitement")
                    
                    with st.form(f"finalization_form_{image['id']}"):
                        result_notes = st.text_area(
                            "Notes de résultat final",
                            placeholder="Détails sur le résultat final, traitement administré, évolution du patient, etc.",
                            value=additional_info.get('ground_truth_notes', ''),
                            key=f"result_notes_final_{image['id']}"
                        )
                        
                        # Diagnostic final (peut être différent du diagnostic initial après traitement)
                        current_diagnostic = annotation.get('label', 'sain')
                        final_diagnostic = st.radio(
                            "Diagnostic final après traitement",
                            ['sain', 'malade'],
                            index=0 if current_diagnostic == 'sain' else 1,
                            help="Diagnostic final basé sur l'évolution après traitement",
                            key=f"final_diagnostic_{image['id']}"
                        )
                        
                        outcome = st.selectbox(
                            "Issue du traitement",
                            ['Guérison complète', 'Amélioration', 'Stable', 'Aggravation', 'Décès', 'Autre'],
                            key=f"outcome_{image['id']}"
                        )
                        
                        if outcome == 'Autre':
                            outcome_other = st.text_input("Préciser l'issue", key=f"outcome_other_{image['id']}")
                        else:
                            outcome_other = None
                        
                        submitted = st.form_submit_button("✅ Consigner le Verdict Final", type="primary")
                        
                        if submitted:
                            # Mettre à jour l'annotation avec le verdict final
                            updated_additional_info = {
                                **additional_info,
                                'ground_truth_notes': result_notes,
                                'final_diagnostic': final_diagnostic,
                                'treatment_outcome': outcome,
                                'treatment_outcome_other': outcome_other if outcome == 'Autre' else None,
                                'finalized_at': datetime.now().isoformat()
                            }
                            
                            medical_annotation_data = {
                                'label': final_diagnostic,
                                'confidence': annotation.get('confidence', 0.9),
                                'notes': annotation.get('notes', ''),
                                'user_role': 'Médecin',
                                'additional_info': updated_additional_info
                            }
                            
                            self.data_manager.update_annotation(
                                image['id'],
                                st.session_state.current_user_name,
                                medical_annotation_data
                            )
                            
                            st.success("✅ Verdict final consigné avec succès")
                            st.rerun()
                    
                    # Afficher le verdict si déjà consigné
                    if has_final_verdict:
                        st.divider()
                        st.subheader("Verdict Déjà Consigné")
                        st.write(f"**Diagnostic final:** {additional_info.get('final_diagnostic', annotation.get('label', 'N/A'))}")
                        st.write(f"**Issue du traitement:** {additional_info.get('treatment_outcome', 'N/A')}")
                        if additional_info.get('ground_truth_notes'):
                            st.write(f"**Notes:** {additional_info.get('ground_truth_notes')}")
                        st.write(f"**Consigné le:** {additional_info.get('finalized_at', 'N/A')}")
    
    def _generate_complete_export(self, validated_images, export_format, include_images, split_dataset):
        """Génère un export complet avec plusieurs formats"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = f"data/exports/export_{timestamp}"
            os.makedirs(export_dir, exist_ok=True)
            
            results = {}
            
            # Préparer les données complètes
            export_data = []
            patients_data = []
            
            for v in validated_images:
                img = v['image']
                ann = v['annotation']
                patient = self.data_manager.get_patient_by_id(img.get('patient_id'))
                pred = self.data_manager.get_prediction_by_image(img['id'])
                
                # Données pour CSV et analyses statistiques
                patient_row = {
                    'image_id': img['id'],
                    'patient_id': img.get('patient_id', 'N/A'),
                    'label': ann.get('label', 'N/A'),  # malade ou sain
                    'label_numeric': 1 if ann.get('label') == 'malade' else 0,
                    'sexe': patient.get('metadata', {}).get('sex', 'N/A') if patient else 'N/A',
                    'age': patient.get('metadata', {}).get('age', 'N/A') if patient else 'N/A',
                    'zone_geographique': patient.get('metadata', {}).get('institution_name', 'N/A') if patient else 'N/A',
                    'station': patient.get('metadata', {}).get('station_name', 'N/A') if patient else 'N/A',
                    'date_examen': img.get('exam_date', 'N/A'),
                    'modality': img.get('modality', 'N/A'),
                    'body_part': img.get('body_part', 'N/A'),
                    'patient_position': img.get('patient_position', 'N/A'),
                    'view_position': img.get('view_position', 'N/A'),
                    'prediction_originale': pred.get('label', 'N/A') if pred else 'N/A',
                    'confidence_modele': pred.get('confidence', 0.0) if pred else 0.0,
                    'confidence_medecin': ann.get('confidence', 0.0),
                    'ground_truth': ann.get('additional_info', {}).get('ground_truth', 'Non déterminé'),
                    'validated_by': ann.get('user_name', 'N/A'),
                    'validated_at': ann.get('created_at', 'N/A'),
                    'image_path': img.get('image_path', 'N/A')
                }
                
                # Ajouter les informations complémentaires si disponibles
                additional_info = ann.get('additional_info', {})
                if additional_info:
                    patient_row['symptoms'] = additional_info.get('symptoms', '')
                    patient_row['comorbidities'] = additional_info.get('comorbidities', '')
                    patient_row['spo2'] = additional_info.get('spo2', '')
                    patient_row['temperature'] = additional_info.get('temperature', '')
                    patient_row['crp'] = additional_info.get('crp', '')
                    patient_row['image_quality'] = additional_info.get('image_quality', '')
                    patient_row['urgency'] = additional_info.get('urgency', '')
                
                patients_data.append(patient_row)
                
                # Données pour JSON complet
                export_data.append({
                    'image_id': img['id'],
                    'patient_id': img.get('patient_id'),
                    'image_path': img.get('image_path'),
                    'final_label': ann.get('label'),
                    'label_numeric': 1 if ann.get('label') == 'malade' else 0,
                    'ground_truth': ann.get('additional_info', {}).get('ground_truth'),
                    'confidence': ann.get('confidence'),
                    'notes': ann.get('notes', ''),
                    'validated_by': ann.get('user_name'),
                    'validated_at': ann.get('created_at'),
                    'patient_metadata': patient.get('metadata', {}) if patient else {},
                    'original_prediction': pred if pred else None,
                    'clinical_metadata': additional_info
                })
            
            # Générer les exports selon le format demandé
            if export_format in ['CSV (Analyses statistiques)', 'Tous les formats']:
                csv_path = os.path.join(export_dir, f"patients_data_{timestamp}.csv")
                df = pd.DataFrame(patients_data)
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                results['CSV'] = csv_path
            
            if export_format in ['JSON complet', 'Tous les formats']:
                json_path = os.path.join(export_dir, f"export_complet_{timestamp}.json")
                export_metadata = {
                    'export_date': datetime.now().isoformat(),
                    'total_samples': len(export_data),
                    'pneumonia_count': sum(1 for d in export_data if d['final_label'] == 'malade'),
                    'healthy_count': sum(1 for d in export_data if d['final_label'] == 'sain'),
                    'export_version': '1.0',
                    'samples': export_data
                }
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(export_metadata, f, indent=2, ensure_ascii=False, default=str)
                results['JSON'] = json_path
            
            if export_format in ['Structure de dossiers (Réentraînement)', 'Tous les formats']:
                # Créer la structure de dossiers pour le réentraînement
                training_dir = os.path.join(export_dir, 'training_data')
                
                if split_dataset:
                    train_dir = os.path.join(training_dir, 'train')
                    val_dir = os.path.join(training_dir, 'validation')
                    test_dir = os.path.join(training_dir, 'test')
                    
                    for label in ['malade', 'sain']:
                        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
                        os.makedirs(os.path.join(val_dir, label), exist_ok=True)
                        os.makedirs(os.path.join(test_dir, label), exist_ok=True)
                else:
                    for label in ['malade', 'sain']:
                        os.makedirs(os.path.join(training_dir, label), exist_ok=True)
                
                # Créer le fichier labels.csv
                labels_data = []
                
                # Copier les images et créer les labels
                import random
                random.seed(42)  # Pour la reproductibilité
                
                for v in validated_images:
                    img = v['image']
                    ann = v['annotation']
                    label = ann.get('label', 'sain')
                    image_path = img.get('image_path')
                    
                    if not image_path or not os.path.exists(image_path):
                        continue
                    
                    # Déterminer le dossier de destination
                    if split_dataset:
                        rand = random.random()
                        if rand < 0.7:
                            dest_folder = 'train'
                        elif rand < 0.85:
                            dest_folder = 'validation'
                        else:
                            dest_folder = 'test'
                    else:
                        dest_folder = None
                    
                    # Copier l'image
                    if include_images:
                        file_ext = os.path.splitext(image_path)[1] or '.png'
                        new_filename = f"{img['id']}{file_ext}"
                        
                        if dest_folder:
                            dest_path = os.path.join(training_dir, dest_folder, label, new_filename)
                        else:
                            dest_path = os.path.join(training_dir, label, new_filename)
                        
                        try:
                            shutil.copy2(image_path, dest_path)
                            # Récupérer l'issue du traitement depuis les annotations
                            additional_info = ann.get('additional_info', {})
                            treatment_outcome = additional_info.get('treatment_outcome', 'N/A')
                            if treatment_outcome == 'Autre' and additional_info.get('treatment_outcome_other'):
                                treatment_outcome = additional_info.get('treatment_outcome_other')
                            
                            labels_data.append({
                                'image_path': os.path.relpath(dest_path, training_dir),
                                'label': label,
                                'label_numeric': 1 if label == 'malade' else 0,
                                'patient_id': img.get('patient_id'),
                                'image_id': img['id'],
                                'treatment_outcome': treatment_outcome
                            })
                        except Exception as e:
                            st.warning(f"Impossible de copier {image_path}: {e}")
                    else:
                        # Juste ajouter la référence
                        # Récupérer l'issue du traitement depuis les annotations
                        additional_info = ann.get('additional_info', {})
                        treatment_outcome = additional_info.get('treatment_outcome', 'N/A')
                        if treatment_outcome == 'Autre' and additional_info.get('treatment_outcome_other'):
                            treatment_outcome = additional_info.get('treatment_outcome_other')
                        
                        labels_data.append({
                            'image_path': image_path,
                            'label': label,
                            'label_numeric': 1 if label == 'malade' else 0,
                            'patient_id': img.get('patient_id'),
                            'image_id': img['id'],
                            'treatment_outcome': treatment_outcome
                        })
                
                # Sauvegarder labels.csv
                labels_df = pd.DataFrame(labels_data)
                labels_csv_path = os.path.join(training_dir, 'labels.csv')
                labels_df.to_csv(labels_csv_path, index=False, encoding='utf-8-sig')
                
                # Créer un fichier ZIP
                zip_path = os.path.join(export_dir, f"training_data_{timestamp}.zip")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(training_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, training_dir)
                            zipf.write(file_path, arcname)
                
                results['Structure de dossiers'] = zip_path
            
            return results
            
        except Exception as e:
            st.error(f"Erreur lors de l'export: {str(e)}")
            return None
    
    def _render_results_tab(self):
        """Onglet de résultats et export"""
        st.subheader("Résultats Finalisés et Export")
        
        # Récupérer les images validées par le médecin
        images = self.data_manager.get_all_images()
        medical_annotations = {a.get('image_id'): a for a in self.data_manager.get_all_annotations() 
                              if a.get('user_role') == 'Médecin'}
        
        validated_images = []
        for img in images:
            if img['id'] in medical_annotations:
                validated_images.append({
                    'image': img,
                    'annotation': medical_annotations[img['id']]
                })
        
        if not validated_images:
            st.info("Aucun patient validé pour le moment")
            return
        
        st.write(f"**{len(validated_images)} patient(s) validé(s)**")
        
        # Statistiques
        confirmed_sick = sum(1 for v in validated_images if v['annotation'].get('label') == 'malade')
        confirmed_healthy = sum(1 for v in validated_images if v['annotation'].get('label') == 'sain')
        
        col1, col2 = st.columns(2)
        col1.metric("Pneumonie confirmée", confirmed_sick)
        col2.metric("Absence de pneumonie", confirmed_healthy)
        
        # Liste des patients validés
        st.subheader("Liste des Patients Validés")
        
        df_results = pd.DataFrame([{
            'ID Image': v['image']['id'],
            'ID Patient': v['image'].get('patient_id', 'N/A'),
            'Date Examen': v['image'].get('exam_date', 'N/A'),
            'Diagnostic Final': v['annotation'].get('label', 'N/A'),
            'Confiance': v['annotation'].get('confidence', 0.0),
            'Vérité Terrain': v['annotation'].get('additional_info', {}).get('ground_truth', 'Non déterminé'),
            'Validé par': v['annotation'].get('user_name', 'N/A'),
            'Date Validation': v['annotation'].get('created_at', 'N/A')
        } for v in validated_images])
        
        st.dataframe(df_results, use_container_width=True)
        
        # Sélection des patients à finaliser
        st.subheader("Finalisation du Lot")
        
        selected_images = st.multiselect(
            "Sélectionner les patients à finaliser",
            [v['image']['id'] for v in validated_images],
            default=[v['image']['id'] for v in validated_images]
        )
        
        if st.button("✅ Marquer comme Finalisé", type="primary"):
            self.data_manager.mark_batch_finalized(
                selected_images,
                st.session_state.current_user_name
            )
            st.success(f"✅ {len(selected_images)} patient(s) marqué(s) comme finalisé(s)")
            st.rerun()
        
        # Export complet pour réentraînement et analyses
        st.divider()
        st.subheader("📥 Export Complet pour Réentraînement et Analyses")
        
        # Utiliser les images validées (pas seulement finalisées) pour l'export
        if not validated_images:
            st.info("Aucun patient validé disponible pour l'export")
        else:
            st.write(f"**{len(validated_images)} patient(s) validé(s) disponible(s) pour export**")
            
            # Options d'export
            export_format = st.radio(
                "Format d'export",
                ['CSV (Analyses statistiques)', 'Structure de dossiers (Réentraînement)', 'JSON complet', 'Tous les formats'],
                help="Choisissez le format d'export selon votre besoin"
            )
            
            include_images = st.checkbox(
                "Inclure les images dans l'export",
                value=True,
                help="Crée une copie des images dans le dossier d'export (pour réentraînement)"
            )
            
            split_dataset = st.checkbox(
                "Séparer en train/validation/test",
                value=False,
                help="Divise automatiquement les données (70% train, 15% validation, 15% test)"
            )
            
            if st.button("📥 Générer l'Export", type="primary"):
                with st.spinner("Préparation de l'export en cours..."):
                    export_results = self._generate_complete_export(
                        validated_images,
                        export_format,
                        include_images,
                        split_dataset
                    )
                    
                    if export_results:
                        st.success("✅ Export généré avec succès !")
                        
                        # Afficher les fichiers générés
                        for file_type, file_path in export_results.items():
                            if file_path and os.path.exists(file_path):
                                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Taille en MB
                                st.write(f"**{file_type}:** {os.path.basename(file_path)} ({file_size:.2f} MB)")
                                
                                # Bouton de téléchargement
                                with open(file_path, 'rb') as f:
                                    st.download_button(
                                        label=f"📥 Télécharger {file_type}",
                                        data=f.read(),
                                        file_name=os.path.basename(file_path),
                                        mime="application/zip" if file_path.endswith('.zip') else 
                                             "text/csv" if file_path.endswith('.csv') else
                                             "application/json",
                                        key=f"download_{file_type}"
                                    )
                    else:
                        st.error("❌ Erreur lors de la génération de l'export")
        
        # Historique des patients validés
        st.divider()
        st.subheader("📜 Historique des Patients")
        
        # Récupérer tous les patients validés par le médecin
        images = self.data_manager.get_all_images()
        all_annotations = self.data_manager.get_all_annotations()
        
        # Récupérer toutes les annotations médicales (peut y en avoir plusieurs par image)
        medical_annotations_by_image = {}
        for ann in all_annotations:
            if ann.get('user_role') == 'Médecin':
                image_id = ann.get('image_id')
                if image_id:
                    if image_id not in medical_annotations_by_image:
                        medical_annotations_by_image[image_id] = []
                    medical_annotations_by_image[image_id].append(ann)
        
        # Pour chaque image avec annotation médicale, prendre la plus récente pour l'affichage
        validated_patients = []
        for img in images:
            image_id = img['id']
            if image_id in medical_annotations_by_image:
                # Prendre la dernière annotation (version la plus récente)
                medical_anns = medical_annotations_by_image[image_id]
                latest_ann = max(medical_anns, key=lambda x: x.get('version', 0))
                patient_id = img.get('patient_id', 'N/A')
                validated_patients.append({
                    'image_id': image_id,
                    'patient_id': patient_id,
                    'image': img,
                    'annotation': latest_ann,
                    'label': latest_ann.get('label', 'N/A'),
                    'validated_by': latest_ann.get('user_name', 'N/A'),
                    'validated_at': latest_ann.get('created_at', 'N/A')
                })
        
        if not validated_patients:
            st.info("ℹ️ Aucun patient validé pour le moment. Les patients validés apparaîtront ici après validation médicale.")
        else:
            st.write(f"**{len(validated_patients)} patient(s) validé(s)**")
            
            # Créer une liste déroulante pour sélectionner un patient
            patient_options = {
                f"{p['patient_id']} - {p['image'].get('exam_date', 'N/A')} - {p['label']}": p['image_id']
                for p in validated_patients
            }
            
            if not patient_options:
                st.warning("⚠️ Aucun patient disponible dans la liste")
            else:
                selected_patient_label = st.selectbox(
                    "Sélectionner un patient pour voir son historique",
                    list(patient_options.keys()),
                    key="history_patient_select"
                )
                
                if selected_patient_label:
                    selected_image_id = patient_options[selected_patient_label]
                    selected_patient = next(p for p in validated_patients if p['image_id'] == selected_image_id)
                    
                    # Afficher les informations du patient
                    st.write("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**ID Patient:** {selected_patient['patient_id']}")
                    with col2:
                        st.write(f"**Diagnostic:** {selected_patient['label']}")
                    with col3:
                        st.write(f"**Validé par:** {selected_patient['validated_by']}")
                    
                    st.write(f"**Date de validation:** {selected_patient['validated_at']}")
                    
                    # Récupérer toutes les annotations pour ce patient (historique complet)
                    patient_annotations = [a for a in all_annotations 
                                          if a.get('image_id') == selected_image_id]
                    patient_annotations.sort(key=lambda x: x.get('version', 0))
                    
                    # Récupérer l'audit log
                    audit_log = self.data_manager.get_audit_log(selected_image_id)
                    
                    # Afficher l'historique des annotations
                    st.subheader("📋 Historique des Modifications")
                    
                    if patient_annotations or audit_log:
                        # Combiner et trier par date
                        history_entries = []
                        
                        # Ajouter les annotations
                        for ann in patient_annotations:
                            history_entries.append({
                                'type': 'annotation',
                                'timestamp': ann.get('created_at', ''),
                                'user': ann.get('user_name', 'N/A'),
                                'role': ann.get('user_role', 'N/A'),
                                'version': ann.get('version', 0),
                                'label': ann.get('label', 'N/A'),
                                'confidence': ann.get('confidence', 0.0),
                                'notes': ann.get('notes', ''),
                                'data': ann
                            })
                        
                        # Ajouter les entrées d'audit
                        for entry in audit_log:
                            history_entries.append({
                                'type': 'audit',
                                'timestamp': entry.get('timestamp', ''),
                                'user': entry.get('user_name', 'N/A'),
                                'action': entry.get('action', 'N/A'),
                                'details': entry.get('details', {}),
                                'data': entry
                            })
                        
                        # Trier par timestamp
                        history_entries.sort(key=lambda x: x.get('timestamp', ''))
                        
                        # Afficher dans une liste déroulante (expandable)
                        for entry in history_entries:
                            timestamp = entry.get('timestamp', 'N/A')
                            user = entry.get('user', 'N/A')
                            
                            if entry['type'] == 'annotation':
                                role = entry.get('role', 'N/A')
                                version = entry.get('version', 0)
                                label = entry.get('label', 'N/A')
                                confidence = entry.get('confidence', 0.0)
                                notes = entry.get('notes', '')
                                
                                with st.expander(f"📝 Version {version} - {role} - {timestamp} - {user}"):
                                    st.write(f"**Type:** Annotation")
                                    st.write(f"**Rôle:** {role}")
                                    st.write(f"**Version:** {version}")
                                    st.write(f"**Diagnostic:** {label}")
                                    st.write(f"**Confiance:** {confidence:.2f}")
                                    if notes:
                                        st.write(f"**Notes:** {notes}")
                                    
                                    # Afficher les informations supplémentaires
                                    additional_info = entry['data'].get('additional_info', {})
                                    if additional_info:
                                        st.write("**Informations complémentaires:**")
                                        if additional_info.get('symptoms'):
                                            st.write(f"- Symptômes: {additional_info.get('symptoms')}")
                                        if additional_info.get('comorbidities'):
                                            st.write(f"- Comorbidités: {additional_info.get('comorbidities')}")
                                        if additional_info.get('spo2'):
                                            st.write(f"- SpO₂: {additional_info.get('spo2')}%")
                                        if additional_info.get('temperature'):
                                            st.write(f"- Température: {additional_info.get('temperature')}°C")
                                        if additional_info.get('crp'):
                                            st.write(f"- CRP: {additional_info.get('crp')} mg/L")
                                        if additional_info.get('image_quality'):
                                            st.write(f"- Qualité image: {additional_info.get('image_quality')}")
                                        if additional_info.get('urgency'):
                                            st.write(f"- Urgence: {additional_info.get('urgency')}")
                                        if additional_info.get('validated_at'):
                                            st.write(f"- Validé le: {additional_info.get('validated_at')}")
                            
                            elif entry['type'] == 'audit':
                                action = entry.get('action', 'N/A')
                                details = entry.get('details', {})
                                
                                with st.expander(f"🔍 {action} - {timestamp} - {user}"):
                                    st.write(f"**Type:** Action système")
                                    st.write(f"**Action:** {action}")
                                    
                                    if details:
                                        st.write("**Détails:**")
                                        if 'old_label' in details and 'new_label' in details:
                                            st.write(f"- Changement: {details.get('old_label')} → {details.get('new_label')}")
                                        if 'version' in details:
                                            st.write(f"- Version: {details.get('version')}")
                                        if 'image_ids' in details:
                                            st.write(f"- Images concernées: {len(details.get('image_ids', []))}")
                                        if 'count' in details:
                                            st.write(f"- Nombre: {details.get('count')}")
                                        if 'action_type' in details:
                                            st.write(f"- Type d'action: {details.get('action_type')}")
                                        if 'new_status' in details:
                                            st.write(f"- Nouveau statut: {details.get('new_status')}")
                    else:
                        st.info("Aucun historique disponible pour ce patient")

