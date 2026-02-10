import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

class DataManager:
    """Gestionnaire centralisé des données de l'application"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.patients_file = os.path.join(data_dir, "patients.json")
        self.images_file = os.path.join(data_dir, "images.json")
        self.predictions_file = os.path.join(data_dir, "predictions.json")
        self.annotations_file = os.path.join(data_dir, "annotations.json")
        self.audit_log_file = os.path.join(data_dir, "audit_log.json")
        
        # Créer le répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialiser les fichiers JSON s'ils n'existent pas
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialise les fichiers JSON s'ils n'existent pas"""
        files = {
            self.patients_file: [],
            self.images_file: [],
            self.predictions_file: [],
            self.annotations_file: [],
            self.audit_log_file: []
        }
        
        for file_path, default_value in files.items():
            if not os.path.exists(file_path):
                self._save_json(file_path, default_value)
    
    def _load_json(self, file_path: str) -> List[Dict]:
        """Charge un fichier JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_json(self, file_path: str, data: List[Dict]):
        """Sauvegarde un fichier JSON"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    # ========== Gestion des patients ==========
    
    def add_patient(self, patient_id: str, metadata: Dict) -> str:
        """Ajoute un nouveau patient"""
        patients = self._load_json(self.patients_file)
        
        # Vérifier si le patient existe déjà
        existing = next((p for p in patients if p['patient_id'] == patient_id), None)
        if existing:
            return existing['id']
        
        patient = {
            'id': f"pat_{len(patients) + 1}",
            'patient_id': patient_id,
            'metadata': metadata,
            'created_at': datetime.now().isoformat()
        }
        
        patients.append(patient)
        self._save_json(self.patients_file, patients)
        return patient['id']
    
    def get_patient_by_id(self, patient_id: str) -> Optional[Dict]:
        """Récupère un patient par son ID"""
        patients = self._load_json(self.patients_file)
        return next((p for p in patients if p['patient_id'] == patient_id), None)
    
    def get_all_patients(self) -> List[Dict]:
        """Récupère tous les patients"""
        return self._load_json(self.patients_file)
    
    # ========== Gestion des images ==========
    
    def add_image(self, image_data: Dict) -> str:
        """Ajoute une nouvelle image"""
        images = self._load_json(self.images_file)
        
        image = {
            'id': f"img_{len(images) + 1}",
            **image_data,
            'created_at': datetime.now().isoformat(),
            'status': 'pending'  # pending, processing, completed, failed
        }
        
        images.append(image)
        self._save_json(self.images_file, images)
        return image['id']
    
    def update_image_status(self, image_id: str, status: str, error: Optional[str] = None):
        """Met à jour le statut d'une image"""
        images = self._load_json(self.images_file)
        for img in images:
            if img['id'] == image_id:
                img['status'] = status
                if error:
                    img['error'] = error
                img['updated_at'] = datetime.now().isoformat()
                break
        self._save_json(self.images_file, images)
    
    def get_image(self, image_id: str) -> Optional[Dict]:
        """Récupère une image par son ID"""
        images = self._load_json(self.images_file)
        return next((img for img in images if img['id'] == image_id), None)
    
    def get_images_by_patient(self, patient_id: str) -> List[Dict]:
        """Récupère toutes les images d'un patient"""
        images = self._load_json(self.images_file)
        return [img for img in images if img.get('patient_id') == patient_id]
    
    def get_all_images(self) -> List[Dict]:
        """Récupère toutes les images"""
        return self._load_json(self.images_file)
    
    # ========== Gestion des prédictions ==========
    
    def add_prediction(self, prediction_data: Dict) -> str:
        """Ajoute une prédiction du modèle"""
        predictions = self._load_json(self.predictions_file)
        
        prediction = {
            'id': f"pred_{len(predictions) + 1}",
            **prediction_data,
            'created_at': datetime.now().isoformat()
        }
        
        predictions.append(prediction)
        self._save_json(self.predictions_file, predictions)
        return prediction['id']
    
    def get_prediction_by_image(self, image_id: str) -> Optional[Dict]:
        """Récupère la prédiction pour une image"""
        predictions = self._load_json(self.predictions_file)
        return next((p for p in predictions if p.get('image_id') == image_id), None)
    
    def get_all_predictions(self) -> List[Dict]:
        """Récupère toutes les prédictions"""
        return self._load_json(self.predictions_file)
    
    # ========== Gestion des annotations ==========
    
    def add_annotation(self, annotation_data: Dict) -> str:
        """Ajoute une annotation (préparateur ou médecin)"""
        annotations = self._load_json(self.annotations_file)
        
        annotation = {
            'id': f"ann_{len(annotations) + 1}",
            **annotation_data,
            'created_at': datetime.now().isoformat(),
            'version': 1
        }
        
        annotations.append(annotation)
        self._save_json(self.annotations_file, annotations)
        
        # Journaliser le changement
        self._log_change(annotation_data.get('user_name'), 'annotation_created', annotation)
        
        return annotation['id']
    
    def update_annotation(self, image_id: str, user_name: str, updates: Dict) -> Optional[str]:
        """Met à jour une annotation existante"""
        annotations = self._load_json(self.annotations_file)
        
        # Trouver l'annotation la plus récente pour cette image
        image_annotations = [a for a in annotations if a.get('image_id') == image_id]
        if not image_annotations:
            return None
        
        # Trier par version et prendre la plus récente
        latest = max(image_annotations, key=lambda x: x.get('version', 0))
        
        # Créer une nouvelle version
        old_label = latest.get('label')
        new_annotation = {
            'id': f"ann_{len(annotations) + 1}",
            'image_id': image_id,
            'patient_id': latest.get('patient_id'),
            'label': updates.get('label', latest.get('label')),
            'confidence': updates.get('confidence', latest.get('confidence')),
            'notes': updates.get('notes', latest.get('notes', '')),
            'additional_info': updates.get('additional_info', latest.get('additional_info', {})),
            'user_name': user_name,
            'user_role': updates.get('user_role', latest.get('user_role')),
            'created_at': datetime.now().isoformat(),
            'version': latest.get('version', 0) + 1,
            'previous_version_id': latest['id']
        }
        
        annotations.append(new_annotation)
        self._save_json(self.annotations_file, annotations)
        
        # Journaliser le changement
        self._log_change(user_name, 'annotation_updated', {
            'image_id': image_id,
            'old_label': old_label,
            'new_label': new_annotation['label'],
            'version': new_annotation['version']
        })
        
        return new_annotation['id']
    
    def get_annotation_by_image(self, image_id: str) -> Optional[Dict]:
        """Récupère l'annotation la plus récente pour une image"""
        annotations = self._load_json(self.annotations_file)
        image_annotations = [a for a in annotations if a.get('image_id') == image_id]
        if not image_annotations:
            return None
        return max(image_annotations, key=lambda x: x.get('version', 0))
    
    def get_all_annotations(self) -> List[Dict]:
        """Récupère toutes les annotations"""
        return self._load_json(self.annotations_file)
    
    def is_patient_annotated(self, patient_id: str) -> bool:
        """Vérifie si un patient a été annoté par le préparateur"""
        images = self.get_images_by_patient(patient_id)
        for img in images:
            annotation = self.get_annotation_by_image(img['id'])
            if annotation and annotation.get('user_role') == 'Préparateur':
                return True
        return False
    
    def are_all_patients_annotated(self, patient_ids: List[str]) -> bool:
        """Vérifie si tous les patients d'une liste ont été annotés"""
        return all(self.is_patient_annotated(pid) for pid in patient_ids)
    
    # ========== Gestion des lots ==========
    
    def mark_batch_for_review(self, image_ids: List[str], user_name: str):
        """Marque un lot d'images comme prêt pour revue médicale"""
        images = self._load_json(self.images_file)
        for img in images:
            if img['id'] in image_ids:
                img['status'] = 'ready_for_review'
                img['sent_for_review_at'] = datetime.now().isoformat()
                img['sent_by'] = user_name
        self._save_json(self.images_file, images)
        
        self._log_change(user_name, 'batch_sent_for_review', {
            'image_ids': image_ids,
            'count': len(image_ids)
        })
    
    def get_images_for_review(self) -> List[Dict]:
        """Récupère les images en attente de revue médicale"""
        images = self._load_json(self.images_file)
        return [img for img in images if img.get('status') == 'ready_for_review']
    
    def mark_batch_finalized(self, image_ids: List[str], user_name: str):
        """Marque un lot comme finalisé par le médecin"""
        images = self._load_json(self.images_file)
        for img in images:
            if img['id'] in image_ids:
                img['status'] = 'finalized'
                img['finalized_at'] = datetime.now().isoformat()
                img['finalized_by'] = user_name
        self._save_json(self.images_file, images)
        
        self._log_change(user_name, 'batch_finalized', {
            'image_ids': image_ids,
            'count': len(image_ids)
        })
    
    def start_treatment(self, image_id: str, user_name: str, action_type: str, details: Dict) -> str:
        """Démarre un traitement pour un patient"""
        # Créer ou mettre à jour l'annotation avec les informations de traitement
        annotation = self.get_annotation_by_image(image_id)
        image = self.get_image(image_id)
        
        if not image:
            return None
        
        treatment_data = {
            'action_type': action_type,  # 'prescription', 'examens', 'hospitalisation', 'orientation'
            'details': details,
            'started_at': datetime.now().isoformat(),
            'started_by': user_name,
            'status': 'en_traitement'
        }
        
        # Mettre à jour ou créer l'annotation avec les infos de traitement
        if annotation:
            updates = {
                'label': annotation.get('label'),
                'confidence': annotation.get('confidence', 0.5),
                'notes': annotation.get('notes', ''),
                'additional_info': {
                    **annotation.get('additional_info', {}),
                    'treatment': treatment_data
                },
                'user_role': annotation.get('user_role', 'Médecin')
            }
            self.update_annotation(image_id, user_name, updates)
        else:
            # Créer une nouvelle annotation pour le traitement
            self.add_annotation({
                'image_id': image_id,
                'patient_id': image.get('patient_id'),
                'label': 'malade',  # Par défaut si traitement démarré
                'confidence': 0.5,
                'notes': f"Traitement démarré: {action_type}",
                'additional_info': {
                    'treatment': treatment_data
                },
                'user_name': user_name,
                'user_role': 'Médecin'
            })
        
        # Mettre à jour le statut de l'image
        self.update_image_status(image_id, 'en_traitement')
        
        # Journaliser
        self._log_change(user_name, 'treatment_started', {
            'image_id': image_id,
            'action_type': action_type,
            'details': details
        })
        
        return image_id
    
    def update_treatment_status(self, image_id: str, user_name: str, new_status: str, notes: str = ""):
        """Met à jour le statut d'un traitement"""
        annotation = self.get_annotation_by_image(image_id)
        if annotation and annotation.get('additional_info', {}).get('treatment'):
            treatment = annotation['additional_info']['treatment']
            treatment['status'] = new_status
            treatment['updated_at'] = datetime.now().isoformat()
            treatment['updated_by'] = user_name
            if notes:
                treatment['notes'] = notes
            
            updates = {
                'label': annotation.get('label'),
                'confidence': annotation.get('confidence', 0.5),
                'notes': annotation.get('notes', ''),
                'additional_info': annotation.get('additional_info', {}),
                'user_role': annotation.get('user_role', 'Médecin')
            }
            self.update_annotation(image_id, user_name, updates)
            
            # Mettre à jour le statut de l'image
            self.update_image_status(image_id, new_status)
            
            self._log_change(user_name, 'treatment_status_updated', {
                'image_id': image_id,
                'new_status': new_status
            })
    
    def get_patients_in_treatment(self) -> List[Dict]:
        """Récupère tous les patients en traitement"""
        images = self._load_json(self.images_file)
        annotations = self._load_json(self.annotations_file)
        
        # Créer un dictionnaire des annotations les plus récentes par image
        latest_annotations = {}
        for ann in annotations:
            img_id = ann.get('image_id')
            if img_id:
                if img_id not in latest_annotations:
                    latest_annotations[img_id] = ann
                elif ann.get('version', 0) > latest_annotations[img_id].get('version', 0):
                    latest_annotations[img_id] = ann
        
        # Filtrer les images avec traitement
        patients_in_treatment = []
        for img in images:
            ann = latest_annotations.get(img['id'])
            if ann and ann.get('additional_info', {}).get('treatment'):
                treatment = ann['additional_info']['treatment']
                if treatment.get('status') in ['en_traitement', 'en_attente_examens', 'hospitalise']:
                    patients_in_treatment.append({
                        'image': img,
                        'annotation': ann,
                        'treatment': treatment
                    })
        
        return patients_in_treatment
    
    def get_patients_with_completed_treatment(self) -> List[Dict]:
        """Récupère tous les patients avec traitement terminé (statut 'termine')"""
        images = self._load_json(self.images_file)
        annotations = self._load_json(self.annotations_file)
        
        # Créer un dictionnaire des annotations les plus récentes par image
        latest_annotations = {}
        for ann in annotations:
            img_id = ann.get('image_id')
            if img_id:
                if img_id not in latest_annotations:
                    latest_annotations[img_id] = ann
                elif ann.get('version', 0) > latest_annotations[img_id].get('version', 0):
                    latest_annotations[img_id] = ann
        
        # Filtrer les images avec traitement terminé
        completed_patients = []
        for img in images:
            ann = latest_annotations.get(img['id'])
            if ann and ann.get('additional_info', {}).get('treatment'):
                treatment = ann['additional_info']['treatment']
                if treatment.get('status') == 'termine':
                    completed_patients.append({
                        'image': img,
                        'annotation': ann,
                        'treatment': treatment
                    })
        
        return completed_patients
    
    # ========== Journalisation ==========
    
    def _log_change(self, user_name: str, action: str, details: Dict):
        """Journalise un changement dans le système"""
        log = self._load_json(self.audit_log_file)
        
        log_entry = {
            'id': f"log_{len(log) + 1}",
            'user_name': user_name,
            'action': action,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        log.append(log_entry)
        self._save_json(self.audit_log_file, log)
    
    def get_audit_log(self, image_id: Optional[str] = None) -> List[Dict]:
        """Récupère le journal d'audit"""
        log = self._load_json(self.audit_log_file)
        if image_id:
            return [entry for entry in log if image_id in str(entry.get('details', {}))]
        return log
    
    # ========== Utilitaires ==========
    
    def get_patient_summary(self, patient_id: str) -> Dict:
        """Récupère un résumé complet d'un patient"""
        patient = self.get_patient_by_id(patient_id)
        if not patient:
            return {}
        
        images = self.get_images_by_patient(patient_id)
        summary = {
            'patient': patient,
            'images': [],
            'predictions': [],
            'annotations': []
        }
        
        for img in images:
            summary['images'].append(img)
            pred = self.get_prediction_by_image(img['id'])
            if pred:
                summary['predictions'].append(pred)
            ann = self.get_annotation_by_image(img['id'])
            if ann:
                summary['annotations'].append(ann)
        
        return summary
    
    def get_dataframe_for_preparator(self) -> pd.DataFrame:
        """Crée un DataFrame pour l'affichage dans la vue préparateur"""
        images = self.get_all_images()
        predictions = {p.get('image_id'): p for p in self.get_all_predictions()}
        annotations = {a.get('image_id'): a for a in self.get_all_annotations()}
        patients = {p['patient_id']: p for p in self.get_all_patients()}
        
        rows = []
        for img in images:
            patient_id = img.get('patient_id')
            patient = patients.get(patient_id, {})
            pred = predictions.get(img['id'], {})
            ann = annotations.get(img['id'], {})
            
            rows.append({
                'ID Image': img['id'],
                'ID Patient': patient_id,
                'Date Examen': img.get('exam_date', 'N/A'),
                'Sexe': patient.get('metadata', {}).get('sex', 'N/A'),
                'Prédiction Modèle': pred.get('label', 'En attente'),
                'Annotation Préparateur': ann.get('label', 'Non annoté') if ann.get('user_role') == 'Préparateur' else 'Non annoté',
                'Statut': img.get('status', 'pending'),
                'Version': ann.get('version', 0)
            })
        
        return pd.DataFrame(rows)
    
    def get_dataframe_for_doctor(self) -> pd.DataFrame:
        """Crée un DataFrame pour l'affichage dans la vue médecin"""
        images = self.get_images_for_review()
        predictions = {p.get('image_id'): p for p in self.get_all_predictions()}
        annotations = {a.get('image_id'): a for a in self.get_all_annotations()}
        patients = {p['patient_id']: p for p in self.get_all_patients()}
        
        rows = []
        for img in images:
            patient_id = img.get('patient_id')
            patient = patients.get(patient_id, {})
            pred = predictions.get(img['id'], {})
            ann = annotations.get(img['id'], {})
            
            # Priorité : malade > sain
            priority = 0
            if ann.get('label') == 'malade':
                priority = 2
            elif ann.get('label') == 'sain':
                priority = 1
            
            rows.append({
                'ID Image': img['id'],
                'ID Patient': patient_id,
                'Date Examen': img.get('exam_date', 'N/A'),
                'Sexe': patient.get('metadata', {}).get('sex', 'N/A'),
                'Classification Préparateur': ann.get('label', 'N/A') if ann.get('user_role') == 'Préparateur' else 'N/A',
                'Prédiction Modèle': pred.get('label', 'N/A'),
                'Priorité': priority,
                'Statut': 'En attente de validation'
            })
        
        # Trier par priorité décroissante
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('Priorité', ascending=False)
        
        return df

