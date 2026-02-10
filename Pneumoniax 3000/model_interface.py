import numpy as np
from typing import Dict, Optional, Tuple
from PIL import Image
import os
import tensorflow as tf
from keras.utils import load_img, img_to_array

class ModelInterface:
    """
    Interface pour le modèle de détection de pneumonie
    
    Utilise un modèle TensorFlow/Keras pour faire les prédictions.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise l'interface du modèle
        
        Args:
            model_path: Chemin vers le modèle (par défaut: 'model.h5' dans le dossier du projet)
        """
        # Chemin par défaut vers le modèle
        if model_path is None:
            # Chercher model.h5 dans le dossier du projet
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'model.h5')
            
            # Si pas trouvé, chercher dans le dossier parent ou Downloads
            if not os.path.exists(model_path):
                parent_model = os.path.join(os.path.dirname(current_dir), 'main_project 3', 'model.h5')
                downloads_model = os.path.join(os.path.expanduser('~'), 'Downloads', 'main_project 3', 'model.h5')
                
                if os.path.exists(parent_model):
                    model_path = parent_model
                elif os.path.exists(downloads_model):
                    model_path = downloads_model
        
        self.model_path = model_path
        self.model = None
        
        # Charger le modèle si le chemin existe
        if model_path and os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"✅ Modèle chargé depuis: {model_path}")
            except Exception as e:
                print(f"❌ Erreur lors du chargement du modèle: {e}")
                self.model = None
        else:
            print(f"⚠️  Modèle non trouvé à: {model_path}")
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Prépare l'image pour la prédiction (même preprocessing que dans deployment.py)
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Array numpy prêt pour la prédiction
        """
        # Charger l'image en 256x256 (comme dans deployment.py)
        img = load_img(image_path, target_size=(256, 256))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Ajouter dimension batch
        
        return img
    
    def predict(self, image_path: str) -> Dict:
        """
        Prédit la présence de pneumonie sur une image
        
        Args:
            image_path: Chemin vers l'image à analyser
            
        Returns:
            Dictionnaire avec:
                - label: 'sain' ou 'malade'
                - confidence: score de confiance entre 0 et 1
        """
        if self.model is None:
            return {
                'label': 'error',
                'confidence': 0.0,
                'error': 'Modèle non chargé'
            }
        
        if not os.path.exists(image_path):
            return {
                'label': 'error',
                'confidence': 0.0,
                'error': f"Image non trouvée: {image_path}"
            }
        
        try:
            # Preprocessing
            img = self._preprocess_image(image_path)
            
            # Prédiction (comme dans deployment.py)
            pred = self.model.predict(img, verbose=0)[0][0]  # verbose=0 pour éviter les logs
            
            # Le modèle retourne une probabilité (0-1)
            # Dans le code original, une valeur élevée = malade
            # pred est la probabilité d'être malade
            
            # Convertir en label et confidence
            if pred >= 0.5:
                label = 'malade'
                confidence = float(pred)
            else:
                label = 'sain'
                confidence = float(1 - pred)  # Confidence d'être sain
            
            return {
                'label': label,
                'confidence': round(confidence, 3),
                'raw_prediction': round(float(pred), 3)  # Probabilité brute d'être malade
            }
            
        except Exception as e:
            return {
                'label': 'error',
                'confidence': 0.0,
                'error': f"Erreur lors de la prédiction: {str(e)}"
            }
    
    def predict_batch(self, image_paths: list) -> Dict[str, Dict]:
        """
        Prédit sur un lot d'images
        
        Args:
            image_paths: Liste des chemins vers les images
            
        Returns:
            Dictionnaire avec image_path comme clé et le résultat de prédiction comme valeur
        """
        results = {}
        for image_path in image_paths:
            if os.path.exists(image_path):
                results[image_path] = self.predict(image_path)
            else:
                results[image_path] = {
                    'label': 'error',
                    'confidence': 0.0,
                    'error': f"Image non trouvée: {image_path}"
                }
        return results
    
    def load_model(self, model_path: str):
        """
        Charge le modèle depuis un fichier
        
        Args:
            model_path: Chemin vers le fichier du modèle
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            print(f"✅ Modèle chargé depuis: {model_path}")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            self.model = None
