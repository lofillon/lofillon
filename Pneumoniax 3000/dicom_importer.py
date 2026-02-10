import pydicom
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from PIL import Image
import io

class DICOMImporter:
    """Gestionnaire d'import de fichiers DICOM"""
    
    def __init__(self):
        self.required_tags = ['PatientID']
        self.optional_tags = [
            'StudyDate', 'StudyTime', 'PatientSex', 'PatientAge',
            'InstitutionName', 'StationName', 'StudyDescription',
            'Modality', 'BodyPartExamined', 'PatientPosition',
            'ViewPosition'
        ]
    
    def read_dicom_file(self, file_path: str) -> Tuple[Optional[pydicom.Dataset], Optional[str]]:
        """
        Lit un fichier DICOM et retourne le dataset et une erreur éventuelle
        """
        try:
            ds = pydicom.dcmread(file_path)
            return ds, None
        except Exception as e:
            return None, str(e)
    
    def extract_metadata(self, ds: pydicom.Dataset) -> Dict:
        """
        Extrait les métadonnées d'un dataset DICOM
        """
        metadata = {}
        
        # Tags obligatoires
        metadata['patient_id'] = self._get_tag_value(ds, 'PatientID', 'UNKNOWN')
        
        # Vérifier que le PatientID est présent
        if metadata['patient_id'] == 'UNKNOWN':
            raise ValueError("PatientID manquant dans le fichier DICOM")
        
        # Tags optionnels
        metadata['exam_date'] = self._get_tag_value(ds, 'StudyDate', '')
        metadata['exam_time'] = self._get_tag_value(ds, 'StudyTime', '')
        metadata['sex'] = self._get_tag_value(ds, 'PatientSex', '')
        metadata['age'] = self._get_tag_value(ds, 'PatientAge', '')
        metadata['institution_name'] = self._get_tag_value(ds, 'InstitutionName', '')
        metadata['station_name'] = self._get_tag_value(ds, 'StationName', '')
        metadata['study_description'] = self._get_tag_value(ds, 'StudyDescription', '')
        metadata['modality'] = self._get_tag_value(ds, 'Modality', '')
        metadata['body_part'] = self._get_tag_value(ds, 'BodyPartExamined', '')
        metadata['patient_position'] = self._get_tag_value(ds, 'PatientPosition', '')
        metadata['view_position'] = self._get_tag_value(ds, 'ViewPosition', '')
        
        # Formater la date si disponible
        if metadata['exam_date']:
            try:
                # Format DICOM: YYYYMMDD
                date_str = str(metadata['exam_date'])
                if len(date_str) == 8:
                    metadata['exam_date_formatted'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                else:
                    metadata['exam_date_formatted'] = metadata['exam_date']
            except:
                metadata['exam_date_formatted'] = metadata['exam_date']
        else:
            metadata['exam_date_formatted'] = ''
        
        return metadata
    
    def extract_image_array(self, ds: pydicom.Dataset) -> Optional[np.ndarray]:
        """
        Extrait l'array numpy de l'image DICOM
        """
        try:
            pixel_array = ds.pixel_array
            
            # Normaliser si nécessaire
            if pixel_array.dtype != np.uint8:
                # Normaliser entre 0 et 255
                pixel_array = pixel_array.astype(np.float64)
                pixel_array = pixel_array - pixel_array.min()
                if pixel_array.max() > 0:
                    pixel_array = (pixel_array / pixel_array.max()) * 255
                pixel_array = pixel_array.astype(np.uint8)
            
            return pixel_array
        except Exception as e:
            print(f"Erreur lors de l'extraction de l'image: {e}")
            return None
    
    def convert_to_pil_image(self, pixel_array: np.ndarray) -> Optional[Image.Image]:
        """
        Convertit un array numpy en image PIL
        """
        try:
            if len(pixel_array.shape) == 2:
                # Image en niveaux de gris
                return Image.fromarray(pixel_array, mode='L')
            elif len(pixel_array.shape) == 3:
                # Image couleur
                return Image.fromarray(pixel_array)
            else:
                return None
        except Exception as e:
            print(f"Erreur lors de la conversion PIL: {e}")
            return None
    
    def save_image_preview(self, pixel_array: np.ndarray, output_path: str) -> bool:
        """
        Sauvegarde une prévisualisation de l'image
        """
        try:
            pil_image = self.convert_to_pil_image(pixel_array)
            if pil_image:
                # Redimensionner si trop grande (max 1024px)
                max_size = 1024
                if max(pil_image.size) > max_size:
                    pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                pil_image.save(output_path, format='PNG')
                return True
            return False
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return False
    
    def _get_tag_value(self, ds: pydicom.Dataset, tag: str, default: str = '') -> str:
        """
        Récupère la valeur d'un tag DICOM de manière sécurisée
        """
        try:
            value = getattr(ds, tag, None)
            if value is None:
                return default
            return str(value)
        except:
            return default
    
    def import_batch(self, file_paths: List[str], images_dir: str = "data/images") -> List[Dict]:
        """
        Importe un lot de fichiers DICOM
        
        Retourne une liste de dictionnaires avec les métadonnées et chemins des images
        """
        os.makedirs(images_dir, exist_ok=True)
        
        results = []
        
        for file_path in file_paths:
            result = {
                'file_path': file_path,
                'success': False,
                'error': None,
                'metadata': None,
                'image_path': None
            }
            
            # Lire le fichier DICOM
            ds, error = self.read_dicom_file(file_path)
            if error:
                result['error'] = error
                results.append(result)
                continue
            
            # Extraire les métadonnées
            try:
                metadata = self.extract_metadata(ds)
                result['metadata'] = metadata
            except Exception as e:
                result['error'] = f"Erreur métadonnées: {str(e)}"
                results.append(result)
                continue
            
            # Extraire l'image
            pixel_array = self.extract_image_array(ds)
            if pixel_array is None:
                result['error'] = "Impossible d'extraire l'image"
                results.append(result)
                continue
            
            # Sauvegarder l'image
            file_name = os.path.basename(file_path)
            image_name = f"{metadata['patient_id']}_{os.path.splitext(file_name)[0]}.png"
            image_path = os.path.join(images_dir, image_name)
            
            if self.save_image_preview(pixel_array, image_path):
                result['image_path'] = image_path
                result['success'] = True
            else:
                result['error'] = "Impossible de sauvegarder l'image"
            
            results.append(result)
        
        return results

