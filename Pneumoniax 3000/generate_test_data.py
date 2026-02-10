"""
Script pour générer des données de test DICOM simulées
Note: Ce script crée des fichiers DICOM basiques pour tester l'application.
Pour un usage réel, utilisez de vrais fichiers DICOM.
"""

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid
import numpy as np
from datetime import datetime
import os

def create_test_dicom(patient_id: str, output_path: str, is_sick: bool = False):
    """
    Crée un fichier DICOM de test
    
    Args:
        patient_id: ID du patient
        output_path: Chemin de sortie
        is_sick: Si True, simule une image de patient malade
    """
    # Créer un dataset DICOM basique
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1'  # Chest X-Ray
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'  # Implicit VR Little Endian
    
    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Métadonnées obligatoires
    ds.PatientID = patient_id
    ds.PatientName = f"Test^Patient^{patient_id}"
    ds.PatientSex = np.random.choice(['M', 'F'])
    ds.PatientAge = f"{np.random.randint(20, 80):03d}Y"
    
    # Date et heure de l'examen
    now = datetime.now()
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    
    # Informations de l'étude
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    
    # Type d'examen
    ds.Modality = "CR"  # Computed Radiography
    ds.StudyDescription = "CHEST"
    ds.BodyPartExamined = "CHEST"
    ds.PatientPosition = np.random.choice(["STANDING", "SUPINE"])
    ds.ViewPosition = np.random.choice(["PA", "AP"])
    
    # Institution
    ds.InstitutionName = "Hôpital Test"
    ds.StationName = "Station-01"
    
    # Créer une image simulée (512x512 pixels en niveaux de gris)
    # Pour un vrai fichier DICOM, vous devriez charger une vraie image
    rows = 512
    cols = 512
    
    # Générer une image simulée
    if is_sick:
        # Image avec des zones plus sombres (simulant une opacité)
        pixel_array = np.random.randint(100, 200, (rows, cols), dtype=np.uint16)
        # Ajouter une zone d'opacité
        center_x, center_y = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        mask = (x - center_x)**2 + (y - center_y)**2 < (rows // 4)**2
        pixel_array[mask] = np.random.randint(50, 150, mask.sum(), dtype=np.uint16)
    else:
        # Image normale
        pixel_array = np.random.randint(150, 250, (rows, cols), dtype=np.uint16)
    
    # Paramètres d'image
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [0.143, 0.143]  # mm
    ds.PixelRepresentation = 0
    
    # Ajouter les pixels
    ds.PixelData = pixel_array.tobytes()
    
    # Sauvegarder
    ds.save_as(output_path, write_like_original=False)
    print(f"✅ Fichier DICOM créé: {output_path}")

def main():
    """Génère plusieurs fichiers DICOM de test"""
    output_dir = "test_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Génération de fichiers DICOM de test...")
    print("=" * 50)
    
    # Générer 5 patients sains
    for i in range(1, 6):
        patient_id = f"PATIENT_S{i:03d}"
        output_path = os.path.join(output_dir, f"{patient_id}.dcm")
        create_test_dicom(patient_id, output_path, is_sick=False)
    
    # Générer 3 patients malades
    for i in range(1, 4):
        patient_id = f"PATIENT_M{i:03d}"
        output_path = os.path.join(output_dir, f"{patient_id}.dcm")
        create_test_dicom(patient_id, output_path, is_sick=True)
    
    print("=" * 50)
    print(f"✅ {8} fichiers DICOM de test créés dans '{output_dir}/'")
    print("\nVous pouvez maintenant importer ces fichiers dans l'application Streamlit!")

if __name__ == "__main__":
    main()

