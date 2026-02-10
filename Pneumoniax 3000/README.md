# Application d'Analyse de Radiographies Thoraciques pour la Pneumonie

Application Streamlit pour l'analyse de radiographies thoraciques avec détection de pneumonie par intelligence artificielle.

## Fonctionnalités

### Rôle Préparateur

1. **Import DICOM et Images Simples**
   - Import en lot de fichiers DICOM avec extraction automatique des métadonnées (PatientID, date, sexe, etc.)
   - Import d'images simples (PNG, JPG, JPEG) avec saisie manuelle d'un ID patient par image
   - Réinitialisation automatique du sélecteur après chaque import
   - Liaison des images aux patients

2. **Analyse par Modèle**
   - Lancement du modèle TensorFlow/Keras de détection sur les images importées
   - Suivi du statut de chaque analyse (en attente, en cours, terminé, échec)
   - Liste des erreurs avec raisons
   - Prédiction automatique : sain/malade

3. **Visualisation et Filtrage**
   - Liste filtrable des prédictions (sain/malade)
   - Filtres par prédiction et annotation
   - Tri par date, ID patient ou prédiction
   - Affichage avec texte blanc

4. **Annotation et Validation**
   - Complétion des informations patient (symptômes, comorbidités, signes vitaux, biologie)
   - Classification manuelle (sain/malade)
   - Reclassification manuelle si nécessaire
   - Journalisation de tous les changements avec versioning
   - Historique complet des modifications

5. **Envoi au Médecin**
   - Validation obligatoire : tous les patients doivent être annotés avant l'envoi
   - Envoi de la liste validée au médecin
   - Vérification automatique que chaque patient a été annoté

### Rôle Médecin

1. **Liste Priorisée**
   - Liste des patients à revoir, triée par priorité (malades en premier)
   - Filtres par classification et priorité
   - Statistiques (malades, sains)
   - Inspection détaillée de chaque patient

2. **Inspection et Validation**
   - Visualisation des radiographies
   - Affichage de la prédiction du modèle et de la classification du préparateur
   - Validation ou correction de la classification
   - Ajout de commentaires cliniques
   - Enregistrement de la vérité terrain après traitement

3. **Démarrer le Traitement et Liste de Suivi**
   - Démarrage de traitement directement depuis l'interface
   - 4 types d'actions disponibles :
     - **Prescription** : médicament, posologie, durée
     - **Examens complémentaires** : types d'examens, urgence
     - **Hospitalisation** : service, motif, durée estimée
     - **Orientation** : destination, motif
   - Liste de suivi de tous les patients en traitement
   - Mise à jour du statut du traitement (en traitement, en attente d'examens, hospitalisé, terminé)
   - Statistiques des patients en traitement

4. **Résultats Finaux**
   - Enregistrement de la vérité terrain après traitement
   - Finalisation des lots
   - Export des données pour réentraînement du modèle
   - Résumé des cas et étiquettes modifiées

## Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

**Note :** L'application nécessite TensorFlow et Keras. Si vous rencontrez des problèmes de compatibilité, utilisez :
```bash
pip install tensorflow==2.15.0 keras==2.15.0
```

2. Lancer l'application :
```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à `http://localhost:8501`

## Structure des Données

Les données sont stockées dans le répertoire `data/` :
- `patients.json` : Informations sur les patients
- `images.json` : Métadonnées des images DICOM et images simples
- `predictions.json` : Prédictions du modèle (label: sain/malade)
- `annotations.json` : Annotations des préparateurs et médecins avec versioning
- `audit_log.json` : Journal de tous les changements
- `images/` : Images extraites des fichiers DICOM et images simples importées

## Intégration du Modèle

Le fichier `model_interface.py` contient l'interface pour le modèle TensorFlow/Keras. Le modèle `model.h5` est chargé automatiquement au démarrage.

### Structure attendue du modèle

Le modèle doit :
- Accepter des images de taille 256x256 pixels
- Retourner une probabilité entre 0 et 1 (0 = sain, 1 = malade)
- Être sauvegardé au format `.h5`

### Emplacement du modèle

Le système cherche automatiquement `model.h5` dans cet ordre :
1. `/App Pneumonie/model.h5` (dossier du projet)
2. `/App Pneumonie/../main_project 3/model.h5` (dossier parent)
3. `~/Downloads/main_project 3/model.h5` (Downloads)

## Workflow Complet

### 1. Préparateur - Import et Analyse

1. **Import des images**
   - Onglet "📥 Import DICOM" : Import de fichiers DICOM (métadonnées extraites automatiquement)
   - Onglet "📥 Import DICOM" : Import d'images simples (PNG/JPG) avec saisie d'un ID patient par image
   - Le sélecteur de fichiers se réinitialise automatiquement après chaque import

2. **Analyse par le modèle**
   - Onglet "🤖 Analyse Modèle" : Sélectionner les images à analyser
   - Lancer l'analyse : le modèle TensorFlow traite chaque image
   - Suivi du statut en temps réel

3. **Visualisation**j
   - Onglet "📊 Visualisation & Filtrage" : Consulter les prédictions
   - Filtrer par prédiction (sain/malade) et annotation
   - Trier les résultats

4. **Annotation**
   - Onglet "✅ Validation & Envoi" : Sélectionner un patient
   - Pour chaque image du patient :
     - Visualiser l'image et la prédiction du modèle
     - Classifier manuellement (sain/malade)
     - Ajouter des informations complémentaires (symptômes, signes vitaux, biologie, etc.)
     - Enregistrer l'annotation
   - **Important** : Tous les patients doivent être annotés avant l'envoi

5. **Envoi au médecin**
   - Vérifier que tous les patients sont annotés
   - Sélectionner les patients à envoyer
   - Cliquer sur "📤 Envoyer la liste au médecin"

### 2. Médecin - Validation et Traitement

1. **Revue des patients**
   - Onglet "📋 Liste des Patients à Revoir" : Consulter la liste priorisée
   - Sélectionner un patient pour revue détaillée
   - Visualiser l'image, la prédiction et la classification du préparateur

2. **Validation clinique**
   - Valider ou corriger le diagnostic
   - Ajouter des commentaires cliniques
   - Enregistrer la vérité terrain (après traitement)

3. **Démarrer le traitement**
   - Onglet "💊 Démarrer le Traitement & Suivi" : Sélectionner un patient validé
   - Choisir le type d'action :
     - **Prescription** : Prescrire des médicaments
     - **Examens** : Demander des examens complémentaires
     - **Hospitalisation** : Hospitaliser le patient
     - **Orientation** : Orienter vers un autre service
   - Remplir les détails et démarrer le traitement
   - Le statut du patient est mis à jour automatiquement

4. **Suivi des traitements**
   - Consulter la liste de tous les patients en traitement
   - Voir les statistiques (en traitement, en attente d'examens, hospitalisés)
   - Mettre à jour le statut du traitement au fur et à mesure

5. **Finalisation**
   - Onglet "📊 Résultats & Export" : Consulter les patients validés
   - Marquer les lots comme finalisés
   - Exporter les données pour réentraînement du modèle

## Utilisation

1. **Connexion** : Sélectionner votre rôle (Préparateur ou Médecin) et entrer votre nom

2. **Préparateur** :
   - Importer les fichiers DICOM ou images simples
   - Lancer l'analyse sur les images
   - Visualiser et filtrer les résultats
   - Annoter chaque patient avec les informations complémentaires
   - Valider et envoyer au médecin

3. **Médecin** :
   - Consulter la liste priorisée
   - Inspecter chaque image et la prédiction
   - Valider ou corriger le diagnostic
   - Démarrer le traitement directement depuis l'interface
   - Suivre l'évolution des patients en traitement
   - Enregistrer les résultats finaux après traitement
   - Finaliser et exporter pour l'entraînement

## Notes Importantes

- **Journalisation complète** : Tous les changements sont journalisés avec horodatage et utilisateur
- **Versioning** : Le système de versioning permet de suivre l'historique complet des annotations
- **Validation obligatoire** : Les patients doivent obligatoirement être annotés par le préparateur avant l'envoi au médecin
- **Stockage local** : Les données sont stockées localement en JSON (prototype)
- **Modèle TensorFlow** : Le modèle est chargé automatiquement au démarrage (peut prendre 10-30 secondes)
- **Statuts de traitement** : en_traitement, en_attente_examens, hospitalise, termine

## Dépannage

### Le modèle ne charge pas
- Vérifiez que `model.h5` est présent dans le dossier du projet
- Vérifiez les versions de TensorFlow/Keras : `pip show tensorflow keras`
- Le premier chargement peut prendre 30-60 secondes

### Erreur "Module not found"
```bash
pip install -r requirements.txt
```

### L'application est bloquée au chargement
- Le modèle TensorFlow peut prendre du temps à charger
- Attendez 30-60 secondes lors du premier lancement
- Vérifiez les messages dans le terminal

### Problèmes de compatibilité TensorFlow
- Utilisez TensorFlow 2.15.0 et Keras 2.15.0 pour une meilleure compatibilité
- Vérifiez que Python 3.9 est utilisé
