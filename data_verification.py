import os

dossier_principal = "C:\\Users\\hp\\Projets Python\\face_expressions\\fer2013"

for racine, sous_dossiers, fichiers in os.walk(dossier_principal):
    nombre_fichiers = len(fichiers)
    print(f"Folder : {racine} -> {nombre_fichiers} file(s)")
