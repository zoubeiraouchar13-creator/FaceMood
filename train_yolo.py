from pathlib import Path
import shutil
import kagglehub
from ultralytics import YOLO

def prepare_and_train():
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATASET_PATH = PROJECT_ROOT / "fer2013"

    # 1. Télécharger le dataset FER-2013 s'il n'est pas présent
    if not (DATASET_PATH / "train").exists() or not any((DATASET_PATH / "train").iterdir()):
        print("📥 Téléchargement automatique de FER-2013 via Kaggle...")
        path = kagglehub.dataset_download("msambare/fer2013")
        
        print(f"📦 Extraction vers {DATASET_PATH}...")
        downloaded_path = Path(path)
        
        # Copier/déplacer les fichiers dans votre dossier de projet
        if DATASET_PATH.exists():
            shutil.rmtree(DATASET_PATH)
        shutil.copytree(downloaded_path, DATASET_PATH)
        print("✅ Dataset extrait et structuré avec succès !")

    # 2. Lancer l'entraînement YOLOv8
    print("🚀 Démarrage de l'entraînement sur CPU...")
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data=str(DATASET_PATH),
        epochs=15,
        imgsz=48,
        batch=16,
        workers=2,
        device="cpu",
        project=str(PROJECT_ROOT / "runs" / "classify"),
        name="train",
        exist_ok=True
    )

    print("\n✅ Entraînement terminé !")

if __name__ == "__main__":
    prepare_and_train()