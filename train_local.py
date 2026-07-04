import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

# Configuración base
# Descomprime tu .zip de Label Studio directamente dentro de esta carpeta
DATASET_ROOT = Path("custom_data") 
OUTPUT_DIR = Path("data")
TRAIN_RATIO = 0.9
CLASSES_FILE = DATASET_ROOT / "classes.txt"

def split_dataset():
    """Divide dataset de Label Studio en train y val."""
    print("Iniciando split de dataset...")
    
    img_dir = DATASET_ROOT / "images"
    lbl_dir = DATASET_ROOT / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Estructura de Label Studio no encontrada. Asegúrate de tener 'images/' y 'labels/' dentro de {DATASET_ROOT}")

    # Estructuras de salida
    train_img_dir = OUTPUT_DIR / "train" / "images"
    train_lbl_dir = OUTPUT_DIR / "train" / "labels"
    val_img_dir = OUTPUT_DIR / "val" / "images"
    val_lbl_dir = OUTPUT_DIR / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Buscar imágenes
    img_extensions = ['.jpg', '.jpeg', '.png']
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]
    
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)
    
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    def move_files(img_list, img_dest_dir, lbl_dest_dir):
        for img_path in img_list:
            # Mover imagen
            shutil.copy(img_path, img_dest_dir / img_path.name)
            # Buscar y mover etiqueta correspondiente
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_dest_dir / lbl_path.name)
            else:
                print(f"Advertencia: No se encontró etiqueta para {img_path.name}")

    move_files(train_imgs, train_img_dir, train_lbl_dir)
    move_files(val_imgs, val_img_dir, val_lbl_dir)
    print(f"Split completado. Train: {len(train_imgs)}, Val: {len(val_imgs)}")

def create_data_yaml():
    """Genera data.yaml basado en classes.txt."""
    print("Generando data.yaml...")
    
    if not CLASSES_FILE.exists():
        raise FileNotFoundError(f"No se encontró {CLASSES_FILE}")
        
    with open(CLASSES_FILE, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
        
    data = {
        'path': str(OUTPUT_DIR.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
        
    print(f"data.yaml creado en {yaml_path}")
    return str(yaml_path)

def train_model(yaml_path):
    """Entrena modelo YOLOv8n."""
    print("Iniciando entrenamiento...")
    
    model = YOLO('yolov8n.pt')
    
    model.train(
        data=yaml_path,
        epochs=60,
        imgsz=480,
        project='runs/detect',
        name='train_local',
        exist_ok=True,
        batch=16,
        workers=4
    )
    print("Entrenamiento finalizado. Pesos en runs/detect/train_local/weights/best.pt")

if __name__ == "__main__":
    # 1. Dividir datos
    split_dataset()
    
    # 2. Crear YAML
    yaml_path = create_data_yaml()
    
    # 3. Entrenar
    train_model(yaml_path)