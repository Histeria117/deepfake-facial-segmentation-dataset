import os
import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import onnxruntime as ort
import site
import sysconfig
# Importar torch primero evita conflictos de DLL con cuDNN en Windows.
# PyTorch carga sus propias DLLs CUDA/cuDNN.
try:
    import torch
    print("Torch importado correctamente.")
    print("Torch:", torch.__version__)
    print("Torch CUDA disponible:", torch.cuda.is_available())
except Exception as e:
    print("Error importando torch:", e)
    raise

import onnxruntime as ort

try:
    # Sin directory="", para que ONNX Runtime busque primero DLLs compatibles,
    # incluyendo las cargadas por PyTorch.
    ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
    print("ONNX Runtime DLLs precargadas.")
except Exception as e:
    print("No se pudieron precargar DLLs con ONNX Runtime:", e)

print("ONNX providers disponibles:", ort.get_available_providers())

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

BASE_DIR = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")
IMAGE_DIR = BASE_DIR / "data_raw" / "face_dataset" / "images"
MASK_BASE_DIR = BASE_DIR / "face_parsing_output" / "binary_masks"
OUTPUT_DIR = BASE_DIR / "dataset_rostros_avanzado" / "faceswap"
PREVIEW_DIR = BASE_DIR / "preview_faceswap_avanzado"
MODEL_PATH = BASE_DIR / "models" / "insightface" / "inswapper_128.onnx"

IMAGE_SIZE = 512
MAX_SAMPLES = 100
USE_GPU = True

random.seed(45)

def ensure_dirs():
    for folder in [
        OUTPUT_DIR / "imagen_original",
        OUTPUT_DIR / "mascara_autentica",
        OUTPUT_DIR / "mascara_fake",
        PREVIEW_DIR
    ]:
        folder.mkdir(parents=True, exist_ok=True)
def get_image_files():
    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    files = []
    for p in IMAGE_DIR.rglob("*"):
        if p.suffix.lower() in valid_ext:
            files.append(p)
    return sorted(files)
def read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img

def read_mask(mask_path):
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def get_masks_for_image(stem):
    full_face = read_mask(MASK_BASE_DIR / "full_face" / f"{stem}.png")

    if full_face is None:
        return None

    if np.sum(full_face) == 0:
        return None

    return {
        "full_face": full_face
    }


def valid_item(image_path):
    stem = image_path.stem
    masks = get_masks_for_image(stem)
    if masks is None:
        return None

    if np.sum(masks["full_face"]) == 0:
        return None

    return {
        "image_path": image_path,
        "stem": stem,
        "masks": masks
    }


def largest_face(faces):
    if not faces:
        return None

    def face_area(face):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    return max(faces, key=face_area)


def init_models():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No encontré el modelo en:\n{MODEL_PATH}\n"
            "Coloca ahí el archivo inswapper_128.onnx"
        )

    providers = ["CPUExecutionProvider"]
    ctx_id = -1

    if USE_GPU:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0

    import onnxruntime as ort

    print("USE_GPU:", USE_GPU)
    print("ONNX available providers:", ort.get_available_providers())
    print("Providers solicitados:", providers)
    print("ctx_id:", ctx_id)
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    swapper = get_model(str(MODEL_PATH), providers=providers)

    return app, swapper


def clean_binary_mask(mask):
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)

    mask_filled = cv2.bitwise_or(mask, flood_inv)

    _, mask_filled = cv2.threshold(mask_filled, 127, 255, cv2.THRESH_BINARY)

    return mask_filled


def make_fake_and_auth_masks(masks):
    fake_mask = masks["full_face"].copy()
    fake_mask = clean_binary_mask(fake_mask)
    authentic_mask = np.zeros_like(fake_mask)
    return fake_mask, authentic_mask


def save_sample(sample_id, manipulated, authentic_mask, fake_mask):
    img_path = OUTPUT_DIR / "imagen_original" / f"{sample_id}.png"
    auth_path = OUTPUT_DIR / "mascara_autentica" / f"{sample_id}.png"
    fake_path = OUTPUT_DIR / "mascara_fake" / f"{sample_id}.png"

    cv2.imwrite(str(img_path), manipulated)
    cv2.imwrite(str(auth_path), authentic_mask)
    cv2.imwrite(str(fake_path), fake_mask)

    return img_path, auth_path, fake_path


def save_preview(sample_id, manipulated, authentic_mask, fake_mask):
    auth_bgr = cv2.cvtColor(authentic_mask, cv2.COLOR_GRAY2BGR)
    fake_bgr = cv2.cvtColor(fake_mask, cv2.COLOR_GRAY2BGR)

    grid = np.hstack([manipulated, auth_bgr, fake_bgr])
    out_path = PREVIEW_DIR / f"{sample_id}.png"
    cv2.imwrite(str(out_path), grid)


def main():
    ensure_dirs()

    app, swapper = init_models()

    image_files = get_image_files()
    print("Imágenes encontradas:", len(image_files))

    items = []
    for path in image_files:
        item = valid_item(path)
        if item is not None:
            items.append(item)

    print("Imágenes válidas con máscaras:", len(items))

    if len(items) < 2:
        print("Necesitas al menos 2 imágenes válidas.")
        return

    generated = 0
    records = []

    random.shuffle(items)

    for target_item in tqdm(items):
        if generated >= MAX_SAMPLES:
            break

        target_img = read_image(target_item["image_path"])
        if target_img is None:
            continue

        source_candidates = [x for x in items if x["stem"] != target_item["stem"]]
        if not source_candidates:
            continue

        source_item = random.choice(source_candidates)
        source_img = read_image(source_item["image_path"])
        if source_img is None:
            continue

        # Detectar cara principal en target y source
        target_faces = app.get(target_img)
        source_faces = app.get(source_img)

        target_face = largest_face(target_faces)
        source_face = largest_face(source_faces)

        if target_face is None or source_face is None:
            continue

        try:
            swapped = swapper.get(
                target_img,
                target_face,
                source_face,
                paste_back=True
            )
        except Exception as e:
            print(f"Error en swap {target_item['stem']} <- {source_item['stem']}: {e}")
            continue

        fake_mask, authentic_mask = make_fake_and_auth_masks(target_item["masks"])

        if np.sum(fake_mask) == 0:
            continue

        sample_id = f"faceswap_adv_{generated:05d}"

        img_path, auth_path, fake_path = save_sample(
            sample_id,
            swapped,
            authentic_mask,
            fake_mask
        )

        save_preview(sample_id, swapped, authentic_mask, fake_mask)

        records.append({
            "id": sample_id,
            "tipo": "faceswap_avanzado",
            "target_image": str(target_item["image_path"]),
            "source_image": str(source_item["image_path"]),
            "imagen_original": str(img_path),
            "mascara_autentica": str(auth_path),
            "mascara_fake": str(fake_path)
        })

        generated += 1

    metadata_path = OUTPUT_DIR / "metadata.csv"
    pd.DataFrame(records).to_csv(metadata_path, index=False)

    print("Muestras generadas:", generated)
    print("Metadata:", metadata_path)
    print("Preview:", PREVIEW_DIR)
    print("Salida:", OUTPUT_DIR)


if __name__ == "__main__":
    main()