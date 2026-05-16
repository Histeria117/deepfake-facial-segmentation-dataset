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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="insightface.*"
)
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

BATCH_NUM = 0
BATCH_SIZE = 1000
MAX_NEW_SAMPLES = 1000
IMAGE_SIZE = 512
USE_GPU = True
random.seed(45)

BATCH_NAME = f"batch_{BATCH_NUM:03d}"
METHOD_NAME = "faceswap"
METHOD_PREFIX = "fs"
BASE_DIR = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")
IMAGE_DIR = BASE_DIR / "data_raw" / "face_dataset" / "images_faceswap"
MASK_BASE_DIR = BASE_DIR / "face_parsing_output_faceswap" / "binary_masks"
OUTPUT_DIR = BASE_DIR / "dataset_batches" / BATCH_NAME / METHOD_NAME
PREVIEW_DIR = BASE_DIR / "dataset_batches" / BATCH_NAME / f"preview_{METHOD_NAME}"
MODEL_PATH = BASE_DIR / "models" / "insightface" / "inswapper_128.onnx"


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

def get_next_batch_index():
    img_dir = OUTPUT_DIR / "imagen_original"
    pattern = f"{METHOD_PREFIX}_b{BATCH_NUM:03d}_*.png"
    existing_files = list(img_dir.glob(pattern))

    if not existing_files:
        return 0

    indices = []

    for file in existing_files:
        try:
            idx = int(file.stem.split("_")[-1])
            indices.append(idx)
        except ValueError:
            pass

    if not indices:
        return 0

    return max(indices) + 1

def load_existing_metadata():
    """
    Carga metadata existente para poder continuar sin repetir imágenes base.
    """
    metadata_path = OUTPUT_DIR / "metadata.csv"

    if not metadata_path.exists():
        return [], set()

    df = pd.read_csv(metadata_path)

    records = df.to_dict("records")

    if "target_image" in df.columns:
        processed_targets = set(df["target_image"].astype(str).tolist())
    else:
        processed_targets = set()

    return records, processed_targets

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


def get_face_area(face):
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def choose_main_face(
    faces,
    img_shape,
    min_main_area_ratio=0.06,
    min_significant_area_ratio=0.025,
    ambiguity_ratio=0.60
):
    """
    Selecciona el rostro principal.

    Regla:
    - Usa el rostro más grande.
    - Ignora rostros pequeños.
    - Descarta solo si hay otro rostro grande cercano en tamaño.
    """

    if faces is None or len(faces) == 0:
        return None, "no_faces", 0

    h, w = img_shape[:2]
    img_area = h * w

    faces_sorted = sorted(faces, key=get_face_area, reverse=True)

    main_face = faces_sorted[0]
    main_area = get_face_area(main_face)
    main_ratio = main_area / img_area

    if main_ratio < min_main_area_ratio:
        return None, f"main_face_too_small_{main_ratio:.3f}", len(faces)

    significant_faces = [
        f for f in faces_sorted
        if (get_face_area(f) / img_area) >= min_significant_area_ratio
    ]

    if len(significant_faces) >= 2:
        second_area = get_face_area(significant_faces[1])
        second_vs_main = second_area / max(main_area, 1)

        if second_vs_main >= ambiguity_ratio:
            return None, f"ambiguous_faces_{second_vs_main:.2f}", len(faces)

    return main_face, "ok", len(faces)

    def face_area(face):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    return max(faces, key=face_area)
def mask_for_selected_face(full_face_mask, face, padding_ratio=0.30):
    """
    Recorta la máscara full_face usando el bbox del rostro seleccionado.
    Esto evita que la mascara_fake tome otro rostro si hay más de una cara.
    """

    if face is None:
        return None

    h, w = full_face_mask.shape

    x1, y1, x2, y2 = face.bbox.astype(int)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    bbox_mask = np.zeros_like(full_face_mask)
    bbox_mask[y1:y2, x1:x2] = 255

    selected_mask = cv2.bitwise_and(full_face_mask, bbox_mask)

    if np.sum(selected_mask) == 0:
        return None

    _, selected_mask = cv2.threshold(selected_mask, 127, 255, cv2.THRESH_BINARY)

    return selected_mask

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
    app.prepare(ctx_id=ctx_id, det_size=(320, 320), det_thresh=0.65)

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


def make_fake_and_auth_masks(masks, target_face):
    full_face_mask = masks["full_face"].copy()

    fake_mask = mask_for_selected_face(full_face_mask, target_face)

    if fake_mask is None:
        return None, None

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

    records, processed_targets = load_existing_metadata()

    next_batch_index = get_next_batch_index()
    generated_this_run = 0

    print(f"Batch actual: {BATCH_NAME}")
    print(f"Método: {METHOD_NAME}")
    print(f"Muestras existentes en metadata: {len(records)}")
    print(f"Targets ya procesados: {len(processed_targets)}")
    print(f"Siguiente índice dentro del batch: {next_batch_index}")

    print(f"Batch actual: {BATCH_NAME}")
    print(f"Método: {METHOD_NAME}")
    print(f"Siguiente índice dentro del batch: {next_batch_index}")

    random.shuffle(items)

    for target_item in tqdm(items):
        if generated_this_run >= MAX_NEW_SAMPLES:
            break

        if next_batch_index >= BATCH_SIZE:
            print(f"Batch {BATCH_NAME} completado.")
            break
        target_key = str(target_item["image_path"])

        if target_key in processed_targets:
            continue
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

        # Detectar rostro principal en target
        target_faces = app.get(target_img)

        target_face, target_reason, target_faces_count = choose_main_face(
            target_faces,
            target_img.shape
        )

        if target_face is None:
            print(f"Descartada target {target_item['stem']}: {target_reason}, detectados={target_faces_count}")
            continue

        # Buscar source válido
        source_face = None
        valid_source_item = None
        valid_source_img = None
        source_reason = None
        source_faces_count = 0

        random.shuffle(source_candidates)

        for candidate in source_candidates:
            candidate_img = read_image(candidate["image_path"])

            if candidate_img is None:
                continue

            candidate_faces = app.get(candidate_img)

            candidate_face, candidate_reason, candidate_count = choose_main_face(
                candidate_faces,
                candidate_img.shape
            )

            if candidate_face is not None:
                source_face = candidate_face
                valid_source_item = candidate
                valid_source_img = candidate_img
                source_reason = candidate_reason
                source_faces_count = candidate_count
                break

        if source_face is None:
            print(f"Descartada target {target_item['stem']}: no se encontró source válido")
            continue

        source_item = valid_source_item
        source_img = valid_source_img

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

        fake_mask, authentic_mask = make_fake_and_auth_masks(
            target_item["masks"],
            target_face
        )
        if fake_mask is None or authentic_mask is None:
            print(f"Descartada {target_item['stem']}: máscara no coincide con rostro seleccionado")
            continue

        if np.sum(fake_mask) == 0:
            continue

        if np.sum(fake_mask) == 0:
            continue

        sample_id = f"{METHOD_PREFIX}_b{BATCH_NUM:03d}_{next_batch_index:04d}"

        img_path, auth_path, fake_path = save_sample(
            sample_id,
            swapped,
            authentic_mask,
            fake_mask
        )

        save_preview(sample_id, swapped, authentic_mask, fake_mask)

        records.append({
            "id": sample_id,
            "tipo": METHOD_NAME,
            "batch": BATCH_NAME,
            "batch_num": BATCH_NUM,
            "batch_index": next_batch_index,
            "global_index": BATCH_NUM * BATCH_SIZE + next_batch_index,
            "target_stem": target_item["stem"],
            "target_image": str(target_item["image_path"]),
            "source_image": str(source_item["image_path"]),
            "target_faces_detected": target_faces_count,
            "source_faces_detected": source_faces_count,
            "face_selection": "main_largest_non_ambiguous",
            "target_face_area": get_face_area(target_face),
            "source_face_area": get_face_area(source_face),
            "imagen_original": str(img_path),
            "mascara_autentica": str(auth_path),
            "mascara_fake": str(fake_path)
        })
        processed_targets.add(target_key)

        metadata_path = OUTPUT_DIR / "metadata.csv"
        pd.DataFrame(records).to_csv(metadata_path, index=False)

        next_batch_index += 1
        generated_this_run += 1


    metadata_path = OUTPUT_DIR / "metadata.csv"
    pd.DataFrame(records).to_csv(metadata_path, index=False)

    print("Muestras generadas en esta ejecución:", generated_this_run)
    print("Total en metadata:", len(records))
    print("Metadata:", metadata_path)
    print("Preview:", PREVIEW_DIR)
    print("Salida:", OUTPUT_DIR)


if __name__ == "__main__":
    main()