import os
import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")

IMAGE_DIR = BASE_DIR / "data_raw" / "face_dataset" / "images"
MASK_BASE_DIR = BASE_DIR / "face_parsing_output" / "binary_masks"

OUTPUT_DIR = BASE_DIR / "dataset_rostros_avanzado" / "local"
PREVIEW_DIR = BASE_DIR / "preview_rostros_avanzado"

MAX_SAMPLES = 50
IMAGE_SIZE = 512

REGIONS = ["left_eye", "right_eye", "nose", "lips"]

random.seed(42)
MIN_REGIONS_PER_SAMPLE = 2
MAX_REGIONS_PER_SAMPLE = 4
def choose_num_regions():

    r = random.random()

    if r < 0.70:
        return 1
    elif r < 0.95:
        return 2
    else:
        return 3

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
    masks = {}
    for region in ["full_face"] + REGIONS:
        mask_path = MASK_BASE_DIR / region / f"{stem}.png"
        mask = read_mask(mask_path)
        if mask is None:
            return None
        masks[region] = mask
    return masks


def valid_item(image_path):
    stem = image_path.stem
    masks = get_masks_for_image(stem)
    if masks is None:
        return None

    if np.sum(masks["full_face"]) == 0:
        return None

    has_any_region = any(np.sum(masks[r]) > 0 for r in REGIONS)
    if not has_any_region:
        return None

    return {
        "image_path": image_path,
        "stem": stem,
        "masks": masks
    }


def bounding_rect_from_mask(mask):
    if np.sum(mask) == 0:
        return None
    x, y, w, h = cv2.boundingRect(mask)
    if w < 5 or h < 5:
        return None
    return x, y, w, h


def expand_rect(x, y, w, h, img_w, img_h, scale=0.25):
    px = int(w * scale)
    py = int(h * scale)

    nx = max(0, x - px)
    ny = max(0, y - py)
    nw = min(img_w - nx, w + 2 * px)
    nh = min(img_h - ny, h + 2 * py)

    return nx, ny, nw, nh


def color_transfer_simple(src_patch, dst_patch, src_mask):
    """
    Ajuste sencillo de color usando media y std por canal
    sobre la región útil del parche fuente.
    """
    src = src_patch.astype(np.float32)
    dst = dst_patch.astype(np.float32)

    mask = (src_mask > 0).astype(np.uint8)
    if np.sum(mask) < 10:
        return src_patch

    result = src.copy()

    for c in range(3):
        src_vals = src[:, :, c][mask > 0]
        dst_vals = dst[:, :, c][mask > 0]

        if len(src_vals) < 10 or len(dst_vals) < 10:
            continue

        src_mean, src_std = src_vals.mean(), src_vals.std()
        dst_mean, dst_std = dst_vals.mean(), dst_vals.std()

        if src_std < 1e-6:
            src_std = 1.0
        if dst_std < 1e-6:
            dst_std = 1.0

        channel = src[:, :, c]
        channel = (channel - src_mean) * (dst_std / src_std) + dst_mean
        result[:, :, c] = np.clip(channel, 0, 255)

    return result.astype(np.uint8)


def build_source_canvas_for_clone(target_img, source_img, target_mask, source_mask):
    """
    Toma la región del source, la adapta al bbox del target,
    ajusta color y construye un canvas para seamlessClone.
    """
    rect_t = bounding_rect_from_mask(target_mask)
    rect_s = bounding_rect_from_mask(source_mask)

    if rect_t is None or rect_s is None:
        return None, None

    tx, ty, tw, th = rect_t
    sx, sy, sw, sh = rect_s

    img_h, img_w = target_img.shape[:2]

    tx, ty, tw, th = expand_rect(tx, ty, tw, th, img_w, img_h, scale=0.15)
    sx, sy, sw, sh = expand_rect(sx, sy, sw, sh, img_w, img_h, scale=0.15)

    src_patch = source_img[sy:sy + sh, sx:sx + sw]
    src_mask_patch = source_mask[sy:sy + sh, sx:sx + sw]

    dst_patch = target_img[ty:ty + th, tx:tx + tw]
    dst_mask_patch = target_mask[ty:ty + th, tx:tx + tw]

    if src_patch.size == 0 or dst_patch.size == 0:
        return None, None

    src_patch_resized = cv2.resize(src_patch, (tw, th), interpolation=cv2.INTER_LINEAR)
    src_mask_resized = cv2.resize(src_mask_patch, (tw, th), interpolation=cv2.INTER_NEAREST)

    _, src_mask_resized = cv2.threshold(src_mask_resized, 127, 255, cv2.THRESH_BINARY)

    # Ajuste de color
    src_patch_matched = color_transfer_simple(src_patch_resized, dst_patch, src_mask_resized)

    # Canvas completo para seamlessClone
    src_canvas = np.zeros_like(target_img)
    src_canvas[ty:ty + th, tx:tx + tw] = src_patch_matched

    clone_mask = np.zeros(target_mask.shape, dtype=np.uint8)
    clone_mask[ty:ty + th, tx:tx + tw] = src_mask_resized

    # Intersección con target_mask para que la región fake quede precisa
    clone_mask = cv2.bitwise_and(clone_mask, target_mask)

    # Limpiar un poco la máscara
    kernel = np.ones((5, 5), np.uint8)
    clone_mask = cv2.morphologyEx(clone_mask, cv2.MORPH_CLOSE, kernel)
    clone_mask = cv2.GaussianBlur(clone_mask, (5, 5), 0)
    _, clone_mask_bin = cv2.threshold(clone_mask, 20, 255, cv2.THRESH_BINARY)

    return src_canvas, clone_mask_bin


def seamless_region_clone(target_img, source_img, target_mask, source_mask):
    src_canvas, clone_mask = build_source_canvas_for_clone(
        target_img, source_img, target_mask, source_mask
    )

    if src_canvas is None or clone_mask is None or np.sum(clone_mask) == 0:
        return None, None

    rect = bounding_rect_from_mask(clone_mask)
    if rect is None:
        return None, None

    x, y, w, h = rect
    center = (x + w // 2, y + h // 2)

    try:
        result = cv2.seamlessClone(
            src_canvas,
            target_img,
            clone_mask,
            center,
            cv2.NORMAL_CLONE
        )
    except Exception:
        return None, None

    return result, clone_mask


def make_authentic_mask(full_face_mask, fake_mask):
    authentic = cv2.bitwise_and(full_face_mask, cv2.bitwise_not(fake_mask))
    return authentic


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

        possible_regions = [
            r for r in REGIONS
            if np.sum(target_item["masks"][r]) > 0 and np.sum(source_item["masks"][r]) > 0
        ]

        if not possible_regions:
            continue

        full_face_mask = target_item["masks"]["full_face"]

        num_regions = choose_num_regions()
        num_regions = min(num_regions, len(possible_regions))

        selected_regions = random.sample(possible_regions, num_regions)

        manipulated_img = target_img.copy()
        fake_mask_union = np.zeros_like(full_face_mask)

        used_regions = []

        for region in selected_regions:
            target_mask = target_item["masks"][region]
            source_mask = source_item["masks"][region]

            manipulated_candidate, clone_mask = seamless_region_clone(
                manipulated_img,
                source_img,
                target_mask,
                source_mask
            )

            if manipulated_candidate is None or clone_mask is None or np.sum(clone_mask) == 0:
                continue

            manipulated_img = manipulated_candidate

            # Para entrenamiento usamos la máscara precisa del parsing,
            # no la máscara interna de seamlessClone.
            fake_mask_union = cv2.bitwise_or(fake_mask_union, target_mask)

            used_regions.append(region)

        if len(used_regions) == 0 or np.sum(fake_mask_union) == 0:
            continue

        # Limpiar máscara fake final
        kernel = np.ones((3, 3), np.uint8)
        fake_mask_union = cv2.morphologyEx(fake_mask_union, cv2.MORPH_CLOSE, kernel)
        _, fake_mask_union = cv2.threshold(fake_mask_union, 127, 255, cv2.THRESH_BINARY)

        authentic_mask = make_authentic_mask(full_face_mask, fake_mask_union)

        sample_id = f"local_adv_{generated:05d}"

        img_path, auth_path, fake_path = save_sample(
            sample_id,
            manipulated_img,
            authentic_mask,
            fake_mask_union
        )

        save_preview(sample_id, manipulated_img, authentic_mask, fake_mask_union)

        records.append({
            "id": sample_id,
            "tipo": "local_avanzado",
            "regions": ",".join(used_regions),
            "num_regions": len(used_regions),
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