import os
import cv2
import random
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")

INPUT_DIR = BASE_DIR / "data_raw" / "face_dataset" / "images"
OUTPUT_DIR = BASE_DIR / "dataset_rostros"
PREVIEW_DIR = BASE_DIR / "preview_rostros"

MAX_LOCAL = 5
MAX_FACESWAP = 5

IMAGE_SIZE = 512

random.seed(42)

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.45
)


def ensure_dirs():
    folders = [
        "local/imagen_original",
        "local/mascara_autentica",
        "local/mascara_fake",
        "faceswap/imagen_original",
        "faceswap/mascara_autentica",
        "faceswap/mascara_fake",
    ]

    for folder in folders:
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def get_image_files():
    valid_ext = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    files = []

    for path in INPUT_DIR.rglob("*"):
        if path.suffix.lower() in valid_ext:
            files.append(path)

    return sorted(files)


def read_face_image(path):
    img = cv2.imread(str(path))

    if img is None:
        return None

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


def indices_from_connections(connections):
    indices = set()

    for a, b in connections:
        indices.add(a)
        indices.add(b)

    return sorted(list(indices))


IDX_FACE_OVAL = indices_from_connections(mp_face_mesh.FACEMESH_FACE_OVAL)
IDX_LEFT_EYE = indices_from_connections(mp_face_mesh.FACEMESH_LEFT_EYE)
IDX_RIGHT_EYE = indices_from_connections(mp_face_mesh.FACEMESH_RIGHT_EYE)
IDX_LIPS = indices_from_connections(mp_face_mesh.FACEMESH_LIPS)

IDX_NOSE = [
    1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98,
    115, 168, 195, 197, 220, 275, 294, 326, 327, 344
]


def detect_landmarks(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    h, w = img_bgr.shape[:2]

    points = []

    for lm in landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    return points


def mask_from_landmark_indices(shape, landmarks, indices):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = []

    for idx in indices:
        if idx < len(landmarks):
            x, y = landmarks[idx]

            if 0 <= x < w and 0 <= y < h:
                pts.append([x, y])

    if len(pts) < 3:
        return mask

    pts = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    return mask


def fallback_face_mask(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cx = w // 2
    cy = int(h * 0.48)

    ax = int(w * 0.32)
    ay = int(h * 0.42)

    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    return mask


def fallback_region_mask(shape, region):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if region == "left_eye":
        cx = int(w * 0.38)
        cy = int(h * 0.40)
        ax = int(w * 0.09)
        ay = int(h * 0.045)

    elif region == "right_eye":
        cx = int(w * 0.62)
        cy = int(h * 0.40)
        ax = int(w * 0.09)
        ay = int(h * 0.045)

    elif region == "lips":
        cx = int(w * 0.50)
        cy = int(h * 0.70)
        ax = int(w * 0.16)
        ay = int(h * 0.07)

    elif region == "nose":
        cx = int(w * 0.50)
        cy = int(h * 0.56)
        ax = int(w * 0.09)
        ay = int(h * 0.13)

    else:
        return mask

    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return mask


def build_masks(img_bgr):
    landmarks = detect_landmarks(img_bgr)

    if landmarks is not None:
        face_mask = mask_from_landmark_indices(img_bgr.shape, landmarks, IDX_FACE_OVAL)

        regions = {
            "left_eye": mask_from_landmark_indices(img_bgr.shape, landmarks, IDX_LEFT_EYE),
            "right_eye": mask_from_landmark_indices(img_bgr.shape, landmarks, IDX_RIGHT_EYE),
            "lips": mask_from_landmark_indices(img_bgr.shape, landmarks, IDX_LIPS),
            "nose": mask_from_landmark_indices(img_bgr.shape, landmarks, IDX_NOSE),
        }

        return face_mask, regions, True

    face_mask = fallback_face_mask(img_bgr.shape)

    regions = {
        "left_eye": fallback_region_mask(img_bgr.shape, "left_eye"),
        "right_eye": fallback_region_mask(img_bgr.shape, "right_eye"),
        "lips": fallback_region_mask(img_bgr.shape, "lips"),
        "nose": fallback_region_mask(img_bgr.shape, "nose"),
    }

    return face_mask, regions, False


def alpha_blend(base_bgr, altered_bgr, mask_uint8, blur_size=31):
    if blur_size % 2 == 0:
        blur_size += 1

    alpha = cv2.GaussianBlur(mask_uint8, (blur_size, blur_size), 0)
    alpha = alpha.astype(np.float32) / 255.0
    alpha = alpha[..., None]

    out = base_bgr.astype(np.float32) * (1 - alpha) + altered_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def copy_region_from_source(target_img, source_img, target_mask, source_mask):
    if np.sum(target_mask) == 0 or np.sum(source_mask) == 0:
        return None

    sx, sy, sw, sh = cv2.boundingRect(source_mask)
    dx, dy, dw, dh = cv2.boundingRect(target_mask)

    if sw < 5 or sh < 5 or dw < 5 or dh < 5:
        return None

    source_roi = source_img[sy:sy + sh, sx:sx + sw]
    patch = cv2.resize(source_roi, (dw, dh), interpolation=cv2.INTER_LINEAR)

    canvas = target_img.copy()
    canvas[dy:dy + dh, dx:dx + dw] = patch

    return canvas


def synthetic_local_perturbation(img_bgr, mask):
    blurred = cv2.GaussianBlur(img_bgr, (31, 31), 0)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.85, 0, 255)

    shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    altered = cv2.addWeighted(blurred, 0.45, shifted, 0.55, 0)
    return alpha_blend(img_bgr, altered, mask, blur_size=21)


def make_authentic_mask(face_mask, fake_mask):
    return cv2.bitwise_and(face_mask, cv2.bitwise_not(fake_mask))


def save_triplet(kind, sample_id, img_bgr, authentic_mask, fake_mask):
    img_dir = OUTPUT_DIR / kind / "imagen_original"
    auth_dir = OUTPUT_DIR / kind / "mascara_autentica"
    fake_dir = OUTPUT_DIR / kind / "mascara_fake"

    img_path = img_dir / f"{sample_id}.png"
    auth_path = auth_dir / f"{sample_id}.png"
    fake_path = fake_dir / f"{sample_id}.png"

    cv2.imwrite(str(img_path), img_bgr)
    cv2.imwrite(str(auth_path), authentic_mask)
    cv2.imwrite(str(fake_path), fake_mask)

    return img_path, auth_path, fake_path


def make_preview(kind):
    img_dir = OUTPUT_DIR / kind / "imagen_original"
    auth_dir = OUTPUT_DIR / kind / "mascara_autentica"
    fake_dir = OUTPUT_DIR / kind / "mascara_fake"

    files = sorted(os.listdir(img_dir))

    for file in files:
        img = cv2.imread(str(img_dir / file))
        auth = cv2.imread(str(auth_dir / file), cv2.IMREAD_GRAYSCALE)
        fake = cv2.imread(str(fake_dir / file), cv2.IMREAD_GRAYSCALE)

        if img is None or auth is None or fake is None:
            continue

        auth_bgr = cv2.cvtColor(auth, cv2.COLOR_GRAY2BGR)
        fake_bgr = cv2.cvtColor(fake, cv2.COLOR_GRAY2BGR)

        grid = np.hstack([img, auth_bgr, fake_bgr])
        out_path = PREVIEW_DIR / f"{kind}_{file}"

        cv2.imwrite(str(out_path), grid)


def main():
    ensure_dirs()

    image_files = get_image_files()

    print("Imágenes encontradas:", len(image_files))

    if len(image_files) < 2:
        print("Necesitas al menos 2 imágenes en:", INPUT_DIR)
        return

    generated_local = 0
    generated_faceswap = 0

    records = []

    regions = ["left_eye", "right_eye", "lips", "nose"]

    shuffled = image_files[:]
    random.shuffle(shuffled)

    for img_path in tqdm(shuffled):
        target_img = read_face_image(img_path)

        if target_img is None:
            continue

        target_face_mask, target_regions, landmarks_ok = build_masks(target_img)

        if np.sum(target_face_mask) == 0:
            continue

        source_path = random.choice([p for p in image_files if p != img_path])
        source_img = read_face_image(source_path)

        if source_img is None:
            continue

        source_face_mask, source_regions, source_landmarks_ok = build_masks(source_img)

        if generated_local < MAX_LOCAL:
            region = random.choice(regions)

            fake_mask = target_regions[region]
            source_mask = source_regions[region]

            if np.sum(fake_mask) > 0:
                canvas = copy_region_from_source(
                    target_img,
                    source_img,
                    fake_mask,
                    source_mask
                )

                if canvas is not None:
                    manipulated = alpha_blend(target_img, canvas, fake_mask, blur_size=21)
                else:
                    manipulated = synthetic_local_perturbation(target_img, fake_mask)

                authentic_mask = make_authentic_mask(target_face_mask, fake_mask)

                sample_id = f"local_face_{generated_local:05d}"

                img_out, auth_out, fake_out = save_triplet(
                    "local",
                    sample_id,
                    manipulated,
                    authentic_mask,
                    fake_mask
                )

                records.append({
                    "id": sample_id,
                    "tipo": "local",
                    "region": region,
                    "target_image": str(img_path),
                    "source_image": str(source_path),
                    "target_landmarks_ok": landmarks_ok,
                    "source_landmarks_ok": source_landmarks_ok,
                    "imagen_original": str(img_out),
                    "mascara_autentica": str(auth_out),
                    "mascara_fake": str(fake_out)
                })

                generated_local += 1

        if generated_faceswap < MAX_FACESWAP:
            fake_mask = target_face_mask

            canvas = copy_region_from_source(
                target_img,
                source_img,
                fake_mask,
                source_face_mask
            )

            if canvas is not None:
                manipulated = alpha_blend(target_img, canvas, fake_mask, blur_size=41)

                authentic_mask = make_authentic_mask(target_face_mask, fake_mask)

                sample_id = f"faceswap_face_{generated_faceswap:05d}"

                img_out, auth_out, fake_out = save_triplet(
                    "faceswap",
                    sample_id,
                    manipulated,
                    authentic_mask,
                    fake_mask
                )

                records.append({
                    "id": sample_id,
                    "tipo": "faceswap_sintetico",
                    "region": "full_face",
                    "target_image": str(img_path),
                    "source_image": str(source_path),
                    "target_landmarks_ok": landmarks_ok,
                    "source_landmarks_ok": source_landmarks_ok,
                    "imagen_original": str(img_out),
                    "mascara_autentica": str(auth_out),
                    "mascara_fake": str(fake_out)
                })

                generated_faceswap += 1

        if generated_local >= MAX_LOCAL and generated_faceswap >= MAX_FACESWAP:
            break

    metadata_path = OUTPUT_DIR / "metadata.csv"
    pd.DataFrame(records).to_csv(metadata_path, index=False)

    make_preview("local")
    make_preview("faceswap")

    print("Locales generadas:", generated_local)
    print("Faceswap generadas:", generated_faceswap)
    print("Metadata:", metadata_path)
    print("Previews:", PREVIEW_DIR)


if __name__ == "__main__":
    main()