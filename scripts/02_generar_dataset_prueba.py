import os
import cv2
import json
import random
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm


BASE_DIR = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")

FILTRADO_IMG_DIR = BASE_DIR / "wider_filtrado" / "imagenes"
FILTRADO_META_PATH = BASE_DIR / "wider_filtrado" / "metadata_filtrado.json"

DATASET_DIR = BASE_DIR / "dataset_prueba"

MAX_LOCAL = 5
MAX_FACESWAP = 5

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
        (DATASET_DIR / folder).mkdir(parents=True, exist_ok=True)

    (BASE_DIR / "preview").mkdir(parents=True, exist_ok=True)


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


def expand_box(face, img_w, img_h, pad=0.35):
    x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]

    w = x2 - x1
    h = y2 - y1

    px = int(w * pad)
    py = int(h * pad)

    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(img_w - 1, x2 + px)
    ny2 = min(img_h - 1, y2 + py)

    return nx1, ny1, nx2, ny2


def detect_landmarks_in_bbox(img_rgb, face):
    img_h, img_w = img_rgb.shape[:2]

    x1, y1, x2, y2 = expand_box(face, img_w, img_h)
    crop = img_rgb[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    results = face_mesh.process(crop)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    crop_h, crop_w = crop.shape[:2]

    points = []

    for lm in landmarks.landmark:
        px = int(lm.x * crop_w) + x1
        py = int(lm.y * crop_h) + y1
        points.append((px, py))

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


def ellipse_face_mask(shape, face):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    ax = int((x2 - x1) * 0.45)
    ay = int((y2 - y1) * 0.55)

    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    return mask


def approximate_region_mask(shape, face, region):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]

    fw = x2 - x1
    fh = y2 - y1

    if region == "left_eye":
        cx = int(x1 + fw * 0.35)
        cy = int(y1 + fh * 0.38)
        ax = int(fw * 0.15)
        ay = int(fh * 0.08)

    elif region == "right_eye":
        cx = int(x1 + fw * 0.65)
        cy = int(y1 + fh * 0.38)
        ax = int(fw * 0.15)
        ay = int(fh * 0.08)

    elif region == "lips":
        cx = int(x1 + fw * 0.50)
        cy = int(y1 + fh * 0.73)
        ax = int(fw * 0.22)
        ay = int(fh * 0.10)

    elif region == "nose":
        cx = int(x1 + fw * 0.50)
        cy = int(y1 + fh * 0.55)
        ax = int(fw * 0.14)
        ay = int(fh * 0.18)

    else:
        return mask

    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    return mask


def build_face_info(img_rgb, face):
    landmarks = detect_landmarks_in_bbox(img_rgb, face)

    if landmarks is not None:
        face_mask = mask_from_landmark_indices(img_rgb.shape, landmarks, IDX_FACE_OVAL)

        regions = {
            "left_eye": mask_from_landmark_indices(img_rgb.shape, landmarks, IDX_LEFT_EYE),
            "right_eye": mask_from_landmark_indices(img_rgb.shape, landmarks, IDX_RIGHT_EYE),
            "lips": mask_from_landmark_indices(img_rgb.shape, landmarks, IDX_LIPS),
            "nose": mask_from_landmark_indices(img_rgb.shape, landmarks, IDX_NOSE),
        }

        return {
            "bbox": face,
            "landmarks_ok": True,
            "face_mask": face_mask,
            "regions": regions
        }

    face_mask = ellipse_face_mask(img_rgb.shape, face)

    regions = {
        "left_eye": approximate_region_mask(img_rgb.shape, face, "left_eye"),
        "right_eye": approximate_region_mask(img_rgb.shape, face, "right_eye"),
        "lips": approximate_region_mask(img_rgb.shape, face, "lips"),
        "nose": approximate_region_mask(img_rgb.shape, face, "nose"),
    }

    return {
        "bbox": face,
        "landmarks_ok": False,
        "face_mask": face_mask,
        "regions": regions
    }


def build_all_faces_info(img_rgb, meta):
    infos = []

    for face in meta["faces"]:
        info = build_face_info(img_rgb, face)

        if np.sum(info["face_mask"]) > 0:
            infos.append(info)

    return infos


def alpha_blend(base_bgr, altered_bgr, mask_uint8, blur_size=31):
    if blur_size % 2 == 0:
        blur_size += 1

    alpha = cv2.GaussianBlur(mask_uint8, (blur_size, blur_size), 0)
    alpha = alpha.astype(np.float32) / 255.0
    alpha = alpha[..., None]

    out = base_bgr.astype(np.float32) * (1 - alpha) + altered_bgr.astype(np.float32) * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def synthetic_local_perturbation(img_bgr, mask):
    blurred = cv2.GaussianBlur(img_bgr, (25, 25), 0)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.85, 0, 255)

    shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    mixed = cv2.addWeighted(blurred, 0.40, shifted, 0.60, 0)

    return alpha_blend(img_bgr, mixed, mask, blur_size=21)


def resized_patch_canvas(img_bgr, src_mask, dst_mask):
    if np.sum(src_mask) == 0 or np.sum(dst_mask) == 0:
        return None

    sx, sy, sw, sh = cv2.boundingRect(src_mask)
    dx, dy, dw, dh = cv2.boundingRect(dst_mask)

    if sw < 5 or sh < 5 or dw < 5 or dh < 5:
        return None

    src_roi = img_bgr[sy:sy + sh, sx:sx + sw]
    patch = cv2.resize(src_roi, (dw, dh), interpolation=cv2.INTER_LINEAR)

    canvas = img_bgr.copy()
    canvas[dy:dy + dh, dx:dx + dw] = patch

    return canvas


def make_authentic_mask(face_infos, fake_mask):
    all_faces_mask = np.zeros_like(fake_mask)

    for info in face_infos:
        all_faces_mask = cv2.bitwise_or(all_faces_mask, info["face_mask"])

    authentic_mask = cv2.bitwise_and(all_faces_mask, cv2.bitwise_not(fake_mask))

    return authentic_mask


def save_triplet(kind, sample_id, img_bgr, authentic_mask, fake_mask):
    img_dir = DATASET_DIR / kind / "imagen_original"
    auth_dir = DATASET_DIR / kind / "mascara_autentica"
    fake_dir = DATASET_DIR / kind / "mascara_fake"

    img_path = img_dir / f"{sample_id}.png"
    auth_path = auth_dir / f"{sample_id}.png"
    fake_path = fake_dir / f"{sample_id}.png"

    cv2.imwrite(str(img_path), img_bgr)
    cv2.imwrite(str(auth_path), authentic_mask)
    cv2.imwrite(str(fake_path), fake_mask)

    return img_path, auth_path, fake_path


def make_preview(kind):
    img_dir = DATASET_DIR / kind / "imagen_original"
    auth_dir = DATASET_DIR / kind / "mascara_autentica"
    fake_dir = DATASET_DIR / kind / "mascara_fake"

    files = sorted(os.listdir(img_dir))

    for file in files:
        img = cv2.imread(str(img_dir / file))
        auth = cv2.imread(str(auth_dir / file), cv2.IMREAD_GRAYSCALE)
        fake = cv2.imread(str(fake_dir / file), cv2.IMREAD_GRAYSCALE)

        if img is None or auth is None or fake is None:
            continue

        auth_bgr = cv2.cvtColor(auth, cv2.COLOR_GRAY2BGR)
        fake_bgr = cv2.cvtColor(fake, cv2.COLOR_GRAY2BGR)

        h = img.shape[0]
        auth_bgr = cv2.resize(auth_bgr, (img.shape[1], h))
        fake_bgr = cv2.resize(fake_bgr, (img.shape[1], h))

        grid = np.hstack([img, auth_bgr, fake_bgr])

        out_path = BASE_DIR / "preview" / f"{kind}_{file}"
        cv2.imwrite(str(out_path), grid)


def main():
    ensure_dirs()

    with open(FILTRADO_META_PATH, "r", encoding="utf-8") as f:
        metadata_general = json.load(f)

    generated_local = 0
    generated_faceswap = 0

    records = []

    regions = ["left_eye", "right_eye", "lips", "nose"]

    for meta in tqdm(metadata_general):
        img_path = FILTRADO_IMG_DIR / meta["image"]
        img_bgr = cv2.imread(str(img_path))

        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        face_infos = build_all_faces_info(img_rgb, meta)

        if len(face_infos) == 0:
            continue

        if generated_local < MAX_LOCAL:
            target = random.choice(face_infos)
            region = random.choice(regions)

            fake_mask = target["regions"][region]

            if np.sum(fake_mask) > 0:
                source_candidates = [f for f in face_infos if f is not target]

                if len(source_candidates) > 0:
                    source = random.choice(source_candidates)
                    source_mask = source["regions"][region]
                    canvas = resized_patch_canvas(img_bgr, source_mask, fake_mask)

                    if canvas is not None:
                        manipulated = alpha_blend(img_bgr, canvas, fake_mask, blur_size=21)
                    else:
                        manipulated = synthetic_local_perturbation(img_bgr, fake_mask)
                else:
                    manipulated = synthetic_local_perturbation(img_bgr, fake_mask)

                authentic_mask = make_authentic_mask(face_infos, fake_mask)

                sample_id = f"local_{generated_local:05d}"

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
                    "source_image": meta["image"],
                    "landmarks_used": target["landmarks_ok"],
                    "imagen_original": str(img_out),
                    "mascara_autentica": str(auth_out),
                    "mascara_fake": str(fake_out)
                })

                generated_local += 1

        if generated_faceswap < MAX_FACESWAP and len(face_infos) >= 2:
            target = random.choice(face_infos)
            source_candidates = [f for f in face_infos if f is not target]

            if len(source_candidates) > 0:
                source = random.choice(source_candidates)

                fake_mask = target["face_mask"]

                sx1 = source["bbox"]["x1"]
                sy1 = source["bbox"]["y1"]
                sx2 = source["bbox"]["x2"]
                sy2 = source["bbox"]["y2"]

                tx1 = target["bbox"]["x1"]
                ty1 = target["bbox"]["y1"]
                tx2 = target["bbox"]["x2"]
                ty2 = target["bbox"]["y2"]

                src_face = img_bgr[sy1:sy2, sx1:sx2]

                if src_face.size > 0 and (tx2 - tx1) > 5 and (ty2 - ty1) > 5:
                    resized_face = cv2.resize(
                        src_face,
                        (tx2 - tx1, ty2 - ty1),
                        interpolation=cv2.INTER_LINEAR
                    )

                    canvas = img_bgr.copy()
                    canvas[ty1:ty2, tx1:tx2] = resized_face

                    manipulated = alpha_blend(img_bgr, canvas, fake_mask, blur_size=35)

                    authentic_mask = make_authentic_mask(face_infos, fake_mask)

                    sample_id = f"faceswap_{generated_faceswap:05d}"

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
                        "source_image": meta["image"],
                        "landmarks_used": target["landmarks_ok"],
                        "imagen_original": str(img_out),
                        "mascara_autentica": str(auth_out),
                        "mascara_fake": str(fake_out)
                    })

                    generated_faceswap += 1

        if generated_local >= MAX_LOCAL and generated_faceswap >= MAX_FACESWAP:
            break

    metadata_csv_path = DATASET_DIR / "metadata.csv"
    pd.DataFrame(records).to_csv(metadata_csv_path, index=False)

    make_preview("local")
    make_preview("faceswap")

    print("Locales generadas:", generated_local)
    print("Faceswap generadas:", generated_faceswap)
    print("Metadata:", metadata_csv_path)
    print("Previews:", BASE_DIR / "preview")


if __name__ == "__main__":
    main()