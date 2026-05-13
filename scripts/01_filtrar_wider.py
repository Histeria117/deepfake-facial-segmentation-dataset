import os
import cv2
import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")

WIDER_ROOT = BASE_DIR / "data_raw" / "WIDER_FACE"
IMAGES_ROOT = WIDER_ROOT / "WIDER_train" / "images"
ANNOTATION_FILE = WIDER_ROOT / "wider_face_split" / "wider_face_train_bbx_gt.txt"

OUT_IMG_DIR = BASE_DIR / "wider_filtrado" / "imagenes"
OUT_META_DIR = BASE_DIR / "wider_filtrado" / "metadata"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_META_DIR.mkdir(parents=True, exist_ok=True)

MIN_FACES = 2
MIN_FACE_SIZE = 80
MAX_IMAGES = 50


def parse_wider_annotations(annotation_file):
    samples = []

    with open(annotation_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0

    while i < len(lines):
        image_rel_path = lines[i]
        i += 1

        if i >= len(lines):
            break

        try:
            num_faces = int(lines[i])
        except:
            break

        i += 1

        faces = []

        for _ in range(num_faces):
            if i >= len(lines):
                break

            parts = lines[i].split()
            i += 1

            if len(parts) < 10:
                continue

            x, y, w, h = map(int, parts[:4])
            blur = int(parts[4])
            expression = int(parts[5])
            illumination = int(parts[6])
            invalid = int(parts[7])
            occlusion = int(parts[8])
            pose = int(parts[9])

            faces.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "blur": blur,
                "expression": expression,
                "illumination": illumination,
                "invalid": invalid,
                "occlusion": occlusion,
                "pose": pose
            })

        samples.append({
            "image_rel_path": image_rel_path,
            "faces": faces
        })

    return samples


def filtrar_rostros(faces, img_w, img_h):
    valid_faces = []

    for face in faces:
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]

        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue

        if face["invalid"] != 0:
            continue

        if face["occlusion"] > 1:
            continue

        if face["blur"] > 1:
            continue

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_w - 1, int(x + w))
        y2 = min(img_h - 1, int(y + h))

        if x2 <= x1 or y2 <= y1:
            continue

        valid_faces.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "w": x2 - x1,
            "h": y2 - y1
        })

    return valid_faces


def main():
    samples = parse_wider_annotations(ANNOTATION_FILE)

    print("Total de imágenes anotadas:", len(samples))

    metadata_general = []
    saved = 0

    for sample in tqdm(samples):
        image_path = IMAGES_ROOT / sample["image_rel_path"]

        if not image_path.exists():
            continue

        img = cv2.imread(str(image_path))

        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        valid_faces = filtrar_rostros(sample["faces"], img_w, img_h)

        if len(valid_faces) < MIN_FACES:
            continue

        filename = f"wider_{saved:05d}.png"
        meta_filename = f"wider_{saved:05d}.json"

        out_img_path = OUT_IMG_DIR / filename
        out_meta_path = OUT_META_DIR / meta_filename

        cv2.imwrite(str(out_img_path), img)

        meta = {
            "id": f"wider_{saved:05d}",
            "image": filename,
            "source_image": sample["image_rel_path"],
            "width": img_w,
            "height": img_h,
            "faces": valid_faces
        }

        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

        metadata_general.append(meta)

        saved += 1

        if saved >= MAX_IMAGES:
            break

    with open(BASE_DIR / "wider_filtrado" / "metadata_filtrado.json", "w", encoding="utf-8") as f:
        json.dump(metadata_general, f, indent=4)

    print("Imágenes filtradas:", saved)
    print("Salida:", OUT_IMG_DIR)


if __name__ == "__main__":
    main()