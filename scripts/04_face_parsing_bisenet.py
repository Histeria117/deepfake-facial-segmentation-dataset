import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms.functional import normalize
from facexlib.parsing import init_parsing_model


BASE_DIR = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")
#Input dir_faceswap
#INPUT_DIR = BASE_DIR / "data_raw" / "face_dataset" / "images_faceswap"

#Output dir_faceswap
#OUTPUT_DIR = BASE_DIR / "face_parsing_output_faceswap"
#----------------------------------------------------------------------------

#Input dir_inpainting
INPUT_DIR = BASE_DIR / "data_raw" / "face_dataset" / "images_inpainting"

#Output dir_inpainting
OUTPUT_DIR = BASE_DIR / "face_parsing_output_inpainting"






IMAGE_SIZE = 512

# Clases típicas de face parsing (CelebAMask-HQ / BiSeNet)
LABELS = {
    "background": [0],
    "skin": [1],
    "left_brow": [2],
    "right_brow": [3],
    "left_eye": [4],
    "right_eye": [5],
    "eye_glass": [6],
    "left_ear": [7],
    "right_ear": [8],
    "ear_ring": [9],
    "nose": [10],
    "mouth": [11],
    "upper_lip": [12],
    "lower_lip": [13],
    "neck": [14],
    "necklace": [15],
    "cloth": [16],
    "hair": [17],
    "hat": [18],
}

# Máscaras que nos interesan para tu proyecto
REGION_MAP = {
    "left_eye": [4],
    "right_eye": [5],
    "nose": [10],
    "mouth": [11],
    "lips": [12, 13],
    "brows": [2, 3],
    # full_face interna, útil para fake/authentic mask
    "full_face": [1, 2, 3, 4, 5,6, 10, 11, 12, 13],
}


def ensure_dirs():
    folders = [
        "parsing_index",
        "preview",
    ]

    for region in REGION_MAP.keys():
        folders.append(f"binary_masks/{region}")

    for folder in folders:
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)


def get_image_files():
    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    files = []

    for path in INPUT_DIR.rglob("*"):
        if path.suffix.lower() in valid_ext:
            files.append(path)

    return sorted(files)


def read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


def preprocess_for_parser(img_bgr, device):
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
    img = img.unsqueeze(0).to(device)
    return img


def parse_image(parser, img_bgr, device):
    x = preprocess_for_parser(img_bgr, device)

    with torch.no_grad():
        out = parser(x)[0]  # shape: [1, C, H, W] o similar
        if out.dim() == 4:
            out = out.squeeze(0)
        parsing = out.argmax(dim=0).cpu().numpy().astype(np.uint8)

    return parsing


def build_binary_mask(parsing_map, class_ids):
    mask = np.isin(parsing_map, class_ids).astype(np.uint8) * 255
    return mask


def colorize_parsing(parsing_map):
    # Paleta simple para visualización
    color_table = {
        0: (0, 0, 0),
        1: (255, 220, 180),   # skin
        2: (0, 255, 0),       # left brow
        3: (0, 200, 0),       # right brow
        4: (255, 0, 0),       # left eye
        5: (200, 0, 0),       # right eye
        6: (255, 255, 0),     # glasses
        7: (255, 0, 255),     # left ear
        8: (200, 0, 255),     # right ear
        9: (255, 128, 0),     # ear ring
        10: (0, 255, 255),    # nose
        11: (0, 128, 255),    # mouth
        12: (128, 0, 255),    # upper lip
        13: (255, 0, 128),    # lower lip
        14: (128, 128, 0),    # neck
        15: (128, 255, 0),    # necklace
        16: (128, 128, 128),  # cloth
        17: (80, 80, 255),    # hair
        18: (255, 255, 255),  # hat
    }

    h, w = parsing_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, rgb in color_table.items():
        color[parsing_map == cls_id] = rgb

    return color


def make_preview(img_bgr, parsing_color, full_face_mask, left_eye_mask, right_eye_mask, nose_mask, lips_mask):
    # Convertir máscaras a 3 canales
    masks = []
    for m in [full_face_mask, left_eye_mask, right_eye_mask, nose_mask, lips_mask]:
        masks.append(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))

    grid_top = np.hstack([img_bgr, parsing_color])
    grid_bottom = np.hstack(masks)

    # Ajustar anchos para concatenar verticalmente
    w_top = grid_top.shape[1]
    w_bottom = grid_bottom.shape[1]

    if w_bottom < w_top:
        pad = np.zeros((grid_bottom.shape[0], w_top - w_bottom, 3), dtype=np.uint8)
        grid_bottom = np.hstack([grid_bottom, pad])
    elif w_top < w_bottom:
        pad = np.zeros((grid_top.shape[0], w_bottom - w_top, 3), dtype=np.uint8)
        grid_top = np.hstack([grid_top, pad])

    preview = np.vstack([grid_top, grid_bottom])
    return preview


def main():
    ensure_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device)

    print("Cargando modelo de face parsing...")
    parser = init_parsing_model(model_name="bisenet", device=device)
    parser.eval()

    image_files = get_image_files()
    print("Imágenes encontradas:", len(image_files))

    if len(image_files) == 0:
        print("No se encontraron imágenes en:", INPUT_DIR)
        return

    for img_path in tqdm(image_files):
        img = read_image(img_path)
        if img is None:
            continue

        try:
            parsing_map = parse_image(parser, img, device)
        except Exception as e:
            print(f"Error procesando {img_path.name}: {e}")
            continue

        stem = img_path.stem

        # Guardar parsing indexado
        cv2.imwrite(str(OUTPUT_DIR / "parsing_index" / f"{stem}.png"), parsing_map)

        # Construir máscaras binarias
        region_masks = {}
        for region_name, class_ids in REGION_MAP.items():
            mask = build_binary_mask(parsing_map, class_ids)
            region_masks[region_name] = mask
            cv2.imwrite(
                str(OUTPUT_DIR / "binary_masks" / region_name / f"{stem}.png"),
                mask
            )

        parsing_color = colorize_parsing(parsing_map)

        preview = make_preview(
            img,
            parsing_color,
            region_masks["full_face"],
            region_masks["left_eye"],
            region_masks["right_eye"],
            region_masks["nose"],
            region_masks["lips"],
        )

        cv2.imwrite(str(OUTPUT_DIR / "preview" / f"{stem}.png"), preview)

    print("Proceso terminado.")
    print("Salida:", OUTPUT_DIR)


if __name__ == "__main__":
    main()