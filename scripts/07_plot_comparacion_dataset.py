import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random


BASE_DIR = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")

LOCAL_META = BASE_DIR / "dataset_rostros_avanzado" / "local_inpainting" / "metadata.csv"
FACESWAP_META = BASE_DIR / "dataset_rostros_avanzado" / "faceswap" / "metadata.csv"

OUT_DIR = BASE_DIR / "comparativas_dataset"

IMAGE_SIZE = 256
MAX_ROWS = 8
RANDOM_SEED = 32


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_img(path, grayscale=False):
    path = Path(path)

    if not path.exists():
        return None

    if grayscale:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    img = cv2.imread(str(path))
    if img is None:
        return None

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


def add_header(img, text, color=(40, 40, 40)):
    header_h = 36
    h, w = img.shape[:2]

    header = np.zeros((header_h, w, 3), dtype=np.uint8)
    header[:] = color

    cv2.putText(
        header,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return np.vstack([header, img])


def add_border(img, color=(30, 30, 30), thickness=2):
    return cv2.copyMakeBorder(
        img,
        thickness,
        thickness,
        thickness,
        thickness,
        cv2.BORDER_CONSTANT,
        value=color
    )


def blank_cell(text="NO DATA"):
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    img[:] = (45, 45, 45)

    cv2.putText(
        img,
        text,
        (40, IMAGE_SIZE // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA
    )

    return img


def path_stem(path_str):
    return Path(path_str).stem


def build_index_by_target(metadata_df):
    index = {}

    for _, row in metadata_df.iterrows():
        target_stem = path_stem(row["target_image"])

        if target_stem not in index:
            index[target_stem] = []

        index[target_stem].append(row)

    return index


def make_row(real_path, fs_row=None, local_row=None):
    real_img = read_img(real_path)

    if real_img is None:
        real_img = blank_cell("REAL?")

    # FaceSwap
    if fs_row is not None:
        fs_img = read_img(fs_row["imagen_original"])
        fs_fake = read_img(fs_row["mascara_fake"], grayscale=True)
        fs_auth = read_img(fs_row["mascara_autentica"], grayscale=True)
    else:
        fs_img = fs_fake = fs_auth = None

    # Local / Inpainting
    if local_row is not None:
        local_img = read_img(local_row["imagen_original"])
        local_fake = read_img(local_row["mascara_fake"], grayscale=True)
        local_auth = read_img(local_row["mascara_autentica"], grayscale=True)
    else:
        local_img = local_fake = local_auth = None

    cells = [
        ("REAL", real_img, (60, 60, 60)),
        ("FACE_SWAP", fs_img if fs_img is not None else blank_cell(), (60, 35, 35)),
        ("FS_MASK_FAKE", fs_fake if fs_fake is not None else blank_cell(), (80, 45, 45)),
        ("FS_MASK_AUTH", fs_auth if fs_auth is not None else blank_cell(), (80, 45, 45)),
        ("LOCAL / INPAINTING", local_img if local_img is not None else blank_cell(), (35, 55, 80)),
        ("LOCAL_MASK_FAKE", local_fake if local_fake is not None else blank_cell(), (35, 65, 95)),
        ("LOCAL_MASK_AUTH", local_auth if local_auth is not None else blank_cell(), (35, 65, 95)),
    ]

    row_imgs = []

    for title, img, color in cells:
        img = add_header(img, title, color=color)
        img = add_border(img)
        row_imgs.append(img)

    return np.hstack(row_imgs)


def main():
    ensure_dirs()
    random.seed(RANDOM_SEED)

    if not LOCAL_META.exists():
        raise FileNotFoundError(f"No encontré metadata local: {LOCAL_META}")

    if not FACESWAP_META.exists():
        raise FileNotFoundError(f"No encontré metadata faceswap: {FACESWAP_META}")

    local_df = pd.read_csv(LOCAL_META)
    fs_df = pd.read_csv(FACESWAP_META)

    local_index = build_index_by_target(local_df)
    fs_index = build_index_by_target(fs_df)

    common_stems = sorted(set(local_index.keys()) & set(fs_index.keys()))

    if len(common_stems) == 0:
        print("No encontré targets en común entre local y faceswap.")
        print("Haré comparación emparejando muestras por orden.")

        max_rows = min(MAX_ROWS, len(local_df), len(fs_df))
        rows = []

        for i in range(max_rows):
            local_row = local_df.iloc[i]
            fs_row = fs_df.iloc[i]
            real_path = local_row["target_image"]

            row_img = make_row(
                real_path=real_path,
                fs_row=fs_row,
                local_row=local_row
            )
            rows.append(row_img)

    else:
        random.shuffle(common_stems)
        selected = common_stems[:MAX_ROWS]

        rows = []

        for stem in selected:
            local_row = random.choice(local_index[stem])
            fs_row = random.choice(fs_index[stem])
            real_path = local_row["target_image"]

            row_img = make_row(
                real_path=real_path,
                fs_row=fs_row,
                local_row=local_row
            )
            rows.append(row_img)

    if len(rows) == 0:
        print("No se pudo generar ninguna fila.")
        return

    grid = np.vstack(rows)

    out_path = OUT_DIR / "comparacion_real_faceswap_local_mascaras.png"
    cv2.imwrite(str(out_path), grid)

    print("Comparativa guardada en:")
    print(out_path)


if __name__ == "__main__":
    main()