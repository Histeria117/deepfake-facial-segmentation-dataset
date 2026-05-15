import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random


BASE_DIR = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")

# Metadata de cada tipo
LOCAL_META = BASE_DIR / "dataset_rostros_avanzado" / "local_inpainting" / "metadata.csv"
FACESWAP_META = BASE_DIR / "dataset_rostros_avanzado" / "faceswap" / "metadata.csv"

OUT_DIR = BASE_DIR / "comparativas_dataset"

IMAGE_SIZE = 256
MAX_ROWS = 8
RANDOM_SEED = 8


# Carpetas donde puede buscar imágenes reales si la ruta del metadata ya no existe
REAL_SEARCH_DIRS = [
    BASE_DIR / "data_raw",
    BASE_DIR / "data_raw" / "face_dataset" / "images",
    BASE_DIR / "data_raw" / "faceswap" / "images",
    BASE_DIR / "data_raw" / "inpainting" / "images",
    BASE_DIR / "data_raw" / "ffhq" / "images",
]

# Carpetas donde puede buscar imágenes/máscaras generadas si la ruta del metadata ya no existe
GENERATED_SEARCH_DIRS = [
    BASE_DIR / "dataset_rostros_avanzado",
    BASE_DIR / "dataset_rostros_avanzado" / "faceswap",
    BASE_DIR / "dataset_rostros_avanzado" / "local_inpainting",
]


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_file_by_name(filename, search_dirs):
    """
    Busca un archivo por nombre dentro de varias carpetas.
    Sirve cuando metadata.csv tiene una ruta vieja.
    """
    filename = Path(filename).name

    for folder in search_dirs:
        if not folder.exists():
            continue

        # Primero búsqueda directa
        direct = folder / filename
        if direct.exists():
            return direct

        # Luego búsqueda recursiva
        matches = list(folder.rglob(filename))
        if matches:
            return matches[0]

    return None


def resolve_path(path_str, kind="generated"):
    """
    Intenta resolver una ruta del metadata.

    1. Usa la ruta exacta si existe.
    2. Si no existe, busca por nombre de archivo en carpetas conocidas.
    """
    if pd.isna(path_str):
        return None

    path = Path(str(path_str))

    if path.exists():
        return path

    if kind == "real":
        return find_file_by_name(path.name, REAL_SEARCH_DIRS)

    return find_file_by_name(path.name, GENERATED_SEARCH_DIRS)


def read_img(path, grayscale=False, kind="generated"):
    path = resolve_path(path, kind=kind)

    if path is None or not path.exists():
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
        (25, IMAGE_SIZE // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA
    )

    return img


def path_stem(path_str):
    if pd.isna(path_str):
        return ""

    return Path(str(path_str)).stem


def build_index_by_target(metadata_df):
    """
    Crea un índice usando el stem de target_image.

    Ejemplo:
    C:/.../000123.png -> 000123
    """
    index = {}

    for _, row in metadata_df.iterrows():
        if "target_image" not in row:
            continue

        target_stem = path_stem(row["target_image"])

        if not target_stem:
            continue

        if target_stem not in index:
            index[target_stem] = []

        index[target_stem].append(row)

    return index


def make_row(real_path, fs_row=None, local_row=None):
    # Imagen real
    real_img = read_img(real_path, grayscale=False, kind="real")

    if real_img is None:
        real_img = blank_cell("REAL?")

    # FaceSwap
    if fs_row is not None:
        fs_img = read_img(fs_row["imagen_original"], kind="generated")
        fs_fake = read_img(fs_row["mascara_fake"], grayscale=True, kind="generated")
        fs_auth = read_img(fs_row["mascara_autentica"], grayscale=True, kind="generated")
    else:
        fs_img = fs_fake = fs_auth = None

    # Local / Inpainting
    if local_row is not None:
        local_img = read_img(local_row["imagen_original"], kind="generated")
        local_fake = read_img(local_row["mascara_fake"], grayscale=True, kind="generated")
        local_auth = read_img(local_row["mascara_autentica"], grayscale=True, kind="generated")
    else:
        local_img = local_fake = local_auth = None

    cells = [
        ("REAL", real_img, (60, 60, 60)),
        ("FACE_SWAP", fs_img if fs_img is not None else blank_cell("NO FS"), (60, 35, 35)),
        ("FS_MASK_FAKE", fs_fake if fs_fake is not None else blank_cell("NO MASK"), (80, 45, 45)),
        ("FS_MASK_AUTH", fs_auth if fs_auth is not None else blank_cell("NO MASK"), (80, 45, 45)),
        ("INPAINTING", local_img if local_img is not None else blank_cell("NO INP"), (35, 55, 80)),
        ("INP_MASK_FAKE", local_fake if local_fake is not None else blank_cell("NO MASK"), (35, 65, 95)),
        ("INP_MASK_AUTH", local_auth if local_auth is not None else blank_cell("NO MASK"), (35, 65, 95)),
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

    print("Muestras local/inpainting:", len(local_df))
    print("Muestras faceswap:", len(fs_df))

    local_index = build_index_by_target(local_df)
    fs_index = build_index_by_target(fs_df)

    common_stems = sorted(set(local_index.keys()) & set(fs_index.keys()))

    rows = []

    if len(common_stems) == 0:
        print("No encontré targets en común entre local y faceswap.")
        print("Haré comparación emparejando muestras por orden.")

        max_rows = min(MAX_ROWS, len(local_df), len(fs_df))

        for i in range(max_rows):
            local_row = local_df.iloc[i]
            fs_row = fs_df.iloc[i]

            # Prioridad: usar la imagen real del local
            real_path = local_row.get("target_image", None)

            # Si local no tiene target_image, intentar con faceswap
            if pd.isna(real_path) or real_path is None:
                real_path = fs_row.get("target_image", None)

            row_img = make_row(
                real_path=real_path,
                fs_row=fs_row,
                local_row=local_row
            )

            rows.append(row_img)

    else:
        print("Targets en común encontrados:", len(common_stems))

        random.shuffle(common_stems)
        selected = common_stems[:MAX_ROWS]

        for stem in selected:
            local_row = random.choice(local_index[stem])
            fs_row = random.choice(fs_index[stem])

            # Usamos la real del local porque debería ser la misma base
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