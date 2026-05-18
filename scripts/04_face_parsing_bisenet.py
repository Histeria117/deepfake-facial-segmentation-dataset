import cv2
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm
from torchvision.transforms.functional import normalize
from facexlib.parsing import init_parsing_model


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

DIR_PRINCIPAL = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")

# ------------------------------------------------------------
# Configuración para FaceSwap
# ------------------------------------------------------------

DIR_ENTRADA = DIR_PRINCIPAL / "data_raw" / "face_dataset" / "images_faceswap"
DIR_SALIDA = DIR_PRINCIPAL / "face_parsing_output_faceswap"

# ------------------------------------------------------------
# Configuración para Inpainting
# ------------------------------------------------------------

# DIR_ENTRADA = DIR_PRINCIPAL / "data_raw" / "face_dataset" / "images_inpainting"
# DIR_SALIDA = DIR_PRINCIPAL / "face_parsing_output_inpainting"


TAM_IMAGEN = 512

ETIQUETAS = {
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


MAPA_REGIONES = {
    "left_eye": [4],
    "right_eye": [5],
    "nose": [10],
    "mouth": [11],
    "lips": [12, 13],
    "brows": [2, 3],
    "full_face": [1, 2, 3, 4, 5, 6, 10, 11, 12, 13],
}

def crear_directorios():
    carpetas = [
        "parsing_index",
        "preview",
    ]

    for region in MAPA_REGIONES.keys():
        carpetas.append(f"binary_masks/{region}")

    for carpeta in carpetas:
        (DIR_SALIDA / carpeta).mkdir(parents=True, exist_ok=True)



def obtener_archivos_imagen():
    extensiones_validas = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    archivos = []

    for ruta in DIR_ENTRADA.rglob("*"):
        if ruta.suffix.lower() in extensiones_validas:
            archivos.append(ruta)

    return sorted(archivos)


def leer_imagen(ruta):
    imagen = cv2.imread(str(ruta))

    if imagen is None:
        return None

    imagen = cv2.resize(
        imagen,
        (TAM_IMAGEN, TAM_IMAGEN),
        interpolation=cv2.INTER_AREA
    )

    return imagen

def preprocesar_para_parser(imagen_bgr, dispositivo):
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    imagen = imagen_rgb.astype(np.float32) / 255.0
    imagen = torch.from_numpy(
        np.transpose(imagen, (2, 0, 1))
    ).float()
    normalize(
        imagen,
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
        inplace=True
    )
    imagen = imagen.unsqueeze(0).to(dispositivo)
    return imagen

def segmentar_imagen(modelo_parser, imagen_bgr, dispositivo):
    entrada = preprocesar_para_parser(imagen_bgr, dispositivo)

    with torch.no_grad():
        salida = modelo_parser(entrada)[0]

        if salida.dim() == 4:
            salida = salida.squeeze(0)
        mapa_parsing = salida.argmax(dim=0).cpu().numpy().astype(np.uint8)

    return mapa_parsing


def construir_mascara_binaria(mapa_parsing, ids_clase):
    mascara = np.isin(mapa_parsing, ids_clase).astype(np.uint8) * 255
    return mascara


def colorear_mapa_parsing(mapa_parsing):
    tabla_colores = {
        0: (0, 0, 0),
        1: (255, 220, 180),    # piel
        2: (0, 255, 0),        # ceja izquierda
        3: (0, 200, 0),        # ceja derecha
        4: (255, 0, 0),        # ojo izquierdo
        5: (200, 0, 0),        # ojo derecho
        6: (255, 255, 0),      # lentes
        7: (255, 0, 255),      # oreja izquierda
        8: (200, 0, 255),      # oreja derecha
        9: (255, 128, 0),      # arete
        10: (0, 255, 255),     # nariz
        11: (0, 128, 255),     # boca
        12: (128, 0, 255),     # labio superior
        13: (255, 0, 128),     # labio inferior
        14: (128, 128, 0),     # cuello
        15: (128, 255, 0),     # collar
        16: (128, 128, 128),   # pañuelo
        17: (80, 80, 255),     # cabello
        18: (255, 255, 255),   # sombrero
    }

    alto, ancho = mapa_parsing.shape
    imagen_color = np.zeros((alto, ancho, 3), dtype=np.uint8)

    for id_clase, color_rgb in tabla_colores.items():
        imagen_color[mapa_parsing == id_clase] = color_rgb

    return imagen_color

def crear_preview(
    imagen_bgr,
    mapa_parsing_color,
    mascara_full_face,
    mascara_ojo_izquierdo,
    mascara_ojo_derecho,
    mascara_nariz,
    mascara_labios
):
    mascaras = []

    for mascara in [
        mascara_full_face,
        mascara_ojo_izquierdo,
        mascara_ojo_derecho,
        mascara_nariz,
        mascara_labios
    ]:
        mascaras.append(cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR))

    fila_superior = np.hstack([
        imagen_bgr,
        mapa_parsing_color
    ])

    fila_inferior = np.hstack(mascaras)
    ancho_superior = fila_superior.shape[1]
    ancho_inferior = fila_inferior.shape[1]

    if ancho_inferior < ancho_superior:
        relleno = np.zeros(
            (fila_inferior.shape[0], ancho_superior - ancho_inferior, 3),
            dtype=np.uint8
        )
        fila_inferior = np.hstack([fila_inferior, relleno])

    elif ancho_superior < ancho_inferior:
        relleno = np.zeros(
            (fila_superior.shape[0], ancho_inferior - ancho_superior, 3),
            dtype=np.uint8
        )
        fila_superior = np.hstack([fila_superior, relleno])

    preview = np.vstack([
        fila_superior,
        fila_inferior
    ])

    return preview
def principal():
    crear_directorios()

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", dispositivo)

    print("Cargando modelo de face parsing...")
    modelo_parser = init_parsing_model(
        model_name="bisenet",
        device=dispositivo
    )

    modelo_parser.eval()

    archivos_imagen = obtener_archivos_imagen()
    print("Imágenes encontradas:", len(archivos_imagen))

    if len(archivos_imagen) == 0:
        print("No se encontraron imágenes en:", DIR_ENTRADA)
        return

    for ruta_imagen in tqdm(archivos_imagen):
        imagen = leer_imagen(ruta_imagen)

        if imagen is None:
            continue

        try:
            mapa_parsing = segmentar_imagen(
                modelo_parser,
                imagen,
                dispositivo
            )

        except Exception as error:
            print(f"Error procesando {ruta_imagen.name}: {error}")
            continue

        nombre_sin_extension = ruta_imagen.stem

        cv2.imwrite(
            str(DIR_SALIDA / "parsing_index" / f"{nombre_sin_extension}.png"),
            mapa_parsing
        )

        for nombre_region, ids_clase in MAPA_REGIONES.items():
            mascara = construir_mascara_binaria(
                mapa_parsing,
                ids_clase
            )

            mascaras_region[nombre_region] = mascara

            cv2.imwrite(
                str(
                    DIR_SALIDA
                    / "binary_masks"
                    / nombre_region
                    / f"{nombre_sin_extension}.png"
                ),
                mascara
            )

        mapa_parsing_color = colorear_mapa_parsing(mapa_parsing)

        preview = crear_preview(
            imagen,
            mapa_parsing_color,
            mascaras_region["full_face"],
            mascaras_region["left_eye"],
            mascaras_region["right_eye"],
            mascaras_region["nose"],
            mascaras_region["lips"],
        )

        cv2.imwrite(
            str(DIR_SALIDA / "preview" / f"{nombre_sin_extension}.png"),
            preview
        )

    print("Proceso terminado.")
    print("Salida:", DIR_SALIDA)

if __name__ == "__main__":
    principal()