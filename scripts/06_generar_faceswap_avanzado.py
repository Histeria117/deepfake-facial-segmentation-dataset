import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
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
except Exception as error:
    print("Error importando torch:", error)
    raise

import onnxruntime as ort

try:
    # Sin directory="", para que ONNX Runtime busque primero DLLs compatibles,
    # incluyendo las cargadas por PyTorch.
    ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
    print("ONNX Runtime DLLs precargadas.")
except Exception as error:
    print("No se pudieron precargar DLLs con ONNX Runtime:", error)

print("ONNX providers disponibles:", ort.get_available_providers())

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


# ============================================================
# CONFIGURACIÓN DE BATCH
# ============================================================

NUMERO_DE_BATCH = 3
TAM_BATCH = 1000
NUM_MAX_MUESTRAS = 1000
TAM_IMAGEN = 512

USAR_GPU = True
SEMILLA_ALEATORIA = 45

random.seed(SEMILLA_ALEATORIA)

NOMBRE_BATCH = f"batch_{NUMERO_DE_BATCH:03d}"
TIPO_DEEPFAKE = "faceswap"
PREFIJO_DEEPFAKE = "fs"


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

DIR_PRINCIPAL = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")

DIR_IMAGEN_BASE = DIR_PRINCIPAL / "data_raw" / "face_dataset" / "images_faceswap"
DIR_MASK_BINARIAS = DIR_PRINCIPAL / "face_parsing_output_faceswap" / "binary_masks"

DIR_FINAL_OUT = DIR_PRINCIPAL / "dataset_batches" / NOMBRE_BATCH / TIPO_DEEPFAKE
DIR_PREVIEW = DIR_PRINCIPAL / "dataset_batches" / NOMBRE_BATCH / f"preview_{TIPO_DEEPFAKE}"

RUTA_MODELO = DIR_PRINCIPAL / "models" / "insightface" / "inswapper_128.onnx"


# ============================================================
# DIRECTORIOS
# ============================================================

def crear_directorios():
    carpetas = [
        DIR_FINAL_OUT / "imagen_original",
        DIR_FINAL_OUT / "mascara_autentica",
        DIR_FINAL_OUT / "mascara_fake",
        DIR_PREVIEW
    ]

    for carpeta in carpetas:
        carpeta.mkdir(parents=True, exist_ok=True)


# ============================================================
# LECTURA DE IMÁGENES Y MÁSCARAS
# ============================================================

def obtener_archivos_imagen():
    extensiones_validas = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    archivos = []

    for ruta in DIR_IMAGEN_BASE.rglob("*"):
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


def leer_mascara(ruta_mascara):
    if not ruta_mascara.exists():
        return None

    mascara = cv2.imread(str(ruta_mascara), cv2.IMREAD_GRAYSCALE)

    if mascara is None:
        return None

    if mascara.shape[:2] != (TAM_IMAGEN, TAM_IMAGEN):
        mascara = cv2.resize(
            mascara,
            (TAM_IMAGEN, TAM_IMAGEN),
            interpolation=cv2.INTER_NEAREST
        )

    _, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)

    return mascara


def obtener_mascaras_de_imagen(nombre_sin_extension):
    mascara_rostro_completo = leer_mascara(
        DIR_MASK_BINARIAS / "full_face" / f"{nombre_sin_extension}.png"
    )

    if mascara_rostro_completo is None:
        return None

    if np.sum(mascara_rostro_completo) == 0:
        return None

    return {
        "full_face": mascara_rostro_completo
    }


def obtener_item_valido(ruta_imagen):
    nombre_sin_extension = ruta_imagen.stem
    mascaras = obtener_mascaras_de_imagen(nombre_sin_extension)

    if mascaras is None:
        return None

    if np.sum(mascaras["full_face"]) == 0:
        return None

    return {
        "image_path": ruta_imagen,
        "stem": nombre_sin_extension,
        "masks": mascaras
    }


# ============================================================
# CONTROL DE BATCH Y CONTINUACIÓN
# ============================================================

def obtener_siguiente_indice_batch():
    dir_imagenes = DIR_FINAL_OUT / "imagen_original"
    patron = f"{PREFIJO_DEEPFAKE}_b{NUMERO_DE_BATCH:03d}_*.png"

    archivos_existentes = list(dir_imagenes.glob(patron))

    if not archivos_existentes:
        return 0

    indices = []

    for archivo in archivos_existentes:
        try:
            indice = int(archivo.stem.split("_")[-1])
            indices.append(indice)
        except ValueError:
            pass

    if not indices:
        return 0

    return max(indices) + 1


def cargar_metadata_existente():
    """
    Carga metadata existente para poder continuar sin repetir imágenes base.
    """
    ruta_metadata = DIR_FINAL_OUT / "metadata.csv"

    if not ruta_metadata.exists():
        return [], set()

    dataframe = pd.read_csv(ruta_metadata)
    registros = dataframe.to_dict("records")

    if "target_image" in dataframe.columns:
        targets_procesados = set(dataframe["target_image"].astype(str).tolist())
    else:
        targets_procesados = set()

    return registros, targets_procesados


# ============================================================
# SELECCIÓN DE ROSTRO PRINCIPAL
# ============================================================

def calcular_area_rostro(rostro):
    bbox = rostro.bbox.astype(int)
    x1, y1, x2, y2 = bbox

    ancho = max(0, x2 - x1)
    alto = max(0, y2 - y1)

    return ancho * alto


def elegir_rostro_principal(
    rostros,
    forma_imagen,
    ratio_min_area_principal=0.06,
    ratio_min_area_significativa=0.025,
    ratio_ambiguedad=0.60
):
    """
    Selecciona el rostro principal.

    Regla:
    - Usa el rostro más grande.
    - Ignora rostros pequeños.
    - Descarta solo si hay otro rostro grande cercano en tamaño.
    """

    if rostros is None or len(rostros) == 0:
        return None, "no_faces", 0

    alto, ancho = forma_imagen[:2]
    area_imagen = alto * ancho

    rostros_ordenados = sorted(
        rostros,
        key=calcular_area_rostro,
        reverse=True
    )

    rostro_principal = rostros_ordenados[0]
    area_principal = calcular_area_rostro(rostro_principal)
    ratio_principal = area_principal / area_imagen

    if ratio_principal < ratio_min_area_principal:
        return None, f"main_face_too_small_{ratio_principal:.3f}", len(rostros)

    rostros_significativos = [
        rostro
        for rostro in rostros_ordenados
        if (calcular_area_rostro(rostro) / area_imagen) >= ratio_min_area_significativa
    ]

    if len(rostros_significativos) >= 2:
        area_segundo = calcular_area_rostro(rostros_significativos[1])
        segundo_vs_principal = area_segundo / max(area_principal, 1)

        if segundo_vs_principal >= ratio_ambiguedad:
            return None, f"ambiguous_faces_{segundo_vs_principal:.2f}", len(rostros)

    return rostro_principal, "ok", len(rostros)


# ============================================================
# MÁSCARAS
# ============================================================

def crear_mascara_para_rostro_seleccionado(
    mascara_rostro_completo,
    rostro,
    ratio_padding=0.30
):
    """
    Recorta la máscara full_face usando el bbox del rostro seleccionado.
    Esto evita que la mascara_fake tome otro rostro si hay más de una cara.
    """

    if rostro is None:
        return None

    alto, ancho = mascara_rostro_completo.shape

    x1, y1, x2, y2 = rostro.bbox.astype(int)

    ancho_bbox = max(1, x2 - x1)
    alto_bbox = max(1, y2 - y1)

    padding_x = int(ancho_bbox * ratio_padding)
    padding_y = int(alto_bbox * ratio_padding)

    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(ancho, x2 + padding_x)
    y2 = min(alto, y2 + padding_y)

    mascara_bbox = np.zeros_like(mascara_rostro_completo)
    mascara_bbox[y1:y2, x1:x2] = 255

    mascara_seleccionada = cv2.bitwise_and(
        mascara_rostro_completo,
        mascara_bbox
    )

    if np.sum(mascara_seleccionada) == 0:
        return None

    _, mascara_seleccionada = cv2.threshold(
        mascara_seleccionada,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return mascara_seleccionada


def limpiar_mascara_binaria(mascara):
    _, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    alto, ancho = mascara.shape

    mascara_flood = mascara.copy()
    mascara_auxiliar = np.zeros((alto + 2, ancho + 2), np.uint8)

    cv2.floodFill(mascara_flood, mascara_auxiliar, (0, 0), 255)

    mascara_flood_invertida = cv2.bitwise_not(mascara_flood)
    mascara_rellena = cv2.bitwise_or(mascara, mascara_flood_invertida)

    _, mascara_rellena = cv2.threshold(
        mascara_rellena,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return mascara_rellena


def crear_mascaras_fake_y_autentica(mascaras, rostro_target):
    mascara_rostro_completo = mascaras["full_face"].copy()

    mascara_fake = crear_mascara_para_rostro_seleccionado(
        mascara_rostro_completo,
        rostro_target
    )

    if mascara_fake is None:
        return None, None

    mascara_fake = limpiar_mascara_binaria(mascara_fake)

    mascara_autentica = np.zeros_like(mascara_fake)

    return mascara_fake, mascara_autentica


# ============================================================
# INICIALIZACIÓN DE MODELOS
# ============================================================

def inicializar_modelos():
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No encontré el modelo en:\n{RUTA_MODELO}\n"
            "Coloca ahí el archivo inswapper_128.onnx"
        )

    proveedores = ["CPUExecutionProvider"]
    id_contexto = -1

    if USAR_GPU:
        proveedores = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        id_contexto = 0

    print("USAR_GPU:", USAR_GPU)
    print("ONNX available providers:", ort.get_available_providers())
    print("Proveedores solicitados:", proveedores)
    print("ctx_id:", id_contexto)

    analizador_rostros = FaceAnalysis(
        name="buffalo_l",
        providers=proveedores
    )

    analizador_rostros.prepare(
        ctx_id=id_contexto,
        det_size=(320, 320),
        det_thresh=0.65
    )

    intercambiador_rostro = get_model(
        str(RUTA_MODELO),
        providers=proveedores
    )

    return analizador_rostros, intercambiador_rostro


# ============================================================
# GUARDADO DE RESULTADOS
# ============================================================

def guardar_muestra(id_muestra, imagen_manipulada, mascara_autentica, mascara_fake):
    ruta_imagen = DIR_FINAL_OUT / "imagen_original" / f"{id_muestra}.png"
    ruta_autentica = DIR_FINAL_OUT / "mascara_autentica" / f"{id_muestra}.png"
    ruta_fake = DIR_FINAL_OUT / "mascara_fake" / f"{id_muestra}.png"

    cv2.imwrite(str(ruta_imagen), imagen_manipulada)
    cv2.imwrite(str(ruta_autentica), mascara_autentica)
    cv2.imwrite(str(ruta_fake), mascara_fake)

    return ruta_imagen, ruta_autentica, ruta_fake


def guardar_preview(id_muestra, imagen_manipulada, mascara_autentica, mascara_fake):
    mascara_autentica_bgr = cv2.cvtColor(mascara_autentica, cv2.COLOR_GRAY2BGR)
    mascara_fake_bgr = cv2.cvtColor(mascara_fake, cv2.COLOR_GRAY2BGR)

    comparativa = np.hstack([
        imagen_manipulada,
        mascara_autentica_bgr,
        mascara_fake_bgr
    ])

    ruta_preview = DIR_PREVIEW / f"{id_muestra}.png"
    cv2.imwrite(str(ruta_preview), comparativa)


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def principal():
    crear_directorios()

    analizador_rostros, intercambiador_rostro = inicializar_modelos()

    archivos_imagen = obtener_archivos_imagen()
    print("Imágenes encontradas:", len(archivos_imagen))

    items = []

    for ruta in archivos_imagen:
        item = obtener_item_valido(ruta)

        if item is not None:
            items.append(item)

    print("Imágenes válidas con máscaras:", len(items))

    if len(items) < 2:
        print("Necesitas al menos 2 imágenes válidas.")
        return

    registros, targets_procesados = cargar_metadata_existente()

    siguiente_indice_batch = obtener_siguiente_indice_batch()
    generadas_en_ejecucion = 0

    print(f"Batch actual: {NOMBRE_BATCH}")
    print(f"Método: {TIPO_DEEPFAKE}")
    print(f"Muestras existentes en metadata: {len(registros)}")
    print(f"Targets ya procesados: {len(targets_procesados)}")
    print(f"Siguiente índice dentro del batch: {siguiente_indice_batch}")

    random.shuffle(items)

    for item_target in tqdm(items):
        if generadas_en_ejecucion >= NUM_MAX_MUESTRAS:
            break

        if siguiente_indice_batch >= TAM_BATCH:
            print(f"Batch {NOMBRE_BATCH} completado.")
            break

        llave_target = str(item_target["image_path"])

        if llave_target in targets_procesados:
            continue

        imagen_target = leer_imagen(item_target["image_path"])

        if imagen_target is None:
            continue

        candidatos_fuente = [
            item
            for item in items
            if item["stem"] != item_target["stem"]
        ]

        if not candidatos_fuente:
            continue

        item_fuente = random.choice(candidatos_fuente)
        imagen_fuente = leer_imagen(item_fuente["image_path"])

        if imagen_fuente is None:
            continue

        # Detectar rostro principal en target
        rostros_target = analizador_rostros.get(imagen_target)

        rostro_target, razon_target, cantidad_rostros_target = elegir_rostro_principal(
            rostros_target,
            imagen_target.shape
        )

        if rostro_target is None:
            print(
                f"Descartada target {item_target['stem']}: "
                f"{razon_target}, detectados={cantidad_rostros_target}"
            )
            continue

        # Buscar source válido
        rostro_fuente = None
        item_fuente_valido = None
        imagen_fuente_valida = None
        cantidad_rostros_fuente = 0

        random.shuffle(candidatos_fuente)

        for candidato in candidatos_fuente:
            imagen_candidata = leer_imagen(candidato["image_path"])

            if imagen_candidata is None:
                continue

            rostros_candidatos = analizador_rostros.get(imagen_candidata)

            rostro_candidato, razon_candidato, cantidad_candidatos = elegir_rostro_principal(
                rostros_candidatos,
                imagen_candidata.shape
            )

            if rostro_candidato is not None:
                rostro_fuente = rostro_candidato
                item_fuente_valido = candidato
                imagen_fuente_valida = imagen_candidata
                cantidad_rostros_fuente = cantidad_candidatos
                break

        if rostro_fuente is None:
            print(f"Descartada target {item_target['stem']}: no se encontró source válido")
            continue

        item_fuente = item_fuente_valido
        imagen_fuente = imagen_fuente_valida

        try:
            imagen_swapeada = intercambiador_rostro.get(
                imagen_target,
                rostro_target,
                rostro_fuente,
                paste_back=True
            )
        except Exception as error:
            print(
                f"Error en swap {item_target['stem']} <- "
                f"{item_fuente['stem']}: {error}"
            )
            continue

        mascara_fake, mascara_autentica = crear_mascaras_fake_y_autentica(
            item_target["masks"],
            rostro_target
        )

        if mascara_fake is None or mascara_autentica is None:
            print(
                f"Descartada {item_target['stem']}: "
                "máscara no coincide con rostro seleccionado"
            )
            continue

        if np.sum(mascara_fake) == 0:
            continue

        id_muestra = f"{PREFIJO_DEEPFAKE}_b{NUMERO_DE_BATCH:03d}_{siguiente_indice_batch:04d}"

        ruta_imagen, ruta_autentica, ruta_fake = guardar_muestra(
            id_muestra,
            imagen_swapeada,
            mascara_autentica,
            mascara_fake
        )

        guardar_preview(
            id_muestra,
            imagen_swapeada,
            mascara_autentica,
            mascara_fake
        )

        registros.append({
            "id": id_muestra,
            "tipo": TIPO_DEEPFAKE,
            "batch": NOMBRE_BATCH,
            "batch_num": NUMERO_DE_BATCH,
            "batch_index": siguiente_indice_batch,
            "global_index": NUMERO_DE_BATCH * TAM_BATCH + siguiente_indice_batch,
            "target_stem": item_target["stem"],
            "target_image": str(item_target["image_path"]),
            "source_image": str(item_fuente["image_path"]),
            "target_faces_detected": cantidad_rostros_target,
            "source_faces_detected": cantidad_rostros_fuente,
            "face_selection": "main_largest_non_ambiguous",
            "target_face_area": calcular_area_rostro(rostro_target),
            "source_face_area": calcular_area_rostro(rostro_fuente),
            "imagen_original": str(ruta_imagen),
            "mascara_autentica": str(ruta_autentica),
            "mascara_fake": str(ruta_fake)
        })

        targets_procesados.add(llave_target)

        ruta_metadata = DIR_FINAL_OUT / "metadata.csv"
        pd.DataFrame(registros).to_csv(ruta_metadata, index=False)

        siguiente_indice_batch += 1
        generadas_en_ejecucion += 1

    ruta_metadata = DIR_FINAL_OUT / "metadata.csv"
    pd.DataFrame(registros).to_csv(ruta_metadata, index=False)

    print("Muestras generadas en esta ejecución:", generadas_en_ejecucion)
    print("Total en metadata:", len(registros))
    print("Metadata:", ruta_metadata)
    print("Preview:", DIR_PREVIEW)
    print("Salida:", DIR_FINAL_OUT)


if __name__ == "__main__":
    principal()