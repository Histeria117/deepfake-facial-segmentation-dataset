import cv2
import random
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline


# ============================================================
# CONFIGURACIÓN DE BATCH
# ============================================================

NUMERO_DE_BATCH = 4
TAM_BATCH = 1000
NUM_MAX_MUESTRAS = 100
TAM_IMAGEN = 512

NOMBRE_BATCH = f"batch_{NUMERO_DE_BATCH:03d}"
TIPO_DEEPFAKE = "local_inpainting"
PREFIJO_DEEPFAKE = "inp"


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

DIR_PRINCIPAL = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")

DIR_IMAGEN_BASE = DIR_PRINCIPAL / "data_raw" / "face_dataset" / "images_inpainting"
DIR_MASK_BINARIAS = DIR_PRINCIPAL / "face_parsing_output_inpainting" / "binary_masks"

DIR_FINAL_OUT = DIR_PRINCIPAL / "dataset_batches" / NOMBRE_BATCH / TIPO_DEEPFAKE
DIR_PREVIEW = DIR_PRINCIPAL / "dataset_batches" / NOMBRE_BATCH / f"preview_{TIPO_DEEPFAKE}"


# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================

REGIONES_FACIALES = ["left_eye", "right_eye", "nose", "lips", "brows"]

ID_MODELO = "runwayml/stable-diffusion-inpainting"

NUM_PASOS_INFERENCIA = 20
ESCALA_GUIA = 7.5

MODO_BAJA_VRAM = False
USAR_GPU = torch.cuda.is_available()

SEMILLA_ALEATORIA = 31

random.seed(SEMILLA_ALEATORIA)
np.random.seed(SEMILLA_ALEATORIA)
torch.manual_seed(SEMILLA_ALEATORIA)

MAX_REINTENTOS = 3

UMBRAL_MEDIA_NEGRA = 10
UMBRAL_DESVIACION_NEGRA = 8
UMBRAL_RATIO_NEGRO = 0.92


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
    mascaras = {}

    for region in ["full_face"] + REGIONES_FACIALES:
        ruta_mascara = DIR_MASK_BINARIAS / region / f"{nombre_sin_extension}.png"
        mascara = leer_mascara(ruta_mascara)

        if mascara is None:
            return None

        mascaras[region] = mascara

    return mascaras


def obtener_item_valido(ruta_imagen):
    nombre_sin_extension = ruta_imagen.stem
    mascaras = obtener_mascaras_de_imagen(nombre_sin_extension)

    if mascaras is None:
        return None

    if np.sum(mascaras["full_face"]) == 0:
        return None

    tiene_alguna_region = any(
        np.sum(mascaras[region]) > 0
        for region in REGIONES_FACIALES
    )

    if not tiene_alguna_region:
        return None

    return {
        "image_path": ruta_imagen,
        "stem": nombre_sin_extension,
        "masks": mascaras
    }


# ============================================================
# PROCESAMIENTO DE MÁSCARAS
# ============================================================

def elegir_numero_regiones():
    numero = random.randint(2, 4)
    return numero


def limpiar_mascara_binaria(mascara):
    kernel = np.ones((3, 3), np.uint8)

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    _, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)

    return mascara


def dilatar_mascara(mascara, tam_kernel=15, iteraciones=1):
    kernel = np.ones((tam_kernel, tam_kernel), np.uint8)

    mascara_dilatada = cv2.dilate(mascara, kernel, iterations=iteraciones)
    _, mascara_dilatada = cv2.threshold(
        mascara_dilatada,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return mascara_dilatada


def crear_mascara_autentica(mascara_rostro_completo, mascara_fake):
    mascara_autentica = cv2.bitwise_and(
        mascara_rostro_completo,
        cv2.bitwise_not(mascara_fake)
    )

    return mascara_autentica


# ============================================================
# CONVERSIONES ENTRE OPENCV Y PIL
# ============================================================

def convertir_cv2_a_pil_rgb(imagen_bgr):
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(imagen_rgb)


def convertir_mascara_a_pil(mascara):
    return Image.fromarray(mascara)


def convertir_pil_a_cv2_bgr(imagen_pil):
    imagen_rgb = np.array(imagen_pil)
    imagen_bgr = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)
    return imagen_bgr


# ============================================================
# VALIDACIÓN DE IMÁGENES GENERADAS
# ============================================================

def es_mayormente_negra(imagen_bgr):
    if imagen_bgr is None:
        return True

    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    media = float(np.mean(gris))
    desviacion = float(np.std(gris))
    proporcion_negra = float(np.mean(gris < 10))

    if media < UMBRAL_MEDIA_NEGRA:
        return True

    if desviacion < UMBRAL_DESVIACION_NEGRA:
        return True

    if proporcion_negra > UMBRAL_RATIO_NEGRO:
        return True

    return False


def es_generacion_invalida(imagen_bgr):
    return es_mayormente_negra(imagen_bgr)


# ============================================================
# PROMPT PARA STABLE DIFFUSION INPAINTING
# ============================================================

def construir_prompt(regiones_seleccionadas):
    nombres_regiones = {
        "left_eye": "left eye",
        "right_eye": "right eye",
        "nose": "nose",
        "lips": "lips",
        "brows": "eyebrows"
    }

    regiones_legibles = [
        nombres_regiones[region]
        for region in regiones_seleccionadas
    ]

    if len(regiones_legibles) == 1:
        texto_region = regiones_legibles[0]
    elif len(regiones_legibles) == 2:
        texto_region = f"{regiones_legibles[0]} and {regiones_legibles[1]}"
    else:
        texto_region = (
            ", ".join(regiones_legibles[:-1])
            + f", and {regiones_legibles[-1]}"
        )

    prompt = (
        f"Photorealistic facial portrait. Modify only the {texto_region} region of the face "
        f"in a realistic but clearly altered synthetic way. Preserve identity, pose, hairstyle, "
        f"background, skin tone, lighting, and all unmasked parts of the face. "
        f"Keep the result natural-looking and coherent."
    )

    prompt_negativo = (
        "blurry, low quality, black image, dark image, empty image, corrupted image, "
        "low contrast, extra eyes, extra nose, duplicated features, distorted face, "
        "deformed anatomy, cartoon, painting, unrealistic skin, changed background, bad quality"
    )

    return prompt, prompt_negativo


# ============================================================
# INICIALIZACIÓN DEL MODELO DE INPAINTING
# ============================================================

def inicializar_pipeline_inpainting():
    print("Inicializando pipeline de inpainting...")
    print("GPU disponible:", torch.cuda.is_available())

    tipo_dato = torch.float16 if torch.cuda.is_available() else torch.float32

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        ID_MODELO,
        torch_dtype=tipo_dato,
        safety_checker=None,
        requires_safety_checker=False
    )

    pipeline.enable_attention_slicing()

    try:
        pipeline.vae.enable_slicing()
    except Exception:
        pipeline.enable_vae_slicing()

    if torch.cuda.is_available():
        if MODO_BAJA_VRAM:
            print("Usando modo BAJA_VRAM con CPU/GPU offload.")
            pipeline.enable_sequential_cpu_offload()
        else:
            print("Moviendo pipeline completo a CUDA.")
            pipeline.to("cuda")
    else:
        print("No hay CUDA, usando CPU.")
        pipeline.to("cpu")

    return pipeline


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
# GENERACIÓN CON REINTENTOS
# ============================================================

def generar_inpainting_con_reintentos(
    pipeline,
    imagen_pil,
    mascara_pil,
    prompt,
    prompt_negativo,
    semilla_base,
    nombre_target
):
    for reintento in range(MAX_REINTENTOS):
        semilla_actual = semilla_base + reintento * 10000

        generador = torch.Generator(device="cpu").manual_seed(semilla_actual)

        try:
            resultado = pipeline(
                prompt=prompt,
                negative_prompt=prompt_negativo,
                image=imagen_pil,
                mask_image=mascara_pil,
                num_inference_steps=NUM_PASOS_INFERENCIA,
                guidance_scale=ESCALA_GUIA,
                generator=generador,
                height=TAM_IMAGEN,
                width=TAM_IMAGEN
            ).images[0]

        except Exception as error:
            print(
                f"Error generando inpainting para {nombre_target} "
                f"(reintento {reintento + 1}/{MAX_REINTENTOS}): {error}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

        imagen_candidata = convertir_pil_a_cv2_bgr(resultado)

        if es_generacion_invalida(imagen_candidata):
            print(
                f"Salida negra/inválida en {nombre_target} "
                f"(reintento {reintento + 1}/{MAX_REINTENTOS}, semilla={semilla_actual})"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            continue

        return imagen_candidata, semilla_actual, reintento

    return None, None, None


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def principal():
    crear_directorios()

    pipeline = inicializar_pipeline_inpainting()

    archivos_imagen = obtener_archivos_imagen()
    print("Imágenes encontradas:", len(archivos_imagen))

    items = []

    for ruta in archivos_imagen:
        item = obtener_item_valido(ruta)

        if item is not None:
            items.append(item)

    print("Imágenes válidas con máscaras:", len(items))

    if len(items) == 0:
        print("No hay imágenes válidas.")
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

        regiones_posibles = [
            region
            for region in REGIONES_FACIALES
            if np.sum(item_target["masks"][region]) > 0
        ]

        if not regiones_posibles:
            continue

        mascara_rostro_completo = item_target["masks"]["full_face"]

        num_regiones = elegir_numero_regiones()
        num_regiones = min(num_regiones, len(regiones_posibles))

        regiones_seleccionadas = random.sample(regiones_posibles, num_regiones)

        mascara_fake_union = np.zeros_like(mascara_rostro_completo)

        for region in regiones_seleccionadas:
            mascara_fake_union = cv2.bitwise_or(
                mascara_fake_union,
                item_target["masks"][region]
            )

        mascara_fake_union = limpiar_mascara_binaria(mascara_fake_union)

        if np.sum(mascara_fake_union) == 0:
            continue

        mascara_inpainting = dilatar_mascara(
            mascara_fake_union,
            tam_kernel=15,
            iteraciones=1
        )

        mascara_autentica = crear_mascara_autentica(
            mascara_rostro_completo,
            mascara_fake_union
        )

        prompt, prompt_negativo = construir_prompt(regiones_seleccionadas)

        imagen_pil = convertir_cv2_a_pil_rgb(imagen_target)
        mascara_pil = convertir_mascara_a_pil(mascara_inpainting)

        semilla_base = (
            SEMILLA_ALEATORIA
            + (NUMERO_DE_BATCH * TAM_BATCH)
            + siguiente_indice_batch
        )

        imagen_manipulada, semilla_usada, reintento_usado = generar_inpainting_con_reintentos(
            pipeline=pipeline,
            imagen_pil=imagen_pil,
            mascara_pil=mascara_pil,
            prompt=prompt,
            prompt_negativo=prompt_negativo,
            semilla_base=semilla_base,
            nombre_target=item_target["stem"]
        )

        if imagen_manipulada is None:
            print(f"No se pudo generar una imagen válida para {item_target['stem']}. Se omite.")
            continue

        id_muestra = f"{PREFIJO_DEEPFAKE}_b{NUMERO_DE_BATCH:03d}_{siguiente_indice_batch:04d}"

        ruta_imagen, ruta_autentica, ruta_fake = guardar_muestra(
            id_muestra,
            imagen_manipulada,
            mascara_autentica,
            mascara_fake_union
        )

        guardar_preview(
            id_muestra,
            imagen_manipulada,
            mascara_autentica,
            mascara_fake_union
        )

        registros.append({
            "id": id_muestra,
            "tipo": TIPO_DEEPFAKE,
            "batch": NOMBRE_BATCH,
            "NUMERO_DE_BATCH": NUMERO_DE_BATCH,
            "batch_index": siguiente_indice_batch,
            "global_index": NUMERO_DE_BATCH * TAM_BATCH + siguiente_indice_batch,
            "target_stem": item_target["stem"],
            "regions": ",".join(regiones_seleccionadas),
            "num_regions": len(regiones_seleccionadas),
            "target_image": str(item_target["image_path"]),
            "prompt": prompt,
            "seed_used": semilla_usada,
            "retry_used": reintento_usado,
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