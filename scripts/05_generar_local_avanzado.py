import os
import cv2
import random
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline


BASE_DIR = Path(r"C:\ALURA ONE\PythonProject\tesis_dataset")

IMAGE_DIR = BASE_DIR / "data_raw" / "face_dataset" / "images"
MASK_BASE_DIR = BASE_DIR / "face_parsing_output" / "binary_masks"

# Para no mezclar con el método anterior
OUTPUT_DIR = BASE_DIR / "dataset_rostros_avanzado" / "local_inpainting"
PREVIEW_DIR = BASE_DIR / "preview_local_inpainting"

IMAGE_SIZE = 512
MAX_SAMPLES = 100

REGIONS = ["left_eye", "right_eye", "nose", "lips","brows"]

# Modelo de inpainting
MODEL_ID = "runwayml/stable-diffusion-inpainting"

# Configuración de generación
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5

# Para tu laptop con 4GB VRAM conviene dejar esto en True
LOW_VRAM_MODE = False
USE_GPU = torch.cuda.is_available()

RANDOM_SEED = 45
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


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


def choose_num_regions():
    numero = random.randint(2, 4)
    return numero
def clean_binary_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask
def dilate_mask(mask, kernel_size=15, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)
    _, dilated = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
    return dilated


def make_authentic_mask(full_face_mask, fake_mask):
    authentic = cv2.bitwise_and(full_face_mask, cv2.bitwise_not(fake_mask))
    return authentic


def cv2_to_pil_rgb(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def mask_to_pil(mask):
    return Image.fromarray(mask)


def pil_to_cv2_bgr(img_pil):
    img_rgb = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def build_prompt(selected_regions):
    region_names = {
        "left_eye": "left eye",
        "right_eye": "right eye",
        "nose": "nose",
        "lips": "lips",
        "brows":"eyebrows"
    }

    pretty_regions = [region_names[r] for r in selected_regions]

    if len(pretty_regions) == 1:
        target_text = pretty_regions[0]
    elif len(pretty_regions) == 2:
        target_text = f"{pretty_regions[0]} and {pretty_regions[1]}"
    else:
        target_text = ", ".join(pretty_regions[:-1]) + f", and {pretty_regions[-1]}"

    prompt = (
        f"Photorealistic facial portrait. Modify only the {target_text} region of the face "
        f"in a realistic but clearly altered synthetic way. Preserve identity, pose, hairstyle, "
        f"background, skin tone, lighting, and all unmasked parts of the face. "
        f"Keep the result natural-looking and coherent."
    )

    negative_prompt = (
        "blurry, low quality, extra eyes, extra nose, duplicated features, distorted face, "
        "deformed anatomy, cartoon, painting, unrealistic skin, changed background, bad quality"
    )

    return prompt, negative_prompt


def init_inpaint_pipeline():
    print("Inicializando pipeline de inpainting...")
    print("GPU disponible:", torch.cuda.is_available())

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype
    )

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    # Si tienes poca VRAM, esto es lo más seguro.
    if torch.cuda.is_available():
        if LOW_VRAM_MODE:
            print("Usando modo LOW_VRAM con CPU/GPU offload.")
            pipe.enable_sequential_cpu_offload()
        else:
            print("Moviendo pipeline completo a CUDA.")
            pipe.to("cuda")
    else:
        print("No hay CUDA, usando CPU.")
        pipe.to("cpu")

    return pipe


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

    pipe = init_inpaint_pipeline()

    image_files = get_image_files()
    print("Imágenes encontradas:", len(image_files))

    items = []
    for path in image_files:
        item = valid_item(path)
        if item is not None:
            items.append(item)

    print("Imágenes válidas con máscaras:", len(items))

    if len(items) == 0:
        print("No hay imágenes válidas.")
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

        possible_regions = [
            r for r in REGIONS
            if np.sum(target_item["masks"][r]) > 0
        ]

        if not possible_regions:
            continue

        full_face_mask = target_item["masks"]["full_face"]

        num_regions = choose_num_regions()
        num_regions = min(num_regions, len(possible_regions))

        selected_regions = random.sample(possible_regions, num_regions)

        # Máscara fake exacta para entrenamiento
        fake_mask_union = np.zeros_like(full_face_mask)

        for region in selected_regions:
            fake_mask_union = cv2.bitwise_or(fake_mask_union, target_item["masks"][region])

        fake_mask_union = clean_binary_mask(fake_mask_union)

        if np.sum(fake_mask_union) == 0:
            continue

        # Máscara expandida para el modelo de inpainting
        # Esto le da un poco más de contexto y suele mejorar el resultado visual
        inpaint_mask = dilate_mask(fake_mask_union, kernel_size=15, iterations=1)

        authentic_mask = make_authentic_mask(full_face_mask, fake_mask_union)

        prompt, negative_prompt = build_prompt(selected_regions)

        input_pil = cv2_to_pil_rgb(target_img)
        mask_pil = mask_to_pil(inpaint_mask)

        generator = torch.Generator(device="cpu").manual_seed(RANDOM_SEED + generated)

        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_pil,
                mask_image=mask_pil,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE
            ).images[0]
        except Exception as e:
            print(f"Error generando inpainting para {target_item['stem']}: {e}")
            continue

        manipulated_img = pil_to_cv2_bgr(result)

        sample_id = f"local_inpaint_{generated:05d}"

        img_path, auth_path, fake_path = save_sample(
            sample_id,
            manipulated_img,
            authentic_mask,
            fake_mask_union
        )

        save_preview(sample_id, manipulated_img, authentic_mask, fake_mask_union)

        records.append({
            "id": sample_id,
            "tipo": "local_inpainting",
            "regions": ",".join(selected_regions),
            "num_regions": len(selected_regions),
            "target_image": str(target_item["image_path"]),
            "prompt": prompt,
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