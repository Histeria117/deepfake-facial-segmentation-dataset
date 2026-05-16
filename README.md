# Deepfake Facial Segmentation Dataset Generator

Pipeline para generar un dataset sintético de manipulaciones faciales orientado al entrenamiento de una red neuronal de segmentación basada en una arquitectura tipo U-Net con doble decodificador.

La versión actual del proyecto genera dos tipos principales de manipulación facial:

1. **FaceSwap avanzado**
2. **Inpainting facial localizado**

Cada muestra generada mantiene la estructura:

```text
imagen_original/
mascara_autentica/
mascara_fake/
metadata.csv
```

El objetivo es producir imágenes manipuladas junto con máscaras binarias que indiquen qué regiones del rostro son auténticas y cuáles fueron alteradas.

---

# Distribución de trabajo

Dataset utilizado:

```text
FFHQ https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL
Imágenes utilizadas: 1024x1024
1000 imágenes por carpeta / batch
```

La generación se divide en tandas de 1000 imágenes para mantener control del proceso

## Angel

### FaceSwap

```text
Dataset/carpeta: 25k - 36k
Cantidad aproximada: 13k imágenes
Batches: 0 - 12
```

### Inpainting local

```text
Dataset/carpeta: 0k - 11k
Cantidad aproximada: 12k imágenes
Batches: 0 - 11
```

## Cuauthli

### FaceSwap

```text
Dataset/carpeta: 37k - 49k
Cantidad aproximada: 12k imágenes
Batches: 13 - 25
```

### Inpainting local

```text
Dataset/carpeta: 12k - 24k
Cantidad aproximada: 13k imágenes
Batches: 12 - 25
```

## Regla importante de batches

Cada batch genera máximo 1000 muestras por tipo de alteración.

Ejemplo:

```text
batch_000/
├── faceswap/
└── local_inpainting/

batch_001/
├── faceswap/
└── local_inpainting/
```

Los nombres de archivo deben respetar este formato:

```text
FaceSwap:
fs_b000_0000.png
fs_b000_0001.png
...
fs_b000_0999.png

Inpainting:
inp_b000_0000.png
inp_b000_0001.png
...
inp_b000_0999.png
```

Para cambiar de tanda, solo se debe modificar:

```python
BATCH_NUM = 0
```

Por ejemplo:

```python
BATCH_NUM = 1
```

genera:

```text
fs_b001_0000.png
inp_b001_0000.png
```

Es importante que dos personas no usen el mismo `BATCH_NUM` para generar datos distintos, ya que eso provocaría IDs repetidos.

---

# Objetivo del proyecto

El objetivo de este proyecto es construir un dataset sintético para tareas de segmentación de manipulaciones faciales.

El dataset está diseñado para una arquitectura tipo **U-Net con doble decodificador**, donde una salida aprende a segmentar regiones auténticas y otra salida aprende a segmentar regiones falsas o manipuladas.

El dataset final busca contener aproximadamente:

```text
25,000 imágenes FaceSwap
25,000 imágenes Inpainting localizado
```

Total aproximado:

```text
50,000 imágenes manipuladas
```

Posteriormente, el dataset puede dividirse en:

```text
90% entrenamiento
10% prueba
```

Ejemplo aproximado:

```text
45,000 imágenes para entrenamiento
5,000 imágenes para prueba
```

La separación final debe hacerse cuidando que una misma imagen base de FFHQ no quede al mismo tiempo en entrenamiento y prueba.

---

# Tipos de manipulación

## 1. FaceSwap avanzado

El primer tipo de manipulación consiste en reemplazar la identidad facial de una imagen destino usando una imagen fuente.

Para esto se utiliza:

```text
InsightFace
FaceAnalysis
buffalo_l
inswapper_128.onnx
ONNX Runtime GPU
CUDA
cuDNN
```

## Modelo principal

El modelo usado para el intercambio facial es:

```text
inswapper_128.onnx
```

Este modelo debe colocarse manualmente en:

```text
models/insightface/inswapper_128.onnx
```

## Proceso general

El pipeline de FaceSwap realiza los siguientes pasos:

```text
imagen target
+
imagen source
↓
detección facial con InsightFace
↓
selección del rostro principal
↓
aplicación de InSwapper
↓
imagen con rostro intercambiado
↓
generación de máscara fake y máscara auténtica
```

## Máscaras en FaceSwap

Para FaceSwap, toda la región facial interna se considera manipulada.

```text
mascara_fake = full_face
mascara_autentica = máscara negra
```

La máscara `full_face` se obtiene previamente mediante face parsing e incluye:

```text
piel + cejas + ojos + lentes + nariz + boca + labios
```

No se incluyen:

```text
cabello
cuello
orejas
fondo
```

Esto permite mantener el enfoque en el rostro visible y relevante para la segmentación.

---

## 2. Inpainting facial localizado

El segundo tipo de manipulación consiste en alterar regiones específicas del rostro usando un modelo generativo de inpainting.

A diferencia de un método basado únicamente en copiado de regiones o `seamlessClone`, esta versión usa un modelo generativo para reconstruir o modificar zonas específicas del rostro.

Para esto se utiliza:

```text
Stable Diffusion Inpainting
Diffusers
PyTorch CUDA
GPU NVIDIA
```

## Modelo principal

El modelo usado para inpainting es:

```text
runwayml/stable-diffusion-inpainting
```

Este modelo se descarga automáticamente desde Hugging Face la primera vez que se ejecuta el script.

Después de descargarse, queda almacenado en caché local, normalmente en:

```text
C:\Users\<usuario>\.cache\huggingface\hub
```

La primera ejecución puede tardar bastante, pero las siguientes ejecuciones ya no deberían descargar el modelo completo otra vez.

## Regiones alteradas

El script selecciona aleatoriamente varias regiones faciales:

```text
left_eye
right_eye
nose
lips
brows
```

Actualmente se seleccionan entre 2 y 4 regiones por imagen.

Ejemplos:

```text
nose,lips
left_eye,right_eye,brows
right_eye,nose,lips,brows
```

## Proceso general

El pipeline de inpainting realiza los siguientes pasos:

```text
imagen original
↓
máscaras de face parsing
↓
selección aleatoria de regiones faciales
↓
unión de máscaras seleccionadas
↓
inpainting generativo con Stable Diffusion
↓
imagen manipulada
↓
generación de máscara fake y máscara auténtica
```

## Máscaras en inpainting

Para inpainting localizado:

```text
mascara_fake = unión de regiones alteradas
mascara_autentica = full_face - mascara_fake
```

Ejemplo:

```text
Si se alteran nariz y labios:

mascara_fake = nariz ∪ labios
mascara_autentica = full_face - mascara_fake
```

---

# Face Parsing

Antes de generar FaceSwap o Inpainting se ejecuta un proceso de segmentación facial.

El face parsing permite obtener máscaras binarias por región facial.

Las máscaras principales usadas en este proyecto son:

```text
left_eye
right_eye
nose
mouth
lips
brows
full_face
```

La máscara `full_face` representa la región facial interna y está definida como:

```text
piel + cejas + ojos + lentes + nariz + boca + labios
```

Esta máscara es fundamental porque se usa para construir:

```text
mascara_fake
mascara_autentica
```

Actualmente se manejan carpetas separadas para cada tipo de alteración:

```text
face_parsing_output_faceswap/
face_parsing_output_inpainting/
```

Esto permite trabajar con tandas separadas sin mezclar máscaras.

---

# Estructura actual del repositorio

El repositorio solo contiene los scripts y archivos necesarios para ejecutar el pipeline.

```text
tesis_dataset/
├── scripts/
│   ├── 04_face_parsing_bisenet.py
│   ├── 05_generar_local_avanzado.py
│   ├── 06_generar_faceswap_avanzado.py
│   └── 07_plot_comparacion_dataset.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

Las carpetas con datos, modelos y resultados generados no se suben al repositorio.

---

# Estructura de carpetas de entrada

El proyecto espera una estructura similar a esta:

```text
tesis_dataset/
├── data_raw/
│   └── face_dataset/
│       ├── images_faceswap/
│       │   ├── 00000.png
│       │   ├── 00001.png
│       │   └── ...
│       │
│       └── images_inpainting/
│           ├── 00000.png
│           ├── 00001.png
│           └── ...
│
└── models/
    └── insightface/
        └── inswapper_128.onnx
```

## Imágenes para FaceSwap

```text
data_raw/face_dataset/images_faceswap/
```

## Imágenes para Inpainting

```text
data_raw/face_dataset/images_inpainting/
```

## Modelo InSwapper

```text
models/insightface/inswapper_128.onnx
```

---

# Estructura de carpetas generadas

La salida se organiza por batches:

```text
dataset_batches/
├── batch_000/
│   ├── faceswap/
│   │   ├── imagen_original/
│   │   ├── mascara_autentica/
│   │   ├── mascara_fake/
│   │   └── metadata.csv
│   │
│   ├── preview_faceswap/
│   │
│   ├── local_inpainting/
│   │   ├── imagen_original/
│   │   ├── mascara_autentica/
│   │   ├── mascara_fake/
│   │   └── metadata.csv
│   │
│   └── preview_local_inpainting/
│
├── batch_001/
├── batch_002/
└── ...
```

---

# Formato de nombres

El formato de nombres está diseñado para evitar sobrescrituras al unir batches.

## FaceSwap

```text
fs_b000_0000.png
fs_b000_0001.png
...
fs_b000_0999.png
```

## Inpainting

```text
inp_b000_0000.png
inp_b000_0001.png
...
inp_b000_0999.png
```

Donde:

```text
fs  = FaceSwap
inp = Inpainting
b000 = batch 000
0000 = índice dentro del batch
```

Ejemplo:

```text
fs_b012_0457.png
```

significa:

```text
Método: FaceSwap
Batch: 12
Índice dentro del batch: 457
```

---

# Metadata

Cada batch y cada tipo de alteración genera su propio archivo:

```text
metadata.csv
```

Campos principales:

```text
id
tipo
batch
batch_num
batch_index
global_index
target_stem
target_image
source_image
regions
num_regions
prompt
seed_used
retry_used
imagen_original
mascara_autentica
mascara_fake
```

No todos los campos aparecen en ambos métodos.

Por ejemplo:

- `source_image` aplica a FaceSwap.
- `regions`, `prompt`, `seed_used` y `retry_used` aplican a Inpainting.

---

# Requisitos recomendados

Este proyecto fue probado en Windows con:

```text
Python 3.11
NVIDIA RTX 3050 Laptop GPU
CUDA Toolkit 12.6
PyTorch con CUDA 12.6
ONNX Runtime GPU
cuDNN 9
```

También puede ejecutarse parcialmente en CPU, pero el rendimiento será mucho menor, especialmente para inpainting.

---

# Instalación

## 1. Crear entorno virtual

Desde la carpeta principal del proyecto:

```powershell
py -3.11 -m venv .venv311
```

## 2. Actualizar herramientas base

```powershell
.\.venv311\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

## 3. Instalar PyTorch con CUDA

Para GPU NVIDIA con CUDA 12.6:

```powershell
.\.venv311\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verificar que PyTorch detecte la GPU:

```powershell
.\.venv311\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('is_available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

La salida esperada debe incluir:

```text
is_available: True
```

## 4. Instalar dependencias generales

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

El archivo `requirements.txt` debe incluir dependencias como:

```text
numpy==1.26.4
opencv-python==4.10.0.84
pandas
tqdm
matplotlib
pillow
facexlib
basicsr
insightface==0.7.3
onnxruntime-gpu[cuda,cudnn]
diffusers
transformers
accelerate
safetensors
huggingface_hub
```

---

# Configuración GPU para FaceSwap

El script de FaceSwap usa:

```text
ONNX Runtime GPU
CUDAExecutionProvider
```

Una ejecución correcta debe mostrar:

```text
ONNX providers disponibles: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Esto confirma que FaceSwap está usando GPU.

---

# Configuración GPU para Inpainting

El script de inpainting usa:

```text
PyTorch CUDA
Diffusers
Stable Diffusion Inpainting
```

Una ejecución correcta debe mostrar:

```text
GPU disponible: True
```

Para una GPU de 4 GB, se recomienda iniciar con:

```python
LOW_VRAM_MODE = True
NUM_INFERENCE_STEPS = 15
MAX_NEW_SAMPLES = 100
```

Si la GPU soporta el pipeline completo, puede usarse:

```python
LOW_VRAM_MODE = False
NUM_INFERENCE_STEPS = 20
```

---

# Uso

Los scripts deben ejecutarse en orden.

---

## 1. Generar máscaras con face parsing

```powershell
.\.venv311\Scripts\python.exe .\scripts\04_face_parsing_bisenet.py
```

Este script genera máscaras en carpetas como:

```text
face_parsing_output_faceswap/
face_parsing_output_inpainting/
```

La estructura interna esperada es:

```text
binary_masks/
├── left_eye/
├── right_eye/
├── nose/
├── mouth/
├── lips/
├── brows/
└── full_face/
```

---

## 2. Generar inpainting facial localizado

Antes de ejecutar, configurar el batch:

```python
BATCH_NUM = 0
```

Ejecutar:

```powershell
.\.venv311\Scripts\python.exe .\scripts\05_generar_local_avanzado.py
```

Este script genera:

```text
dataset_batches/
└── batch_000/
    └── local_inpainting/
        ├── imagen_original/
        ├── mascara_autentica/
        ├── mascara_fake/
        └── metadata.csv
```

También genera previews en:

```text
dataset_batches/batch_000/preview_local_inpainting/
```

---

## 3. Generar FaceSwap avanzado

Antes de ejecutar, configurar el batch:

```python
BATCH_NUM = 0
```

También se debe colocar el modelo:

```text
models/insightface/inswapper_128.onnx
```

Ejecutar:

```powershell
.\.venv311\Scripts\python.exe .\scripts\06_generar_faceswap_avanzado.py
```

Este script genera:

```text
dataset_batches/
└── batch_000/
    └── faceswap/
        ├── imagen_original/
        ├── mascara_autentica/
        ├── mascara_fake/
        └── metadata.csv
```

También genera previews en:

```text
dataset_batches/batch_000/preview_faceswap/
```

---

## 4. Generar comparación visual

```powershell
.\.venv311\Scripts\python.exe .\scripts\07_plot_comparacion_dataset.py
```

Este script genera una imagen comparativa en:

```text
comparativas_dataset/
```

La comparación puede incluir:

```text
imagen real
faceswap
máscara fake del faceswap
máscara auténtica del faceswap
inpainting local
máscara fake del inpainting
máscara auténtica del inpainting
```

---

# Unión de batches

Después de generar todos los batches, estos pueden unirse en una carpeta global.

Estructura final sugerida:

```text
dataset_global/
├── imagen_original/
├── mascara_autentica/
├── mascara_fake/
└── metadata.csv
```

La unión debe hacerse por código, copiando y conservando los nombres únicos:

```text
fs_b000_0000.png
inp_b000_0000.png
fs_b001_0000.png
inp_b001_0000.png
```

Esto evita sobrescrituras.

---

# Separación train/test

La separación recomendada es:

```text
90% train
10% test
```

Para un dataset de 50,000 imágenes:

```text
45,000 train
5,000 test
```

La separación debe hacerse preferentemente por `target_stem` o por ID base de FFHQ.

Ejemplo correcto:

```text
ffhq_000123 faceswap     -> train
ffhq_000123 inpainting   -> train
```

Ejemplo incorrecto:

```text
ffhq_000123 faceswap     -> train
ffhq_000123 inpainting   -> test
```

Esto evita fuga de información entre entrenamiento y prueba.

---

# Control de calidad

## Inpainting

El script de inpainting incluye validación para evitar imágenes negras o inválidas.

Se usan reintentos automáticos:

```python
MAX_RETRIES = 3
```

También se guardan en metadata:

```text
seed_used
retry_used
```

Esto permite saber qué seed generó cada muestra.

## FaceSwap

El script de FaceSwap selecciona el rostro principal y evita casos ambiguos cuando hay varios rostros significativos.

También guarda información como:

```text
target_faces_detected
source_faces_detected
face_selection
target_face_area
source_face_area
```

---

# Diferencias entre métodos

| Método | Modelo | Tipo de manipulación | Máscara fake | Máscara auténtica | Usa GPU |
|---|---|---|---|---|---|
| FaceSwap avanzado | InsightFace + InSwapper | Reemplazo completo de identidad facial | full_face | negra | Sí, ONNX Runtime GPU |
| Inpainting localizado | Stable Diffusion Inpainting | Edición generativa parcial del rostro | regiones alteradas | full_face - mascara_fake | Sí, PyTorch CUDA |

---

# Tabla comparativa de resultados

En esta sección se puede colocar una tabla visual con ejemplos generados por ambos métodos.

Ejemplo de estructura sugerida:

| Real | FaceSwap | Máscara Fake FaceSwap | Máscara Auth FaceSwap | Inpainting | Máscara Fake Inpainting | Máscara Auth Inpainting |
|---|---|---|---|---|---|---|
| imagen real | imagen faceswap | máscara fake | máscara auténtica | imagen inpainting | máscara fake | máscara auténtica |

También se puede insertar una imagen generada por el script `07_plot_comparacion_dataset.py`.

Ejemplo:

```markdown
![Comparación de resultados](assets/comparacion_real_faceswap_local_mascaras.png)
```

---

# Archivos y carpetas no incluidos en GitHub

Este repositorio no incluye:

```text
data_raw/
models/
face_parsing_output_faceswap/
face_parsing_output_inpainting/
dataset_batches/
dataset_global/
preview_faceswap_avanzado/
preview_local_inpainting/
comparativas_dataset/
```

Tampoco incluye archivos pesados como:

```text
*.onnx
*.pt
*.pth
*.ckpt
*.safetensors
```

Esto se controla mediante `.gitignore`.

---

# Notas sobre Hugging Face

El modelo de inpainting se descarga desde Hugging Face la primera vez que se ejecuta el script.

Durante la primera descarga puede aparecer:

```text
Fetching files
```

Esto es normal.

También puede aparecer una advertencia sobre solicitudes no autenticadas:

```text
Warning: You are sending unauthenticated requests to the HF Hub
```

No es un error. Solo indica que la descarga puede ser más lenta o tener límites.

---

# Notas sobre rendimiento

FaceSwap suele ser más rápido que inpainting porque usa un modelo ONNX más ligero.

Inpainting con Stable Diffusion es más costoso porque cada imagen requiere varios pasos de difusión.

Ejemplo:

```text
10 imágenes × 20 pasos = 200 pasos de difusión
```

Para acelerar pruebas iniciales se puede usar:

```python
MAX_NEW_SAMPLES = 100
NUM_INFERENCE_STEPS = 10
```

Para mayor calidad:

```python
NUM_INFERENCE_STEPS = 20
```

---

# Problemas comunes

## PyTorch no detecta GPU

Verificar con:

```powershell
.\.venv311\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Debe mostrar:

```text
True
```

## ONNX Runtime cae a CPU

Verificar que aparezca:

```text
Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Si aparece solo:

```text
Applied providers: ['CPUExecutionProvider']
```

entonces ONNX no está usando GPU.

## Primera ejecución de inpainting muy lenta

Es normal si el modelo se está descargando por primera vez.

## Warnings repetidos en FaceSwap

InsightFace puede imprimir warnings de librerías internas durante el alineamiento facial. No necesariamente indican error.

Pueden ocultarse con:

```python
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="insightface.*"
)
```

---

# Estado actual del proyecto

- [x] Face parsing por regiones faciales
- [x] Máscara `full_face` sin cabello, cuello ni orejas
- [x] Inclusión de lentes dentro de `full_face`
- [x] FaceSwap avanzado con InsightFace/InSwapper
- [x] FaceSwap acelerado por GPU usando ONNX Runtime
- [x] Inpainting localizado con Stable Diffusion Inpainting
- [x] Inpainting acelerado por GPU usando PyTorch CUDA
- [x] Máscaras binarias `mascara_fake` y `mascara_autentica`
- [x] Metadata por muestra
- [x] Generación por batches
- [x] Nombres únicos por batch
- [x] Validación contra imágenes negras en inpainting
- [x] Script de comparación visual
- [ ] Script final de unión global de batches
- [ ] Script final de separación train/test
- [ ] Integración con entrenamiento de U-Net dual decoder

---

# Advertencia ética

Este proyecto está diseñado con fines académicos y de investigación en detección y segmentación de manipulaciones faciales.

El uso de técnicas de FaceSwap, inpainting o manipulación facial debe realizarse de manera responsable, respetando la privacidad, consentimiento y derechos de imagen de las personas involucradas.

No se recomienda usar este proyecto para generar contenido engañoso, suplantación de identidad o distribución de imágenes manipuladas sin consentimiento.