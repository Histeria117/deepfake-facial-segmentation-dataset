# Deepfake Facial Segmentation Dataset Generator

Proyecto para generar un dataset sintético de manipulaciones faciales orientado al entrenamiento de una red neuronal de segmentación basada en una arquitectura tipo U-Net con doble decodificador.

El pipeline genera imágenes manipuladas y sus máscaras binarias correspondientes para distinguir regiones auténticas y regiones falsas.

Actualmente se generan dos tipos principales de manipulación facial:

```text
1. FaceSwap avanzado
2. Inpainting facial localizado
```

Cada muestra generada tiene la siguiente estructura:

```text
imagen_original/
mascara_autentica/
mascara_fake/
metadata.csv
```

> Nota: aunque la carpeta se llama `imagen_original`, dentro del dataset final representa la imagen de entrada para la red, es decir, la imagen ya manipulada.

---

# Distribución de trabajo

Dataset utilizado:

```text
FFHQ
Imágenes utilizadas: 1024x1024
1000 imágenes por carpeta / batch
```

La generación se divide en tandas de 1000 imágenes para mantener control del proceso, evitar sobrescrituras y permitir que diferentes personas generen partes del dataset de forma independiente.

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

Para cambiar de tanda, solo se debe modificar la variable:

```python
NUMERO_DE_BATCH = 0
```

Por ejemplo:

```python
NUMERO_DE_BATCH = 1
```

genera:

```text
fs_b001_0000.png
inp_b001_0000.png
```

Es importante que dos personas no usen el mismo `NUMERO_DE_BATCH` para generar datos distintos, ya que eso provocaría IDs repetidos.

---

# Objetivo del proyecto

El objetivo de este proyecto es construir un dataset sintético para tareas de segmentación de manipulaciones faciales.

El dataset está diseñado para una arquitectura tipo U-Net con doble decodificador, donde una salida aprende a segmentar regiones auténticas y otra salida aprende a segmentar regiones falsas o manipuladas.

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

# Modelos utilizados

El proyecto usa tres grupos principales de modelos:

```text
1. BiSeNet / Face Parsing
2. InsightFace + InSwapper
3. Stable Diffusion Inpainting
```

---

## 1. Face Parsing

Modelo usado:

```text
BiSeNet / Face Parsing
```

Este modelo se utiliza para segmentar el rostro en regiones semánticas.

Sirve para obtener máscaras como:

```text
left_eye
right_eye
nose
mouth
lips
brows
full_face
```

La máscara más importante es:

```text
full_face
```

En este proyecto `full_face` incluye:

```text
piel + cejas + ojos + lentes + nariz + boca + labios
```

No incluye:

```text
cabello
cuello
orejas
fondo
```

Esto se hace porque el interés del proyecto es la región facial interna.

---

## 2. FaceSwap avanzado

Modelos / herramientas usadas:

```text
InsightFace
buffalo_l
inswapper_128.onnx
ONNX Runtime GPU
CUDA
cuDNN
```

### buffalo_l

Se usa para:

```text
detectar rostros
obtener landmarks
alinear rostros
analizar identidad facial
```

### inswapper_128.onnx

Es el modelo que realiza el intercambio facial.

Debe colocarse manualmente en:

```text
models/insightface/inswapper_128.onnx
```

En FaceSwap se considera que toda la región facial interna fue manipulada:

```text
mascara_fake = full_face
mascara_autentica = máscara negra
```

---

## 3. Inpainting facial localizado

Modelo usado:

```text
runwayml/stable-diffusion-inpainting
```

Se descarga automáticamente desde Hugging Face la primera vez que se ejecuta el script.

Este modelo modifica regiones específicas del rostro, por ejemplo:

```text
left_eye
right_eye
nose
lips
brows
```

En Inpainting localizado:

```text
mascara_fake = regiones alteradas
mascara_autentica = full_face - mascara_fake
```

Ejemplo:

```text
Si se alteran nariz y labios:

mascara_fake = nariz ∪ labios
mascara_autentica = rostro completo - nariz - labios
```

---

# Estructura del repositorio

El repositorio contiene únicamente scripts y archivos de configuración.

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
├── pyproject.toml
├── uv.lock
└── .gitignore
```

No se suben al repositorio:

```text
data_raw/
models/
face_parsing_output_faceswap/
face_parsing_output_inpainting/
dataset_batches/
dataset_global/
comparativas_dataset/
.venv/
.venv311/
```

---

# Estructura esperada de carpetas

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
├── models/
│   └── insightface/
│       └── inswapper_128.onnx
│
├── scripts/
│   ├── 04_face_parsing_bisenet.py
│   ├── 05_generar_local_avanzado.py
│   ├── 06_generar_faceswap_avanzado.py
│   └── 07_plot_comparacion_dataset.py
│
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```

---

# Instalación desde cero

Esta sección explica cómo clonar el repositorio, instalar el entorno y ejecutar el proyecto.

---

## 1. Instalar Git

Primero se necesita tener Git instalado.

Verificar en PowerShell:

```powershell
git --version
```

Si no aparece versión, instalar Git desde su página oficial.

---

## 2. Clonar el repositorio

En PowerShell o Git Bash, ir a la carpeta donde se quiere guardar el proyecto.

Ejemplo:

```powershell
cd "C:\ALURA ONE\PythonProject"
```

Clonar el repositorio:

```powershell
git clone <URL_DEL_REPOSITORIO>
```

Entrar al proyecto:

```powershell
cd "C:\ALURA ONE\PythonProject\tesis_dataset"
```

---

## 3. Instalar uv

Este proyecto usa `uv` para manejar dependencias y entorno virtual.

Instalar `uv`:

```powershell
py -m pip install --upgrade uv
```

Verificar instalación:

```powershell
uv --version
```

Debe mostrar una versión de `uv`.

---

## 4. Crear entorno virtual con uv

Desde la carpeta del proyecto:

```powershell
cd "C:\ALURA ONE\PythonProject\tesis_dataset"
```

Crear entorno virtual con Python 3.11:

```powershell
uv venv --python 3.11
```

Esto crea una carpeta:

```text
.venv/
```

---

## 5. Instalar dependencias

Ejecutar:

```powershell
uv sync
```

Esto instala las dependencias indicadas en:

```text
pyproject.toml
uv.lock
```

El archivo `uv.lock` sirve para que todos instalen versiones consistentes.

---

## 6. Verificar que PyTorch detecta GPU

Ejecutar:

```powershell
uv run python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('is_available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

La salida esperada debe incluir:

```text
is_available: True
```

Y debe aparecer la GPU NVIDIA, por ejemplo:

```text
NVIDIA GeForce RTX 3050 Laptop GPU
```

---

## 7. Verificar ONNX Runtime GPU

Ejecutar:

```powershell
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

La salida esperada debe incluir:

```text
CUDAExecutionProvider
CPUExecutionProvider
```

Si solo aparece:

```text
CPUExecutionProvider
```

entonces FaceSwap no está usando GPU.

---

# Configurar PyCharm con uv

Este paso es importante para poder usar el botón verde de PyCharm.

## 1. Abrir PyCharm

Abrir el proyecto:

```text
C:\ALURA ONE\PythonProject\tesis_dataset
```

## 2. Agregar intérprete

En PyCharm:

```text
Settings / Configuración
↓
Project
↓
Python Interpreter
↓
Add Interpreter
↓
Select existing
```

Seleccionar este archivo:

```text
C:\ALURA ONE\PythonProject\tesis_dataset\.venv\Scripts\python.exe
```

Ese es el entorno creado por `uv`.

No seleccionar:

```text
.venv311
Python 3.13
Anaconda
WindowsApps
broken interpreter
```

El correcto es:

```text
tesis_dataset\.venv\Scripts\python.exe
```

---

## 3. Verificar dentro de PyCharm

Crear o ejecutar una prueba con:

```python
import torch
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
import diffusers
import insightface

print("torch:", torch.__version__)
print("cuda disponible:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("cv2:", cv2.__version__)
print("numpy:", np.__version__)
print("onnx providers:", ort.get_available_providers())
```

Si aparece:

```text
cuda disponible: True
```

y también:

```text
CUDAExecutionProvider
```

entonces la configuración está correcta.

---

# Alternativa: ejecutar desde terminal con uv

Si no se quiere usar el botón de PyCharm, los scripts se pueden ejecutar así:

```powershell
uv run python .\scripts\04_face_parsing_bisenet.py
```

```powershell
uv run python .\scripts\05_generar_local_avanzado.py
```

```powershell
uv run python .\scripts\06_generar_faceswap_avanzado.py
```

Pero si PyCharm ya está usando:

```text
tesis_dataset\.venv\Scripts\python.exe
```

entonces también se puede usar el botón verde normalmente.

---

# Plan B si uv sync falla

Si `uv sync` falla por PyTorch o CUDA, se puede instalar con un método más directo.

Desde la carpeta del proyecto:

```powershell
uv venv --python 3.11
```

Luego instalar PyTorch CUDA 12.6:

```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Después instalar dependencias restantes:

```powershell
uv pip install -r requirements.txt
```

Verificar:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

---

# Archivos importantes de configuración

## pyproject.toml

Define el proyecto y sus dependencias principales.

## uv.lock

Guarda versiones exactas o compatibles de dependencias.

Este archivo sí debe subirse a GitHub.

## requirements.txt

Se mantiene como respaldo y compatibilidad.

## .gitignore

Evita subir archivos pesados o innecesarios.

---

# Qué subir a GitHub

Sí se debe subir:

```text
scripts/
README.md
requirements.txt
pyproject.toml
uv.lock
.gitignore
```

No se debe subir:

```text
.venv/
.venv311/
data_raw/
models/
dataset_batches/
dataset_global/
face_parsing_output_faceswap/
face_parsing_output_inpainting/
comparativas_dataset/
*.onnx
*.pt
*.pth
*.ckpt
*.safetensors
```

---

# Configuración de .gitignore

El archivo `.gitignore` debe contener algo similar a:

```gitignore
# Entornos virtuales
.venv/
.venv1/
.venv311/
venv/
env/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.ipynb_checkpoints/

# PyCharm
.idea/

# Datos y modelos
data_raw/
models/
*.onnx
*.pt
*.pth
*.ckpt
*.safetensors

# Salidas del pipeline
face_parsing_output_faceswap/
face_parsing_output_inpainting/
dataset_batches/
dataset_global/
comparativas_dataset/

# Sistema
.DS_Store
Thumbs.db

# Temporales
*.log
*.tmp
```

---

# Uso del pipeline

Los scripts deben ejecutarse en orden.

---

## 1. Generar máscaras con Face Parsing

Script:

```text
scripts/04_face_parsing_bisenet.py
```

Este script usa BiSeNet para generar máscaras faciales.

Antes de ejecutarlo, verificar si se quieren generar máscaras para FaceSwap o para Inpainting.

### Para FaceSwap

Usar:

```python
DIR_ENTRADA = DIR_PRINCIPAL / "data_raw" / "face_dataset" / "images_faceswap"
DIR_SALIDA = DIR_PRINCIPAL / "face_parsing_output_faceswap"
```

### Para Inpainting

Usar:

```python
DIR_ENTRADA = DIR_PRINCIPAL / "data_raw" / "face_dataset" / "images_inpainting"
DIR_SALIDA = DIR_PRINCIPAL / "face_parsing_output_inpainting"
```

Ejecutar:

```powershell
uv run python .\scripts\04_face_parsing_bisenet.py
```

O con el botón verde de PyCharm.

Salida esperada:

```text
face_parsing_output_faceswap/
└── binary_masks/
    ├── full_face/
    ├── left_eye/
    ├── right_eye/
    ├── nose/
    ├── mouth/
    ├── lips/
    └── brows/
```

o:

```text
face_parsing_output_inpainting/
└── binary_masks/
    ├── full_face/
    ├── left_eye/
    ├── right_eye/
    ├── nose/
    ├── mouth/
    ├── lips/
    └── brows/
```

---

## 2. Generar Inpainting localizado

Script:

```text
scripts/05_generar_local_avanzado.py
```

Antes de ejecutar, configurar:

```python
NUMERO_DE_BATCH = 0
```

También se puede ajustar temporalmente:

```python
NUM_MAX_MUESTRAS = 100
```

para hacer pruebas cortas.

Ejecutar:

```powershell
uv run python .\scripts\05_generar_local_avanzado.py
```

Salida:

```text
dataset_batches/
└── batch_000/
    ├── local_inpainting/
    │   ├── imagen_original/
    │   ├── mascara_autentica/
    │   ├── mascara_fake/
    │   └── metadata.csv
    │
    └── preview_local_inpainting/
```

---

## 3. Generar FaceSwap

Script:

```text
scripts/06_generar_faceswap_avanzado.py
```

Antes de ejecutar, colocar el modelo:

```text
models/insightface/inswapper_128.onnx
```

Configurar batch:

```python
NUMERO_DE_BATCH = 0
```

Ejecutar:

```powershell
uv run python .\scripts\06_generar_faceswap_avanzado.py
```

Salida:

```text
dataset_batches/
└── batch_000/
    ├── faceswap/
    │   ├── imagen_original/
    │   ├── mascara_autentica/
    │   ├── mascara_fake/
    │   └── metadata.csv
    │
    └── preview_faceswap/
```

---

## 4. Generar comparación visual

Script:

```text
scripts/07_plot_comparacion_dataset.py
```

Ejecutar:

```powershell
uv run python .\scripts\07_plot_comparacion_dataset.py
```

Este script genera una comparación visual entre:

```text
imagen real
faceswap
máscara fake faceswap
máscara auténtica faceswap
inpainting
máscara fake inpainting
máscara auténtica inpainting
```

Salida esperada:

```text
comparativas_dataset/
```

---

# Formato de nombres

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

Cada batch y cada tipo de alteración genera su propio:

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

```text
source_image
```

aplica a FaceSwap.

```text
regions
prompt
seed_used
retry_used
```

aplican a Inpainting.

---

# Reanudar generación

Los scripts están preparados para continuar si se detienen.

Ejemplo:

Si ya existen:

```text
inp_b000_0000.png
...
inp_b000_0109.png
```

el script continuará desde:

```text
inp_b000_0110.png
```

Además, se usa `metadata.csv` para evitar repetir imágenes base ya procesadas.

---

# Control de calidad

## Inpainting

El script incluye validación para evitar imágenes negras o inválidas.

Se usan reintentos:

```python
MAX_REINTENTOS = 3
```

También se guarda:

```text
seed_used
retry_used
```

Esto permite saber qué semilla generó cada muestra.

---

## FaceSwap

El script selecciona el rostro principal y evita algunos casos ambiguos.

Guarda información como:

```text
target_faces_detected
source_faces_detected
face_selection
target_face_area
source_face_area
```

---

# Diferencias entre métodos

| Método | Modelo | Tipo de manipulación | Máscara fake | Máscara auténtica | GPU |
|---|---|---|---|---|---|
| FaceSwap | InsightFace + InSwapper | Reemplazo de identidad facial | full_face | negra | ONNX Runtime GPU |
| Inpainting | Stable Diffusion Inpainting | Edición local del rostro | regiones alteradas | full_face - mascara_fake | PyTorch CUDA |

---

# Unión de batches

Después de generar todos los batches, se pueden unir en una carpeta global.

Estructura sugerida:

```text
dataset_global/
├── imagen_original/
├── mascara_autentica/
├── mascara_fake/
└── metadata.csv
```

Gracias al formato de nombres:

```text
fs_b000_0000.png
inp_b000_0000.png
fs_b001_0000.png
inp_b001_0000.png
```

se evita sobrescribir archivos.

---

# Separación train/test

La separación recomendada es:

```text
90% train
10% test
```

Para 50,000 imágenes:

```text
45,000 train
5,000 test
```

La separación debe hacerse preferentemente por `target_stem` o ID base de FFHQ.

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

# Problemas comunes

## PyTorch no detecta GPU

Ejecutar:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Debe salir:

```text
True
```

Si sale `False`, revisar:

```text
driver NVIDIA
CUDA compatible
instalación de torch con CUDA
entorno correcto de PyCharm
```

---

## ONNX Runtime no usa GPU

Ejecutar:

```powershell
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Debe aparecer:

```text
CUDAExecutionProvider
```

---

## PyCharm usa otro entorno

Si PyCharm usa `.venv311`, `.venv1`, Anaconda o Python 3.13, puede fallar.

El intérprete correcto es:

```text
C:\ALURA ONE\PythonProject\tesis_dataset\.venv\Scripts\python.exe
```

---

## Inpainting tarda mucho

Es normal.

Stable Diffusion es pesado porque cada imagen requiere varios pasos de inferencia.

Para pruebas:

```python
NUM_MAX_MUESTRAS = 10
NUM_PASOS_INFERENCIA = 10
```

Para generación final:

```python
NUM_MAX_MUESTRAS = 1000
NUM_PASOS_INFERENCIA = 20
```

---

## Warnings repetidos en FaceSwap

InsightFace puede mostrar warnings de librerías internas. No necesariamente indican error.

Se pueden ocultar con:

```python
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
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
- [x] Configuración de entorno con uv
- [ ] Script final de unión global de batches
- [ ] Script final de separación train/test
- [ ] Integración con entrenamiento de U-Net dual decoder
