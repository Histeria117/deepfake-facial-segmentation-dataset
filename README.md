# Deepfake Facial Segmentation Dataset Generator

Pipeline para generar un dataset sintético de manipulaciones faciales orientado al entrenamiento de una red neuronal convolucional basada en U-Net con doble decodificador.

La versión actual del proyecto genera dos tipos principales de manipulación facial:

1. **FaceSwap avanzado**
2. **Inpainting facial localizado**

Cada muestra generada mantiene la estructura:

```text
imagen_original/
mascara_autentica/
mascara_fake/
```

El objetivo es producir imágenes manipuladas junto con máscaras binarias que indiquen qué regiones del rostro son auténticas y cuáles fueron alteradas.

---

## Objetivo del proyecto

El objetivo de este proyecto es construir un dataset sintético para tareas de segmentación de manipulaciones faciales.

El dataset está diseñado para una arquitectura tipo **U-Net con doble decodificador**, donde una salida aprende a segmentar regiones auténticas y otra salida aprende a segmentar regiones falsas o manipuladas.

La estructura de salida esperada es:

```text
dataset_rostros_avanzado/
├── faceswap/
│   ├── imagen_original/
│   ├── mascara_autentica/
│   ├── mascara_fake/
│   └── metadata.csv
│
└── local_inpainting/
    ├── imagen_original/
    ├── mascara_autentica/
    ├── mascara_fake/
    └── metadata.csv
```

---

## Tipos de manipulación

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

### Modelo principal

El modelo usado para el intercambio facial es:

```text
inswapper_128.onnx
```

Este modelo debe colocarse manualmente en:

```text
models/insightface/inswapper_128.onnx
```

### Proceso general

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

### Máscaras en FaceSwap

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

A diferencia del método anterior basado en `seamlessClone`, esta versión ya no copia regiones desde otra imagen. En su lugar, usa un modelo generativo para reconstruir o modificar zonas específicas del rostro.

Para esto se utiliza:

```text
Stable Diffusion Inpainting
Diffusers
PyTorch CUDA
GPU NVIDIA
```

### Modelo principal

El modelo usado para inpainting es:

```text
runwayml/stable-diffusion-inpainting
```

Este modelo se descarga automáticamente desde Hugging Face la primera vez que se ejecuta el script.

Después de descargarse, queda almacenado en caché local, normalmente en:

```text
C:\Users\<usuario>\.cache\huggingface\hub
```

Por lo tanto, la primera ejecución puede tardar bastante, pero las siguientes ejecuciones ya no deberían descargar todo el modelo otra vez.

### Regiones alteradas

El script selecciona aleatoriamente una o varias regiones faciales:

```text
left_eye
right_eye
nose
lips
```

Puede generar muestras con:

```text
1 región alterada
2 regiones alteradas
3 regiones alteradas
```

Por ejemplo:

```text
nose
left_eye,lips
right_eye,nose,lips
```

### Proceso general

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

### Máscaras en inpainting

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

## Face Parsing

Antes de generar FaceSwap o Inpainting se ejecuta un proceso de segmentación facial.

El face parsing permite obtener máscaras binarias por región facial.

 un proceso de segmentación facial.

El faceLas máscaras principales usadas en este proyecto son:

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

---

## Estructura actual del repositorio

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

## Carpetas que debe crear el usuario

Antes de ejecutar el proyecto, se debe crear esta estructura:

```text
tesis_dataset/
├── data_raw/
│   └── face_dataset/
│       └── images/
│           ├── imagen_001.jpg
│           ├── imagen_002.jpg
│           └── ...
│
└── models/
    └── insightface/
        └── inswapper_128.onnx
```

La carpeta:

```text
data_raw/face_dataset/images/
```

debe contener las imágenes base de rostros.

El modelo:

```text
inswapper_128.onnx
```

debe colocarse en:

```text
models/insightface/inswapper_128.onnx
```

---

## Carpetas generadas automáticamente

Al ejecutar los scripts se generan carpetas como:

```text
face_parsing_output/
dataset_rostros_avanzado/
preview_faceswap_avanzado/
preview_local_inpainting/
comparativas_dataset/
```

Estas carpetas están ignoradas por Git porque contienen salidas generadas, imágenes, máscaras o archivos grandes.

---

## Requisitos recomendados

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

## Instalación

### 1. Crear entorno virtual

Desde la carpeta principal del proyecto:

```powershell
py -3.11 -m venv .venv311
```

### 2. Actualizar herramientas base

```powershell
.\.venv311\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

### 3. Instalar PyTorch con CUDA

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

---

## Instalación de dependencias

Instalar dependencias generales:

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

## Configuración GPU para FaceSwap

El script de FaceSwap usa:

```text
ONNX Runtime GPU
CUDAExecutionProvider
```

En Windows puede ser necesario cargar manualmente las rutas de CUDA/cuDNN dentro del script.

La versión actual de `06_generar_faceswap_avanzado.py` agrega rutas como:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
.venv311\Lib\site-packages\nvidia\cudnn\bin
.venv311\Lib\site-packages\nvidia\cublas\bin
.venv311\Lib\site-packages\nvidia\cuda_runtime\bin
```

Una ejecución correcta debe mostrar algo como:

```text
ONNX providers disponibles: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Esto confirma que FaceSwap está usando GPU.

---

## Configuración GPU para Inpainting

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

El script puede ejecutarse en modo de bajo consumo de VRAM usando:

```python
LOW_VRAM_MODE = True
```

Este modo usa offload entre CPU y GPU para evitar errores de memoria en tarjetas con poca VRAM.

Para una GPU de 4 GB, se recomienda iniciar con:

```python
LOW_VRAM_MODE = True
NUM_INFERENCE_STEPS = 15
MAX_SAMPLES = 5
```

Después se puede aumentar gradualmente.

---

## Uso

Los scripts deben ejecutarse en orden.

---

## 1. Generar máscaras con face parsing

```powershell
.\.venv311\Scripts\python.exe .\scripts\04_face_parsing_bisenet.py
```

Este script genera:

```text
face_parsing_output/
├── parsing_index/
├── binary_masks/
│   ├── left_eye/
│   ├── right_eye/
│   ├── nose/
│   ├── mouth/
│   ├── lips/
│   ├── brows/
│   └── full_face/
└── preview/
```

---

## 2. Generar inpainting facial localizado

```powershell
.\.venv311\Scripts\python.exe .\scripts\05_generar_local_avanzado.py
```

Este script genera:

```text
dataset_rostros_avanzado/
└── local_inpainting/
    ├── imagen_original/
    ├── mascara_autentica/
    ├── mascara_fake/
    └── metadata.csv
```

También genera previews en:

```text
preview_local_inpainting/
```

---

## 3. Generar FaceSwap avanzado

Antes de ejecutar este script, colocar el modelo:

```text
models/insightface/inswapper_128.onnx
```

Ejecutar:

```powershell
.\.venv311\Scripts\python.exe .\scripts\06_generar_faceswap_avanzado.py
```

Este script genera:

```text
dataset_rostros_avanzado/
└── faceswap/
    ├── imagen_original/
    ├── mascara_autentica/
    ├── mascara_fake/
    └── metadata.csv
```

También genera previews en:

```text
preview_faceswap_avanzado/
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

La comparación incluye:

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

## Formato de salida

Cada tipo de manipulación genera:

```text
imagen_original/
mascara_autentica/
mascara_fake/
metadata.csv
```

Las máscaras se guardan como imágenes binarias:

```text
0   = fondo o región no perteneciente a la clase
255 = región correspondiente
```

---

## Metadata

Cada carpeta de salida incluye un archivo:

```text
metadata.csv
```

Este archivo almacena información como:

```text
id
tipo
target_image
source_image
regions
num_regions
prompt
imagen_original
mascara_autentica
mascara_fake
```

Los campos pueden variar dependiendo del tipo de manipulación.

---

## Diferencias entre los métodos

| Método | Modelo | Tipo de manipulación | Máscara fake | Máscara auténtica | Usa GPU |
|---|---|---|---|---|---|
| FaceSwap avanzado | InsightFace + InSwapper | Reemplazo completo de identidad facial | full_face | negra | Sí, ONNX Runtime GPU |
| Inpainting localizado | Stable Diffusion Inpainting | Edición generativa parcial del rostro | regiones alteradas | full_face - mascara_fake | Sí, PyTorch CUDA |

---

## Tabla comparativa de resultados

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

Para usar esta opción, crear una carpeta:

```text
assets/
```

y colocar ahí una imagen comparativa ligera.

---

## Archivos y carpetas no incluidos en GitHub

Este repositorio no incluye:

```text
data_raw/
models/
face_parsing_output/
dataset_rostros_avanzado/
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

## Notas sobre Hugging Face

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

## Notas sobre rendimiento

FaceSwap suele ser más rápido que inpainting porque usa un modelo ONNX más ligero.

Inpainting con Stable Diffusion es más costoso porque cada imagen requiere varios pasos de difusión.

Ejemplo:

```text
10 imágenes × 20 pasos = 200 pasos de difusión
```

Para acelerar pruebas iniciales se puede usar:

```python
MAX_SAMPLES = 5
NUM_INFERENCE_STEPS = 10
```

Para mayor calidad:

```python
NUM_INFERENCE_STEPS = 20
```

---

## Problemas comunes

### PyTorch no detecta GPU

Verificar con:

```powershell
.\.venv311\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Debe mostrar:

```text
True
```

### ONNX Runtime cae a CPU

Verificar que aparezca:

```text
Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Si aparece solo:

```text
Applied providers: ['CPUExecutionProvider']
```

entonces ONNX no está usando GPU.

### Error con cuDNN o CUDA

Verificar que estén instaladas las dependencias:

```text
CUDA Toolkit
onnxruntime-gpu[cuda,cudnn]
PyTorch CUDA
```

### Primera ejecución de inpainting muy lenta

Es normal si el modelo se está descargando por primera vez.

---

## Estado actual del proyecto

- [x] Face parsing por regiones faciales
- [x] Máscara `full_face` sin cabello, cuello ni orejas
- [x] Inclusión de lentes dentro de `full_face`
- [x] FaceSwap avanzado con InsightFace/InSwapper
- [x] FaceSwap acelerado por GPU usando ONNX Runtime
- [x] Inpainting localizado con Stable Diffusion Inpainting
- [x] Inpainting acelerado por GPU usando PyTorch CUDA
- [x] Máscaras binarias `mascara_fake` y `mascara_autentica`
- [x] Metadata por muestra
- [x] Script de comparación visual

---

