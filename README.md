# Deepfake Facial Segmentation Dataset Generator

Pipeline para generar un dataset sintético de manipulaciones faciales orientado al entrenamiento de una red neuronal convolucional basada en U-Net con doble decodificador.

El dataset generado separa las imágenes manipuladas y sus máscaras binarias en la siguiente estructura:

```text
imagen_original/
mascara_autentica/
mascara_fake/
```

## Objetivo

El objetivo del proyecto es generar imágenes faciales manipuladas junto con sus respectivas máscaras binarias, separando las regiones auténticas y las regiones falsas del rostro.

Este dataset está pensado para tareas de segmentación supervisada, principalmente en arquitecturas tipo U-Net con doble salida o doble decodificador.

## Tipos de manipulación generados

El pipeline contempla dos tipos principales de manipulación facial:

1. **Faceswap avanzado**
2. **Manipulación local facial avanzada**

---

## Metodología general

El proceso general del pipeline es:

```text
imágenes base de rostros
↓
face parsing
↓
generación de manipulaciones
↓
generación de máscaras binarias
↓
dataset final
```

---

## 1. Face Parsing

Primero se aplica segmentación semántica facial mediante un modelo de face parsing basado en BiSeNet.

El face parsing permite obtener máscaras por región facial, por ejemplo:

```text
piel
cejas
ojos
lentes
nariz
boca
labios
```

Para este proyecto se trabaja con un contexto facial interno, sin incluir cabello, cuello ni orejas.

La máscara principal `full_face` se define como:

```text
piel + cejas + ojos + lentes + nariz + boca + labios
```

Esta máscara se usa como base para construir las máscaras:

```text
mascara_fake
mascara_autentica
```

También se generan máscaras individuales para regiones como:

```text
left_eye
right_eye
nose
lips
mouth
brows
full_face
```

---

## 2. Manipulación local facial avanzada

La manipulación local altera una o varias regiones específicas del rostro.

Las regiones posibles son:

```text
left_eye
right_eye
nose
lips
```

El proceso consiste en:

1. Seleccionar una imagen destino.
2. Seleccionar una imagen fuente.
3. Elegir una o varias regiones faciales de forma aleatoria.
4. Extraer la región correspondiente de la imagen fuente.
5. Ajustar color y tamaño.
6. Insertar la región en la imagen destino usando `cv2.seamlessClone`.
7. Generar la máscara fake como la unión de las regiones alteradas.
8. Generar la máscara auténtica como el resto del rostro no manipulado.

La relación entre máscaras es:

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

Este tipo de manipulación permite generar casos donde solo una parte del rostro fue alterada, lo cual es útil para entrenar modelos de segmentación que no solo detecten falsificaciones globales, sino también manipulaciones parciales.

---

## 3. Faceswap avanzado

Para el faceswap se utiliza InsightFace junto con el modelo InSwapper.

Componentes principales:

```text
InsightFace
FaceAnalysis
buffalo_l
inswapper_128.onnx
onnxruntime / onnxruntime-gpu
```

El proceso consiste en:

1. Seleccionar una imagen destino.
2. Seleccionar una imagen fuente.
3. Detectar el rostro principal en ambas imágenes.
4. Aplicar el intercambio facial con InSwapper.
5. Guardar la imagen manipulada.
6. Usar la máscara `full_face` como `mascara_fake`.
7. Usar una máscara negra como `mascara_autentica`, ya que en faceswap completo toda la región facial interna se considera falsa.

La relación entre máscaras es:

```text
mascara_fake = full_face
mascara_autentica = 0
```

---

## Estructura del proyecto

```text
tesis_dataset/
├── scripts/
│   ├── 04_face_parsing_bisenet.py
│   ├── 05_generar_local_avanzado.py
│   ├── 06_generar_faceswap_avanzado.py
│   └── 07_plot_comparacion_dataset.py
│
├── requirements.txt
├── requirements-gpu.txt
├── README.md
└── .gitignore
```

Las siguientes carpetas no se incluyen en el repositorio porque contienen datos, modelos pesados o resultados generados:

```text
data_raw/
models/
face_parsing_output/
dataset_rostros_avanzado/
preview_rostros_avanzado/
preview_faceswap_avanzado/
comparativas_dataset/
```

---

## Estructura esperada para ejecutar el proyecto

Antes de ejecutar los scripts, el usuario debe crear manualmente esta estructura:

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

## Instalación

Se recomienda usar Python 3.11.

Crear un entorno virtual:

```bash
py -3.11 -m venv .venv311
```

Instalar dependencias para CPU:

```bash
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

Instalar dependencias para GPU:

```bash
.\.venv311\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

---

## Dependencias principales

El proyecto utiliza principalmente:

```text
numpy
opencv-python-headless
pandas
tqdm
matplotlib
mediapipe
torch
torchvision
facexlib
basicsr
insightface
onnxruntime
onnxruntime-gpu
```

---

## Uso

Los scripts deben ejecutarse en orden.

### 1. Generar máscaras con face parsing

```bash
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

### 2. Generar manipulaciones locales avanzadas

```bash
.\.venv311\Scripts\python.exe .\scripts\05_generar_local_avanzado.py
```

Este script genera:

```text
dataset_rostros_avanzado/
└── local/
    ├── imagen_original/
    ├── mascara_autentica/
    ├── mascara_fake/
    └── metadata.csv
```

Las manipulaciones locales pueden afectar una o varias regiones del rostro de forma aleatoria.

---

### 3. Generar faceswap avanzado

Antes de ejecutar este script, el modelo debe estar en:

```text
models/insightface/inswapper_128.onnx
```

Ejecutar:

```bash
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

---

### 4. Generar comparativa visual

```bash
.\.venv311\Scripts\python.exe .\scripts\07_plot_comparacion_dataset.py
```

Este script genera una imagen comparativa con:

```text
imagen real
faceswap
máscara fake del faceswap
máscara auténtica del faceswap
manipulación local
máscara fake local
máscara auténtica local
```

La salida se guarda en:

```text
comparativas_dataset/
```

---

## Salida final del dataset

La salida principal queda en:

```text
dataset_rostros_avanzado/
├── local/
│   ├── imagen_original/
│   ├── mascara_autentica/
│   ├── mascara_fake/
│   └── metadata.csv
│
└── faceswap/
    ├── imagen_original/
    ├── mascara_autentica/
    ├── mascara_fake/
    └── metadata.csv
```

---

## Formato de máscaras

Las máscaras se guardan como imágenes binarias:

```text
0   = fondo / región no perteneciente a esa clase
255 = región correspondiente
```

Para manipulación local:

```text
mascara_fake = región o regiones alteradas
mascara_autentica = full_face - mascara_fake
```

Para faceswap:

```text
mascara_fake = full_face
mascara_autentica = máscara negra
```

---

## Notas importantes

- No se incluyen datasets originales en este repositorio.
- No se incluyen modelos preentrenados pesados.
- No se incluye `inswapper_128.onnx`.
- No se incluyen imágenes generadas.
- El usuario debe colocar manualmente sus imágenes base en `data_raw/face_dataset/images/`.
- El usuario debe colocar manualmente el modelo `inswapper_128.onnx` en `models/insightface/`.
- Se recomienda revisar visualmente las muestras generadas antes de escalar el dataset.
- Para mejores resultados, se recomienda usar imágenes frontales o semi-frontales, con buena iluminación y rostros visibles.

---

## Estado del proyecto

- [x] Face parsing con BiSeNet
- [x] Generación de máscaras binarias por región facial
- [x] Manipulación local facial avanzada
- [x] Soporte para una o varias regiones locales alteradas
- [x] Faceswap avanzado con InsightFace/InSwapper
- [x] Generación de máscaras `mascara_fake` y `mascara_autentica`
- [x] Visualización comparativa



