\# Deepfake Facial Segmentation Dataset Generator



Pipeline para generar un dataset sintético de manipulaciones faciales para entrenamiento de una red tipo U-Net con doble decodificador.



\## Tipos de manipulación



1\. FaceSwap avanzado usando InsightFace/InSwapper.

2\. Manipulación local avanzada usando face parsing, ajuste de color y seamless cloning.



\## Estructura esperada



```text

data\_raw/face\_dataset/images/

models/insightface/inswapper\_128.onnx

scripts/

