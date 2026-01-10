Entregables de entrenamiento

Archivos:
- prepare_yolo.py  : convierte CSV(bboxes) a estructura YOLO (images/labels)
- data.yaml        : ejemplo para Ultralytics YOLO
- train_yolo.sh    : comando de entrenamiento YOLO
- prepare_crnn.py  : convierte CSV(img,label) a train/val CSV
- train_crnn.py    : plantilla PyTorch para entrenar un clasificador (1..99)
- train_crnn.sh    : comando de ejemplo para ejecutar `train_crnn.py`

Uso rápido:
1) Preparar anotaciones CSV (ver scripts para formato)
2) `python prepare_yolo.py --csv annotations.csv --out datasets/digits --split 0.8`
3) `bash train_yolo.sh` (requiere ultralytics y modelo base)
4) `python prepare_crnn.py --csv labels.csv --out datasets/num_class --split 0.8`
5) `bash train_crnn.sh`

Notas:
- Los scripts son plantillas; ajustar rutas y parámetros según dataset.
- Instalaciones recomendadas: pytorch, torchvision, ultralytics, albumentations, pillow.
