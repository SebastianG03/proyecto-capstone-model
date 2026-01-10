#!/bin/bash
# Entrenamiento YOLOv8 (Ultralytics)
# Ajusta ruta a yolov8n.pt si se desea otro backbone
MODEL=yolov8n.pt
yolo detect train model=$MODEL data=entrenamiento/data.yaml epochs=50 imgsz=640 batch=16
