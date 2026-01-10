#!/bin/bash
python entrenamiento/train_crnn.py --train_csv datasets/num_class/train.csv --val_csv datasets/num_class/val.csv --epochs 30 --out models/num_classifier.pth
