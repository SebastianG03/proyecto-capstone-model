"""
train_from_models_csv.py
Lee un CSV de modelos (no incluido en repo) y genera comandos de entrenamiento
o benchmarking para cada fila según su tipo. No ejecuta comandos, sólo los imprime
y los guarda en un archivo `entrenamiento/commands.txt`.

CSV esperado columnas (ejemplos):
model_name,model_type,model_path,imgsz,epochs,batch,notes
model_type: yolo, easyocr, paddleocr, classifier, crnn
"""
import csv, argparse, os

def build_command(row):
    t=row.get('model_type','').lower()
    name=row.get('model_name')
    if t=='yolo':
        imgsz=row.get('imgsz','640')
        epochs=row.get('epochs','50')
        batch=row.get('batch','16')
        # data yaml expected at entrenamiento/data.yaml or provided in notes
        return f"yolo detect train model={row.get('model_path','yolov8n.pt')} data=entrenamiento/data.yaml epochs={epochs} imgsz={imgsz} batch={batch} --name {name}"
    if t=='easyocr':
        return f"# EasyOCR: training requires custom pipeline; see EasyOCR docs for finetune. model={name}"
    if t=='paddleocr':
        return f"# PaddleOCR train: follow PaddleOCR configs. model={name}"
    if t=='classifier':
        return f"python entrenamiento/train_crnn.py --train_csv datasets/{name}/train.csv --val_csv datasets/{name}/val.csv --epochs {row.get('epochs','30')} --out models/{name}.pth"
    if t=='crnn':
        return f"# CRNN training: use PaddleOCR or custom CRNN training. model={name}"
    return f"# Unknown model type for {name}: {t}"

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='Path to models CSV (not in repo)')
    p.add_argument('--out', default='entrenamiento/commands.txt')
    args=p.parse_args()
    cmds=[]
    with open(args.csv,newline='') as f:
        r=csv.DictReader(f)
        for row in r:
            cmd=build_command(row)
            cmds.append(cmd)
    with open(args.out,'w') as f:
        for c in cmds:
            f.write(c+'\n')
    print(f'Wrote {len(cmds)} commands to {args.out}')

if __name__=='__main__':
    main()
