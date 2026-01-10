"""
prepare_yolo.py
Convierte CSV con bboxes a estructura YOLO.
CSV formato: image_path,x1,y1,x2,y2,class
Salida: datasets/<name>/{images,labels}/{train,val}
"""
import argparse, os, csv, shutil, random
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', type=float, default=0.8)
    args=p.parse_args()
    out=Path(args.out)
    (out/'images'/'train').mkdir(parents=True, exist_ok=True)
    (out/'images'/'val').mkdir(parents=True, exist_ok=True)
    (out/'labels'/'train').mkdir(parents=True, exist_ok=True)
    (out/'labels'/'val').mkdir(parents=True, exist_ok=True)
    rows=[]
    with open(args.csv,newline='') as f:
        r=csv.reader(f)
        for row in r:
            if not row: continue
            rows.append(row)
    random.shuffle(rows)
    cut=int(len(rows)*args.split)
    for i,row in enumerate(rows):
        img_path,x1,y1,x2,y2,cls = row
        mode='train' if i<cut else 'val'
        img_dst=out/'images'/mode/Path(img_path).name
        shutil.copy(img_path, img_dst)
        # YOLO normalized coordinates
        from PIL import Image
        w,h=Image.open(img_path).size
        xc = (float(x1)+float(x2))/2.0/w
        yc = (float(y1)+float(y2))/2.0/h
        ww = (float(x2)-float(x1))/w
        hh = (float(y2)-float(y1))/h
        lbl = f"{cls} {xc} {yc} {ww} {hh}\n"
        with open(out/'labels'/mode/Path(img_path).with_suffix('.txt').name,'a') as lf:
            lf.write(lbl)
    print('Done')

if __name__=='__main__':
    main()
