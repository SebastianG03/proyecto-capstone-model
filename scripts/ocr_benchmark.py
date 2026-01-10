"""
Compact OCR benchmark script.
Usage: python scripts/ocr_benchmark.py --imgs DIR --csv LABELS.csv
CSV: filename,expected
Supports: tesseract (pytesseract), easyocr, paddleocr (if installed), ultralytics (YOLO) optional.
Outputs simple accuracy and avg time per method.
"""
import time, argparse, os, csv
from glob import glob
from statistics import mean

def read_csv(path):
    d={}
    with open(path, newline='') as f:
        r=csv.reader(f)
        for row in r:
            if not row: continue
            d[row[0].strip()]=row[1].strip()
    return d

def run_tesseract(imgs, labels):
    try:
        import pytesseract, cv2
    except Exception as e:
        return None
    times, ok = [], 0
    for p in imgs:
        img=cv2.imread(p,0)
        t0=time.time(); txt=pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=0123456789'); dt=time.time()-t0
        times.append(dt)
        if txt.strip().isdigit() and txt.strip()==labels.get(os.path.basename(p),''):
            ok+=1
    return {'name':'tesseract','acc': ok/len(imgs),'t': mean(times)}

def run_easyocr(imgs, labels):
    try:
        import easyocr, cv2
    except Exception:
        return None
    reader=easyocr.Reader(['en'], gpu=False)
    times, ok = [], 0
    for p in imgs:
        img=cv2.imread(p)
        t0=time.time(); res=reader.readtext(img, detail=0); dt=time.time()-t0
        times.append(dt)
        txt=''.join([r for r in res if r.isdigit()])
        if txt==labels.get(os.path.basename(p),''):
            ok+=1
    return {'name':'easyocr','acc': ok/len(imgs),'t': mean(times)}

def run_paddle(imgs, labels):
    try:
        from paddleocr import PaddleOCR
        import cv2
    except Exception:
        return None
    ocr=PaddleOCR(use_angle_cls=False, lang='en')
    times, ok = [], 0
    for p in imgs:
        img=cv2.imread(p)
        t0=time.time(); res=ocr.ocr(img, cls=False); dt=time.time()-t0
        times.append(dt)
        txt=''.join([w for line in res for w in [line[1][0]] if w.isdigit()])
        if txt==labels.get(os.path.basename(p),''):
            ok+=1
    return {'name':'paddleocr','acc': ok/len(imgs),'t': mean(times)}

def run_yolo(imgs, labels):
    try:
        from ultralytics import YOLO
        import cv2
    except Exception:
        return None
    # expects a model at app/res/models/digits.pt
    model_path='app/res/models/digits.pt'
    if not os.path.exists(model_path):
        return None
    model=YOLO(model_path)
    times, ok = [], 0
    for p in imgs:
        img=cv2.imread(p)
        t0=time.time(); res=model.predict(img, imgsz=640, conf=0.3); dt=time.time()-t0
        times.append(dt)
        txt=''
        for out in res:
            for det in out.boxes.cls.tolist() if hasattr(out.boxes,'cls') else []:
                txt+=str(int(det))
        if txt==labels.get(os.path.basename(p),''):
            ok+=1
    return {'name':'yolo_digits','acc': ok/len(imgs),'t': mean(times)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--imgs', required=True)
    p.add_argument('--csv', required=True)
    args=p.parse_args()
    labels=read_csv(args.csv)
    imgs=[f for f in sorted(glob(os.path.join(args.imgs,'*'))) if os.path.basename(f) in labels]
    if not imgs:
        print('No images found or CSV mismatch'); return
    results=[]
    for fn in (run_tesseract, run_easyocr, run_paddle, run_yolo):
        r=fn(imgs, labels)
        if r:
            results.append(r)
    for r in sorted(results, key=lambda x: -x['acc']):
        print(f"{r['name']}: acc={r['acc']:.3f}, avg_time={r['t']:.3f}s")

if __name__=='__main__':
    main()
