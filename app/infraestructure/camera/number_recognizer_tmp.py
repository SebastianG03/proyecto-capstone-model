"""
Temporal model hub for jersey-number recognition.
Designed for review; does NOT replace existing number_recognizer.py.
Provides a thin interface to: tesseract, easyocr, paddleocr, ultralytics YOLO.
"""
import time, csv
import numpy as np
import cv2

class PlayerNumberModelHub:
    def __init__(self, models_spec=None):
        # models_spec: dict name->spec
        self.models = {}
        if models_spec:
            for m in models_spec:
                self.models[m['model_name']] = m

    def load_model(self, model_name):
        spec = self.models.get(model_name)
        if not spec:
            raise KeyError(model_name)
        mtype = spec.get('model_type')
        if mtype == 'yolo':
            try:
                from ultralytics import YOLO
                spec['__model_obj'] = YOLO(spec['model_path'])
            except Exception as e:
                spec['__model_obj'] = None
        elif mtype == 'easyocr':
            try:
                import easyocr
                spec['__model_obj'] = easyocr.Reader(['en'], gpu=False)
            except Exception:
                spec['__model_obj'] = None
        elif mtype == 'paddleocr':
            try:
                from paddleocr import PaddleOCR
                spec['__model_obj'] = PaddleOCR(use_angle_cls=False, lang='en')
            except Exception:
                spec['__model_obj'] = None
        elif mtype == 'tesseract':
            try:
                import pytesseract
                spec['__model_obj'] = pytesseract
            except Exception:
                spec['__model_obj'] = None
        else:
            spec['__model_obj'] = None
        self.models[model_name] = spec
        return spec

    def predict(self, model_name, frame, bbox):
        # returns dict: {'number':str, 'score':float, 'method':model_name}
        spec = self.models.get(model_name)
        if spec is None:
            raise KeyError(model_name)
        if '__model_obj' not in spec:
            self.load_model(model_name)
        mobj = spec.get('__model_obj')
        mtype = spec.get('model_type')
        x1,y1,x2,y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {'number':None,'score':0.0,'method':model_name}
        # simple handlers
        if mtype == 'tesseract' and mobj is not None:
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            txt = mobj.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            txt = ''.join(ch for ch in txt if ch.isdigit())
            return {'number': txt or None, 'score': 1.0 if txt else 0.0, 'method': model_name}
        if mtype == 'easyocr' and mobj is not None:
            res = mobj.readtext(crop, detail=0)
            txt = ''.join([r for r in res if r.isdigit()])
            return {'number': txt or None, 'score': 1.0 if txt else 0.0, 'method': model_name}
        if mtype == 'paddleocr' and mobj is not None:
            res = mobj.ocr(crop, cls=False)
            txt = ''.join([line[1][0] for line in res if line and line[1][0].isdigit()])
            return {'number': txt or None, 'score': 1.0 if txt else 0.0, 'method': model_name}
        if mtype == 'yolo' and mobj is not None:
            # model should detect digits as classes 0..9; assemble by x coordinate
            out = mobj.predict(crop, imgsz=int(spec.get('imgsz',640)), conf=0.2)
            digits = []
            for o in out:
                boxes = getattr(o, 'boxes', None)
                if boxes is None: continue
                try:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls = boxes.cls.cpu().numpy()
                except Exception:
                    continue
                for b,c in zip(xyxy, cls):
                    digits.append((b[0], int(c)))
            if not digits:
                return {'number':None,'score':0.0,'method':model_name}
            digits.sort(key=lambda x: x[0])
            txt = ''.join(str(d[1]) for d in digits)
            return {'number': txt or None, 'score': 1.0, 'method': model_name}
        # fallback
        return {'number':None,'score':0.0,'method':model_name}

def load_models_from_csv(path):
    specs=[]
    with open(path,newline='') as f:
        r=csv.DictReader(f)
        for row in r:
            specs.append(row)
    return specs

if __name__=='__main__':
    # simple demo (not executed during import)
    print('Temporary PlayerNumberModelHub module loaded')
