"""
Temporary ball detector hub.
Proporciona una interfaz unificada para varios backends de detección de balón:
- yolo (Ultralytics): modelo entrenado para balón
- hsv (simple detector por color/segmentación)
- hough (detección por círculos)

Diseñado para revisión; no reemplaza la lógica productiva.
"""
import cv2
import numpy as np
import csv

class BallDetectorHub:
    def __init__(self, specs=None):
        # specs: list of dicts con keys: model_name, model_type, model_path, params...
        self.specs = {s['model_name']: s for s in specs} if specs else {}

    def load(self, model_name):
        spec = self.specs.get(model_name)
        if not spec:
            raise KeyError(model_name)
        mtype = spec.get('model_type')
        if mtype == 'yolo':
            try:
                from ultralytics import YOLO
                spec['__model'] = YOLO(spec['model_path'])
            except Exception:
                spec['__model'] = None
        else:
            # HSV and Hough don't require heavy models
            spec['__model'] = None
        self.specs[model_name] = spec
        return spec

    def predict(self, model_name, frame):
        """Devuelve lista de detecciones: [{'x1':..,'y1':..,'x2':..,'y2':..,'score':..}]"""
        spec = self.specs.get(model_name)
        if spec is None:
            raise KeyError(model_name)
        if '__model' not in spec:
            self.load(model_name)
        mtype = spec.get('model_type')
        if mtype == 'yolo' and spec.get('__model') is not None:
            model = spec['__model']
            out = model.predict(frame, imgsz=int(spec.get('imgsz',640)), conf=float(spec.get('conf',0.25)))
            dets = []
            for o in out:
                boxes = getattr(o, 'boxes', None)
                if boxes is None: continue
                try:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                except Exception:
                    continue
                for b,c in zip(xyxy, confs):
                    dets.append({'x1':int(b[0]), 'y1':int(b[1]), 'x2':int(b[2]), 'y2':int(b[3]), 'score':float(c)})
            return dets
        if mtype == 'hsv':
            # Segmentación simple por color; espera hsv_lower,hsv_upper en spec como comma-separated
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = spec.get('hsv_lower','20,100,100')
            upper = spec.get('hsv_upper','35,255,255')
            lower = np.array([int(x) for x in lower.split(',')], dtype=np.uint8)
            upper = np.array([int(x) for x in upper.split(',')], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            # morfología
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dets = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < int(spec.get('min_area',50)): continue
                x,y,w,h = cv2.boundingRect(c)
                dets.append({'x1':x,'y1':y,'x2':x+w,'y2':y+h,'score':float(min(1.0, area/1000.0))})
            return dets
        if mtype == 'hough':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7,7), 0)
            dp = float(spec.get('dp',1.2))
            minDist = int(spec.get('minDist',30))
            param1 = int(spec.get('param1',50))
            param2 = int(spec.get('param2',30))
            minR = int(spec.get('minRadius',5))
            maxR = int(spec.get('maxRadius',50))
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minR, maxRadius=maxR)
            dets = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype('int')
                for (x,y,r) in circles:
                    dets.append({'x1':x-r,'y1':y-r,'x2':x+r,'y2':y+r,'score':1.0})
            return dets
        # fallback: no detections
        return []

def load_specs_csv(path):
    specs = []
    with open(path,newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            specs.append(row)
    return specs

if __name__=='__main__':
    print('Temporary BallDetectorHub loaded')
