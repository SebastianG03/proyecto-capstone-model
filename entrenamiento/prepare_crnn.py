"""
prepare_crnn.py
Convierte CSV simple (image_path,label) a train/val CSVs y extrae crops si se desea.
"""
import argparse, csv, random, shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', type=float, default=0.8)
    args=p.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows=[]
    with open(args.csv,newline='') as f:
        r=csv.reader(f)
        for row in r:
            if not row: continue
            rows.append(row)
    random.shuffle(rows)
    cut=int(len(rows)*args.split)
    with open(out/'train.csv','w',newline='') as tr, open(out/'val.csv','w',newline='') as va:
        import csv as _csv
        tw=_csv.writer(tr)
        vw=_csv.writer(va)
        for i,row in enumerate(rows):
            if i<cut: tw.writerow(row)
            else: vw.writerow(row)
    print('Done')

if __name__=='__main__':
    main()
