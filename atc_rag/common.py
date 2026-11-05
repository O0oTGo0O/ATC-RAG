import csv, hashlib, io, json, os, random, re, zipfile
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
SPLITS={'train':'train','val':'val','test':'test'}
PH=re.compile(r'<AssertPlaceHolder>')
ASSERT_PATTERN=re.compile(r'\bassert(?:Equals|True|False|That|NotNull|Null|Same|ArrayEquals)\s*\(')
CATEGORIES=['Equals','True','That','NotNull','False','Null','ArrayEquals','Same','Other']
TOKENS=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][A-Za-z_0-9$]*|\d+(?:\.\d+)?|==|!=|<=|>=|&&|\|\||\S')

def write_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def iter_jsonl(path):
 with open(path,encoding='utf-8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def write_jsonl(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with open(path,'w',encoding='utf-8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def assertion_type(assertion):
 m=ASSERT_PATTERN.search(assertion or '')
 if not m:return 'Other'
 return m.group()[len('assert'):].split('(')[0].strip() or 'Other'
def normalize_assertion(text):
 return ' '.join(TOKENS.findall(text or ''))
def dataset_csv_name(split,dataset):
 return f'assert_{split}_{dataset.lower()}.csv'
def iter_dataset(source,split,*,dataset='old',limit=None,offset=0):
 """Dataset ZIP or CSV, columns source,target. Stable row IDs derived from source split and row number."""
 path=Path(source)
 if not path.exists():raise FileNotFoundError(path)
 if path.suffix.lower()=='.zip':
  z=zipfile.ZipFile(path)
  name=dataset_csv_name(split,dataset)
  candidates=[n for n in z.namelist() if n.endswith('/'+name) or n==name]
  if not candidates:
   z.close(); raise RuntimeError(f'Could not find {name} inside {path}. Available: {z.namelist()[:12]}')
  fh=io.TextIOWrapper(z.open(candidates[0]),encoding='utf-8-sig',newline='')
 else:
  z=None; fh=open(path,encoding='utf-8-sig',newline='')
 try:
  reader=csv.DictReader(fh)
  if not {'source','target'}.issubset(set(reader.fieldnames or [])):
   raise ValueError(f'CSV needs source,target columns; got {reader.fieldnames}')
  n=0
  for i,row in enumerate(reader):
   if i<offset:continue
   yield {'id':f'{dataset}:{split}:{i}', 'source':row['source'], 'target':row['target'],
          'source_hash':sha(row['source']), 'assertion_type':assertion_type(row['target'])}
   n+=1
   if limit is not None and n>=limit:break
 finally:
  fh.close()
  if z:z.close()
def set_seed(seed):
 random.seed(seed)
 try:
  import numpy as np
  np.random.seed(seed)
 except ImportError:pass
 try:
  import torch
  torch.manual_seed(seed)
  if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 except ImportError:pass

