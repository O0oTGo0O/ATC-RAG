"""Frozen F1 and F2 bridge; data-independent configuration and no target access."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party'))
from f1_frozen import atom_sets
from f2_evidence_v1 import build

def catalog(source):
 f1=atom_sets(source)[-1]
 return build(source, sorted(f1), primary_k=256)

def summary_text(cat,limit=48):
 tokens=[x['value'] for x in cat['atoms_primary'][:limit]]
 # relation text can be added downstream without changing core F2.
 return ' '.join(tokens)

def retrieval_text(source,include_evidence=True):
 if not include_evidence:return source
 return source+'\n [EVIDENCE] '+summary_text(catalog(source),limit=48)

