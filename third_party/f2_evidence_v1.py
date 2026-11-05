"""F2 v1: evidence preservation and selection, NOT assertion generation.
Source-only. F1 atoms must be supplied by upstream F1. F2 never reads Target.
stdlib-only Python 3.10+. No dependency on GNC weights.
"""
from __future__ import annotations
import argparse,collections,json,re
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Iterable
TOKEN=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|(?:\d+\.\d+|\d+)|[A-Za-z_$][\w$]*|==|!=|<=|>=|&&|\|\||[^\s]')
IDENT=re.compile(r'^[A-Za-z_$][\w$]*$')
SYNTAX=set('()[]{}.,;:')
GRM={'true','false','null','0','1','2','3','10','is','equalTo','not','==','!=','>','<','>=','<=','&&','||','!','get','size','length','toString','isEmpty','contains'}

def split_source(source:str):
    x=source.find('<AssertPlaceHolder>')
    return (source[:x],source[x+len('<AssertPlaceHolder>'):]) if x>=0 else (source,'')

def order_atoms(source:str,atoms:Iterable[str])->list[dict]:
    pre,focal=split_source(source)
    ts=TOKEN.findall(pre);fs=TOKEN.findall(focal)
    s=set(ts);f=set(fs)
    last={}
    for i,v in enumerate(ts):last[v]=i
    count=collections.Counter(ts)
    ranked=[]
    for atom in set(atoms):
        if not isinstance(atom,str) or not atom:continue
        src=atom in s;fc=atom in f
        r=(last.get(atom,-1)+1)/max(1,len(ts)) if src else 0
        score=(9 if src else 0)+(2 if fc else 0)+5*r+(3 if atom in GRM else 0)
        if IDENT.fullmatch(atom) and (atom[0].islower() or atom[0]=='_'):score+=0.2
        if not src and not fc:score-=0.6
        if len(atom)>150:score-=3
        provenance=('test_prefix' if src else 'focal_method' if fc else 'F1_derived')
        category=('syntax' if atom in SYNTAX else 'operator' if atom in ('==','!=','>','<','>=','<=','&&','||','!','+','-','*','/','%','?') else 'literal' if (atom.startswith(('\"',"'")) or atom in ('true','false','null') or atom[0].isdigit()) else 'symbol')
        ranked.append({'value':atom,'score':round(score,4),'category':category,'origin':provenance,
                       'source_mentions':count.get(atom,0), 'evidence':{'in_test_prefix':src,'in_focal':fc},
                       'role':'unspecified', 'type':'unknown'})
    ranked.sort(key=lambda x:(-x['score'],x['value']))
    return ranked

def observed_relations(source:str,limit:int=80)->list[dict]:
    """Only harvest source-attested lexical edges. No inferred method call or expected value."""
    before,tail=split_source(source)
    text=before[-15000:]+"\n"+tail[:3500]
    # Observations are lexical, not full Java type-checked AST.
    pairs=[];seen=set()
    patterns=[('member_call',r'\b([A-Za-z_$][\w$]*)\s*\.\s*([A-Za-z_$][\w$]*)\s*\('),
              ('member_access',r'\b([A-Za-z_$][\w$]*)\s*\.\s*([A-Za-z_$][\w$]*)\b'),
              ('comparison',r'\b([A-Za-z_$][\w$]*)\s*(==|!=|<=|>=|<|>)\s*([A-Za-z_$][\w$]*|\d+)')]
    for kind,pat in patterns:
        for m in list(re.finditer(pat,text))[-60:]:
            k=(kind,m.group(0));
            if k in seen:continue
            seen.add(k)
            pairs.append({'kind':kind,'observed_text':m.group(0),'children':list(m.groups()),'origin':'Source_observed','type_verified':False})
    return pairs[:limit]

def build(source:str,f1_atoms:Iterable[str],primary_k:int=256):
    if primary_k<1:raise ValueError('primary_k must be >=1')
    rank=order_atoms(source,f1_atoms)
    # Grammar separators are retained losslessly but should not consume the key content budget.
    content=[a for a in rank if a['category']!='syntax']
    syntax=[a for a in rank if a['category']=='syntax']
    return {'version':'F2_Evidence_v1','semantics':'No assertion and no Target input',
            'atoms_primary':content[:primary_k], 'atoms_reserve':content[primary_k:],
            'syntax_tokens':syntax,
            'relations_observed':observed_relations(source),
            'stats':{'f1_received':len(set(f1_atoms)),'atoms_retained':len(rank),
                     'primary':min(len(content),primary_k),'reserve':max(0,len(content)-primary_k),'syntax':len(syntax)},
            'contracts':{'F3':'Use atoms and observed relations to instantiate multi-family *slot templates*, not final assertion values.',
                         'F4':'Fill slots with generated expressions; verify Java compiler and test runner externally.',
                         'F5':'Rank valid diverse candidates using observed behavior, mutation evidence and nontriviality.'}}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-file',required=True);p.add_argument('--f1-json',required=True)
    p.add_argument('--out',default='f2_evidence.json');p.add_argument('--primary-k',type=int,default=256)
    a=p.parse_args();x=json.loads(Path(a.f1_json).read_text(encoding='utf8'))
    if isinstance(x,dict) and 'elements' in x:x=x['elements']
    if not isinstance(x,list):raise ValueError('F1 input must be an array of atom strings')
    if not all(isinstance(y,str) for y in x):raise ValueError('F1 elements must be strings')
    out=build(Path(a.source_file).read_text(encoding='utf8'),x,a.primary_k)
    Path(a.out).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(out['stats']))

