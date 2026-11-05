#!/usr/bin/env python3
"""ATC Gate B: fixed two-metric, target-isolated, controlled edit evaluation.
Synthetic paired changes are derived from real OldDataSet Source and reference assertion.
The edit journal and transformed Target are not supplied to the extractor.
"""
import re, csv, json, zipfile, random, math, collections, sys, statistics, hashlib
from pathlib import Path
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parent
DATA=Path(__import__('os').environ.get('OLDDATASET_ZIP', str(ROOT/'OldDataSet.zip' if (ROOT/'OldDataSet.zip').exists() else '/mnt/data/atc_exp/OldDataSet.zip')))
ID=re.compile(r'[A-Za-z_$][A-Za-z_0-9$]*\Z')
LEX=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][A-Za-z_0-9$]*|\d+(?:\.\d+)?|\S')
KEY=set('assert assertEquals assertNotEquals assertSame assertNotSame assertTrue assertFalse assertNull assertNotNull assertArrayEquals assertThat assertThrows Assert Assertions MatcherAssert CoreMatchers org java junit hamcrest com net sun public private static protected final volatile synchronized abstract native strictfp transient class interface enum new return void int byte long float double short char boolean String Object Integer Long Double Float Boolean Byte Short Character Number if else for while do try catch finally throw throws switch case break continue default true false null this super extends implements import package var get set is test Test final override Override'.split())

def ts(s):return LEX.findall(s)
def good(s):return len(s)>=3 and bool(ID.fullmatch(s)) and s not in KEY and not s[0].isupper() and not s.startswith('atc')
def refs(src,assertion):
 st=ts(src);ta=ts(assertion)
 use=set(x for x in ta if good(x))
 counter=collections.Counter(x for x in st if good(x))
 return [w for w in use if counter[w]>0]

def positions(seq,word):return [i for i,x in enumerate(seq) if x==word]
def ctx_score(old,new,a,b):
 # symmetric source-only context similarity, with maximum across occurrences
 pa=positions(a,old);pb=positions(b,new)
 if not pa or not pb:return 0
 def window(t,i):
  ll=t[max(0,i-3):i];rr=t[i+1:i+4]
  return ll,rr
 best=0.0
 for i in pa[:6]:
  la,ra=window(a,i)
  for j in pb[:6]:
   lb,rb=window(b,j)
   # nearest context tokens carry most weight
   left=sum((la[-k]==lb[-k]) for k in range(1,min(len(la),len(lb))+1))/3
   right=sum((ra[k]==rb[k]) for k in range(min(len(ra),len(rb))))/3
   ctx=.5*(left+right)
   # Context includes neighbors on both sides, plus positional prior
   pos=1.0-abs(i/max(1,len(a))-j/max(1,len(b)))
   score=.82*ctx+.18*max(0,pos)
   best=max(best,score)
 # frequencies robust to multiple uses
 freq=min(collections.Counter(a)[old],collections.Counter(b)[new])/max(collections.Counter(a)[old],collections.Counter(b)[new])
 return .87*best+.13*freq

def align_candidates(hist,curr):
 a=ts(hist);b=ts(curr); ca=collections.Counter(x for x in a if good(x));cb=collections.Counter(x for x in b if good(x))
 olds=[x for x in ca if x not in cb];news=[x for x in cb if x not in ca]
 if not olds or not news:return []
 # unique one-to-one structural candidates; no assertion used to create alignments
 matrix=[[ctx_score(x,y,a,b) for y in news] for x in olds]
 rr,cc=linear_sum_assignment([[-z for z in row] for row in matrix])
 return sorted([(olds[i],news[j],float(matrix[i][j])) for i,j in zip(rr,cc) if matrix[i][j]>=.35],key=lambda z:-z[2])

def guided(hist,curr,ha):
 anchors=set(x for x in ts(ha) if good(x))
 return [(x,y,score) for x,y,score in align_candidates(hist,curr) if x in anchors]

def transform(original,target,seed,mode):
 rng=random.Random(seed)
 s=ts(original);a=ts(target)
 anch=refs(original,target)
 # include method names and ids in assertions where possible
 use=collections.Counter(x for x in s if good(x))
 relevant=[x for x in anch if use[x] <= 24]
 distract=[x for x in use if x not in set(anch) and use[x] <= 24]
 if len(relevant)<1 or len(distract)<1:return None
 rng.shuffle(relevant);rng.shuffle(distract)
 # tiers: 1 or 2 relevant, 1-3 irrelevant changes
 nr=2 if mode in (2,3) and len(relevant)>=2 else 1
 nd=min(len(distract), 3 if mode in (1,3) else 1)
 rnames=relevant[:nr];dnames=distract[:nd]
 if not dnames:return None
 repl={}
 for ix,x in enumerate(rnames):repl[x]='mapped_'+hashlib.sha256((str(seed)+':'+x).encode()).hexdigest()[:10]
 for ix,x in enumerate(dnames):repl[x]='mapped_'+hashlib.sha256((str(seed)+':'+x).encode()).hexdigest()[:10]
 changed_source=' '.join(repl.get(x,x) for x in s)
 changed_assert=' '.join(repl.get(x,x) for x in a)
 gold={(x,repl[x]) for x in rnames}
 return changed_source,changed_assert,sorted(gold),repl

def load(name):
 with zipfile.ZipFile(DATA) as z:
  with z.open('OldDataSet/'+name) as f:
   reader=csv.DictReader((x.decode('utf-8',errors='replace') for x in f))
   for idx,row in enumerate(reader):yield idx,row['source'],row['target']

def run_split(fname,n,seed,label):
 random.seed(seed);reservoir=[]
 # stratify on 8 principal assertion families for coverage
 fams=('assertEquals','assertTrue','assertFalse','assertNull','assertNotNull','assertThat','assertSame','assertArrayEquals')
 groups=collections.defaultdict(list)
 for i,src,tgt in load(fname):
  if 'AssertPlaceHolder' not in src:continue
  f=next((x for x in fams if x in tgt),'other')
  # deterministic hash-free sampling using seed from source row id
  if len(refs(src,tgt))==0:continue
  groups[f].append((i,src,tgt))
 # maintain type balance; save source counts
 selected=[]
 quotas={f:max(40,int(n*len(arr)/max(1,sum(map(len,groups.values()))))) for f,arr in groups.items()}
 for f,arr in groups.items():
  take=min(len(arr),quotas[f]); selected.extend(random.Random(seed+len(f)).sample(arr,take))
 random.Random(seed).shuffle(selected)
 selected=selected[:n]
 rows=[];cases=[];stats=collections.Counter();all_baseline=[]
 for index,src,tgt in selected:
  mode=(index+seed)%4
  tr=transform(src,tgt,index+seed*1000003,mode)
  if tr is None:stats['no_distractor_or_anchor']+=1;continue
  current,current_target,gold,editlog=tr
  observed=guided(src,current,tgt)
  obs={(x,y) for x,y,_ in observed}
  found=len(obs&set(tuple(z) for z in gold))
  # equal-budget baseline: highest quality source-alignment diff regardless of assertion
  baseline=set((x,y) for x,y,_ in align_candidates(src,current)[:len(obs)])
  base_hits=len(baseline&set(tuple(z) for z in gold))
  row={'split':label,'row':index,'family':next((x for x in ('assertEquals','assertTrue','assertFalse','assertNull','assertNotNull','assertThat','assertSame','assertArrayEquals') if x in tgt),'other'),
       'mode':mode,'gold_n':len(gold),'pred_n':len(obs),'tp':found,'baseline_n':len(baseline),'baseline_tp':base_hits,
       'n_source_edits':len(editlog),'gold':json.dumps(gold),'pred':json.dumps(observed)}
  rows.append(row)
  if len(cases)<20 and (found<len(gold) or len(obs)>len(gold)):
   cases.append({'index':index,'historical_source':src[:900],'historical_assertion':tgt,'current_source':current[:900],'current_target_FOR_EVAL_ONLY':current_target,'gold':gold,'pred':observed})
 def agg(rr,prefix=''):
  ng=sum(x['gold_n'] for x in rr);np=sum(x['pred_n'] for x in rr);tp=sum(x['tp'] for x in rr)
  bn=sum(x['baseline_n'] for x in rr);bt=sum(x['baseline_tp'] for x in rr)
  return {'cases':len(rr),'gold':ng,'pred':np,'tp':tp,'CP':round(tp/np,4) if np else 0,'CR':round(tp/ng,4) if ng else 0,
          'baseline_equal_budget_precision':round(bt/bn,4) if bn else 0,'baseline_equal_budget_recall':round(bt/ng,4) if ng else 0,
          'cases_with_zero_output':sum(not x['pred_n'] for x in rr)}
 summary={'overall':agg(rows),'by_family':{f:agg([r for r in rows if r['family']==f]) for f in sorted(set(r['family'] for r in rows))},'sampling':dict(stats)}
 with open(ROOT/(label+'_pairs.csv'),'w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
 (ROOT/(label+'_failures.json')).write_text(json.dumps(cases,ensure_ascii=False,indent=2))
 (ROOT/(label+'_summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 print(label,json.dumps(summary['overall']))
 return summary

if __name__=='__main__':
 s=run_split('assert_train_old.csv',2200,29,'development')
 t=run_split('assert_val_old.csv',1800,113,'confirmation')
 (ROOT/'RESULTS.json').write_text(json.dumps({'development':s,'confirmation':t},ensure_ascii=False,indent=2))

