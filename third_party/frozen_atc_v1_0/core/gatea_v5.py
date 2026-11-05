import re,hashlib,difflib,collections,zipfile,json,io,csv,random
from pathlib import Path
T=re.compile(r'[A-Za-z_$][\w$]*|\d+(?:\.\d+)?|"(?:\\.|[^"\\])*"|\S')
KEY=set('if else while for switch case do try catch synchronized return new throw throws class interface enum public private protected static final void test assert void super this'.split())
IDENT=re.compile(r'[A-Za-z_$][\w$]*\Z')
def tokens(s):return T.findall(s)
def prefix(s):
 t=tokens(s); ix=next((i for i,x in enumerate(t) if 'AssertPlaceHolder' in x),len(t));return t[:ix]
def calls(t):
 out=[];n=len(t)
 for k in range(3,n):
  if t[k]!='(' or t[k-2]!='.' or not IDENT.fullmatch(t[k-1]) or not IDENT.fullmatch(t[k-3]):continue
  m=t[k-1];r=t[k-3]
  if m in KEY or r in KEY:continue
  dep=0;j=k
  while j<n:
   if t[j]=='(':dep+=1
   elif t[j]==')':
    dep-=1
    if dep==0:break
   j+=1
  if dep or j>=n or j-k>72:continue
  if j+1<n and t[j+1]=='{':continue
  # call expression includes receiver . name and all args
  expr=' '.join(t[k-3:j+1]);args=' '.join(t[k+1:j]);
  # count args at top level
  sub=t[k+1:j];p=b=c=0;ar=0 if not sub else 1
  for x in sub:
   if x=='(':p+=1
   elif x==')':p-=1
   elif x=='[':b+=1
   elif x==']':b-=1
   elif x=='{':c+=1
   elif x=='}':c-=1
   elif x==',' and p==0 and b==0 and c==0:ar+=1
  out.append(dict(start=k-3,end=j+1,expr=expr,recv=r,method=m,arity=ar,args=args,prev=t[k-4] if k>=4 else '',next=t[j+1] if j+1<n else ''))
 return out

def graph(left,right):
 a=prefix(left);b=prefix(right)
 # canonical orientation independent of caller direction: hash+content secondary
 sw=(tuple(a),left)>(tuple(b),right)
 aa,bb=(b,a) if sw else (a,b)
 ca,cb=calls(aa),calls(bb)
 ia=collections.defaultdict(list);ib=collections.defaultdict(list)
 for x in ca:ia[(x['recv'],x['method'],x['arity'])].append(x)
 for x in cb:ib[(x['recv'],x['method'],x['arity'])].append(x)
 out=[]
 for key,ls in ia.items():
  rs=ib.get(key,[])
  if len(ls)!=1 or len(rs)!=1:continue
  x,y=ls[0],rs[0]
  if x['args']==y['args']:continue
  # code role independently supported by receiver/method/arity and unique occurrence
  score=difflib.SequenceMatcher(None,x['args'].split(),y['args'].split(),autojunk=False).ratio()
  out.append({'left':x['expr'],'right':y['expr'],'receiver':x['recv'],'method':x['method'],'arity':x['arity'],'left_start':x['start'],'right_start':y['start'],'argument_similarity':round(score,3),'kind':'ARGUMENT_CHANGE'})
 return {'pairs':out, 'swapped':sw,'token_count':(len(aa),len(bb))}
def forward(source1,source2):
 g=graph(source1,source2)
 if not g['swapped']:return [(x['left'],x['right']) for x in g['pairs']]
 return [(x['right'],x['left']) for x in g['pairs']]

def run():
 z=zipfile.ZipFile('/mnt/data/AssertPath_ATC_AB_Exploration_20261022.zip')
 items=[json.loads(s) for s in z.read('AB_cases.jsonl').decode().splitlines() if s.strip()]
 rows=[];pairs=[]
 for q in items:
  for rank,r in enumerate(q['references'],1):
   curr=q['current_source'];hist=r['history_source'];f=forward(hist,curr);b=forward(curr,hist)
   ok=f==[(y,x) for x,y in b]
   for x,y in f:
    pairs.append(dict(val_id=q['val_id'],train_id=r['train_id'],rank=rank,source_similarity=round(r['score'],3),historical_expression=x,current_expression=y,BC=ok))
   rows.append((q['val_id'],r['train_id'],rank,len(f),ok))
 return rows,pairs
if __name__=='__main__':
 rows,pairs=run();print('row_count',len(rows),'with_change',sum(n>0 for _,_,_,n,_ in rows),'mappings',len(pairs),'BC',sum(ok for *_,ok in rows)/len(rows));
 print('byrank',collections.Counter(x['rank'] for x in pairs));
 for p in pairs[:35]:print(p)
 Path('/mnt/data/gatea_work/v5_mappings.json').write_text(json.dumps(pairs,indent=1))

