"""Fixed sequence serialization, with bounded per-reference token budgets."""
import sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party'/'frozen_atc_v1_0'))
from atc import build_change_cues
from .features import catalog,summary_text

TOKEN=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][A-Za-z_0-9$]*|\d+(?:\.\d+)?|\S')

def plain_diff(hist,curr,max_pairs=12):
 """Plain lexical diff control with no historical assertion guidance. Deterministic stdlib baseline."""
 from difflib import SequenceMatcher
 a=TOKEN.findall(hist);b=TOKEN.findall(curr)
 diffs=[]
 for op,al,ar,bl,br in SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
  if op=='equal':continue
  diffs.append({'cue_type':op.upper(),'historical_expression':' '.join(a[al:ar][:20]),
                'current_expression':' '.join(b[bl:br][:20]),'status':'DIFF_ONLY'})
  if len(diffs)>=max_pairs:break
 return diffs

def cue_text(cue):
 return (cue['cue_type']+': '+cue['historical_expression']+' => '+cue['current_expression'])

def bound_source_for_atc(source,prefix_budget=120,focal_budget=56):
 """Deterministic ATC input window; frozen algorithm operates unchanged on this bounded source.
 Uses recent test prefix and the beginning of focal context; document this approximation.
 """
 from third_party.f2_evidence_v1 import TOKEN
 if '<AssertPlaceHolder>' in source:
  pre,tail=source.split('<AssertPlaceHolder>',1)
  return ' '.join(TOKEN.findall(pre)[-prefix_budget:])+' <AssertPlaceHolder> '+ ' '.join(TOKEN.findall(tail)[:focal_budget])
 return ' '.join(TOKEN.findall(source)[-prefix_budget:])

def prepare_example(record,refs,*,variant='full',k=3,keep_reserve=False,atc_prefix=120,atc_focal=56):
 """Variants for RQ2: full/no_f2/no_atc/diff_only/single_ref/code_only."""
 if variant=='single_ref':k=1
 if variant=='code_only':refs=[]
 cat=None if variant in ('no_f2','code_only') else catalog(record['source'])
 packed=[]
 for ref in refs[:k]:
  if variant=='no_atc' or variant=='code_only':cues=[];reserve=[]
  elif variant=='diff_only':cues=plain_diff(ref['source'],record['source']);reserve=[]
  else:
   bound_h=bound_source_for_atc(ref['source'],atc_prefix,atc_focal)
   bound_c=bound_source_for_atc(record['source'],atc_prefix,atc_focal)
   atc=build_change_cues(bound_h,ref['assertion'],bound_c,historical_ref_id=ref['ref_id'])
   cues=atc['change_cues'];reserve=atc['reserve_candidates'] if keep_reserve else []
  packed.append({'ref_id':ref['ref_id'],'historical_source':ref['source'],'historical_assertion':ref['assertion'],
                 'change_cues':[cue_text(c) for c in cues],
                 'reserve_candidates':[cue_text(c) for c in reserve],
                 'retrieval_score':ref.get('score')})
 return {'id':record['id'],'source':record['source'],
         'evidence_text': '' if cat is None else summary_text(cat),
         'references':packed,'target':record['target'],'assertion_type':record['assertion_type'],
         'variant':variant,'atc_prefix_budget':atc_prefix,'atc_focal_budget':atc_focal}

def token_clip(tokenizer,txt,limit):
 if limit<=0:return ''
 ids=tokenizer.encode(txt,add_special_tokens=False,truncation=True,max_length=limit)
 return tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)

def serialize(row,tokenizer,max_source_length=512):
 """Preserve each retrieved reference in the encoder sequence before final safety truncation."""
 refs=row.get('references',[])
 if not refs:
  text='[CURRENT_SOURCE] '+token_clip(tokenizer,row['source'],max_source_length-12)
  return text
 n=len(refs)
 current_budget=max(64,int(max_source_length*.47))
 ev_budget=max(16,int(max_source_length*.09))
 total_ref_budget=max(20,max_source_length-current_budget-ev_budget-18-12*n)
 each=max(14,total_ref_budget//n)
 parts=['[CURRENT_SOURCE] '+token_clip(tokenizer,row['source'],current_budget)]
 if row.get('evidence_text'):
  parts.append('[EVIDENCE] '+token_clip(tokenizer,row['evidence_text'],ev_budget))
 for i,ref in enumerate(refs):
  cues=' ; '.join(ref.get('change_cues',[]) or [])
  if ref.get('reserve_candidates'):cues+=' ; '+' ; '.join(ref['reserve_candidates'])
  budget_cue=min(each//2,max(8,each-16))
  parts.append(f'[REF_{i+1}] [CUES] '+token_clip(tokenizer,cues,budget_cue)+
               ' [HIST_ASSERT] '+token_clip(tokenizer,ref['historical_assertion'],max(8,each-budget_cue)))
 return ' '.join(parts)

