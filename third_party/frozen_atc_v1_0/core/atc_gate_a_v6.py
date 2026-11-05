"""ATC Gate-A structural map, 2026-10-22.
Only Source is read. No Target assertions, Java semantic-equivalence inference,
or model-generated expectations are involved.
"""
from collections import Counter
from gatea_v5 import graph, prefix
IGNORE=set('{}().,;:=+-*/<>!?# [ ] <AssertPlaceHolder> final public static new return true false null void int String org java util lang test'.split())

def evidence_jaccard(s1,s2):
    def bag(s):return Counter(x for x in prefix(s) if x not in IGNORE and len(x)>1 and not x.startswith('"'))
    a,b=bag(s1),bag(s2)
    union=sum((a|b).values())
    return sum((a&b).values())/union if union else 0.0

def align(source1,source2):
    """Return a reversible, provenance-preserving change catalog.

    Confirmed = unique receiver+method+arity invocation, changed arguments,
    plus fixed source-context Jaccard >= .55. Other maps stay in reserve.
    This is structural correspondence, not Java semantic equivalence.
    """
    g=graph(source1,source2)
    sim=evidence_jaccard(source1,source2)
    pairs=[]
    for p in g['pairs']:
        p1,p2=(p['right'],p['left']) if g['swapped'] else (p['left'],p['right'])
        start1,start2=(p['right_start'],p['left_start']) if g['swapped'] else (p['left_start'],p['right_start'])
        pairs.append({'historical_expression':p1,'current_expression':p2,
            'historical_span_start':start1,'current_span_start':start2,
            'change_kind':p['kind'], 'call_receiver':p['receiver'],'call_name':p['method'],
            'arity':p['arity'],'context_similarity':round(sim,4),
            'status':'CONFIRMED_STRUCTURAL' if sim>=.55 else 'RESERVE_CANDIDATE'})
    return {'relations':pairs,'confirmed_count':sum(x['status']=='CONFIRMED_STRUCTURAL' for x in pairs),
            'reserve_count':sum(x['status']=='RESERVE_CANDIDATE' for x in pairs),
            'context_similarity':round(sim,4)}

def check_bidirectional(s1,s2):
    forward=align(s1,s2)['relations']
    reverse=align(s2,s1)['relations']
    a={(x['historical_expression'],x['current_expression'],x['status']) for x in forward}
    b={(x['current_expression'],x['historical_expression'],x['status']) for x in reverse}
    return a==b

