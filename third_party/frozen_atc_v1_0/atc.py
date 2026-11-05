"""Frozen ATC v1.0 integration: Gate A v6 + Gate B frozen heuristic kernels.

Only (historical_source, historical_assertion, current_source) are read at inference.
Current reference assertion and evaluation edit journals must never enter this API.

This wrapper unifies existing validated kernels; it does not retrain or change
matching rules, and it does not certify semantic equivalence.
"""
from __future__ import annotations
import sys
from pathlib import Path
CORE = Path(__file__).resolve().parent / 'core'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
from atc_gate_a_v6 import align, check_bidirectional
from run_gateb import guided, ts, good

SCHEMA = 'assertpath.atc.change_cues.v1'

def build_change_cues(historical_source: str, historical_assertion: str,
                      current_source: str, *, historical_ref_id: str = '') -> dict:
    """One historical TAP -> structured, directed Change Cues + reserve candidates."""
    for x in (historical_source, historical_assertion, current_source):
        if not isinstance(x, str):
            raise TypeError('historical_source, historical_assertion, current_source must be strings')
    assertion_anchors = {x for x in ts(historical_assertion) if good(x)}
    raw_structure = align(historical_source, current_source)
    primary, reserve = [], []
    for r in raw_structure['relations']:
        linked = r['call_receiver'] in assertion_anchors
        c = {
            'cue_type': r['change_kind'],
            'historical_expression': r['historical_expression'],
            'current_expression': r['current_expression'],
            'historical_span_start': r['historical_span_start'],
            'current_span_start': r['current_span_start'],
            'assertion_anchor': r['call_receiver'] if linked else None,
            'call_receiver': r['call_receiver'],
            'call_name': r['call_name'],
            'arity': r['arity'],
            'context_similarity': r['context_similarity'],
            'status': r['status'],
            'provenance': 'GATE_A_STRUCTURAL',
        }
        (primary if linked and r['status']=='CONFIRMED_STRUCTURAL' else reserve).append(c)
    for old,new,score in guided(historical_source,current_source,historical_assertion):
        primary.append({
            'cue_type': 'IDENTIFIER_CHANGE',
            'historical_expression': old,
            'current_expression': new,
            'assertion_anchor': old,
            'alignment_score': round(score,6),
            'status': 'ASSERTION_LINKED_CANDIDATE',
            'provenance': 'GATE_B_LEXICAL',
        })
    return {
        'schema': SCHEMA,
        'historical_ref_id': historical_ref_id,
        'direction': 'historical_to_current',
        'change_cues': primary,
        'reserve_candidates': reserve,
        'summary': {
            'primary_n': len(primary), 'reserve_n': len(reserve),
            'structural_n': len(raw_structure['relations']),
            'bidirectional_consistent': check_bidirectional(historical_source,current_source),
        },
    }

def build_multi_reference(current_source: str, historical_references: list[dict]) -> dict:
    """One current Source, k historical {source, assertion, ref_id} entries."""
    outputs = []
    for index, h in enumerate(historical_references):
        outputs.append(build_change_cues(
            h['source'], h['assertion'], current_source,
            historical_ref_id=str(h.get('ref_id', index)),
        ))
    return {'schema': SCHEMA, 'current_source': current_source,
            'references': outputs, 'reference_count': len(outputs)}

if __name__=='__main__':
    import argparse, json
    parser = argparse.ArgumentParser(description='AssertPath frozen ATC v1.0')
    parser.add_argument('input_json', help='JSON file containing historical_source, historical_assertion, current_source')
    args = parser.parse_args()
    q=json.loads(Path(args.input_json).read_text())
    print(json.dumps(build_change_cues(**q), indent=2, ensure_ascii=False))

