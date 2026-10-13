"""Check exported replay remains faithful to held-out results and refusals."""
import json
import hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=json.loads((root/'webui/dist/assets/model-replay.json').read_text())
assert len(p['sessions'])==16
assert sum(len(s['windows']) for s in p['sessions'])==192
unknown=0; gated=0; outputs=0
for s in p['sessions']:
 for w in s['windows']:
  gated+=not w['quality_ok']
  for b in p['backends']:
   for state in p['states']:
    pred=w['predictions'][b]['sim_'+state];outputs+=1
    if pred['probabilities'] is not None:
     assert abs(sum(pred['probabilities'].values())-1)<1e-5
    else:
     assert pred['label']=='unknown' and pred['confidence'] is None
    if pred['label']=='unknown':unknown+=1
    else:
     assert w['quality_ok']
     assert pred['confidence']>=p['metrics'][b]['sim_'+state]['threshold']
for path,digest in p['source_sha256'].items():
 assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
assert gated==3
print(f'PASS: 16 sessions, 192 windows, {outputs} outputs, {unknown} refusals, 3 quality-gated windows, source hashes verified')
