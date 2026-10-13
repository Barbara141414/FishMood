"""Compare JS rules directly with current Python simulator, using unit coefficients."""
import json,subprocess
import numpy as np
from fishwatch.simulation import update_state,action_probabilities,DEFAULT_PARAMETERS
from pathlib import Path
root=Path(__file__).resolve().parents[2]
rng=np.random.default_rng(73)
cases=[{'s':rng.random(4).tolist(),'dt':float(rng.uniform(0,12)),'intake':int(rng.integers(0,4)),'disturb':int(rng.integers(0,2)),'rest':bool(rng.integers(0,2)),'b':float(rng.random())} for _ in range(100)]
js="import {updateState,actionProbabilities} from './webui/dist/latent.js';let input='';for await(const chunk of process.stdin)input+=chunk;console.log(JSON.stringify(JSON.parse(input).map(c=>({s:updateState(c.s,c.dt,c.intake,c.disturb,c.rest),p:actionProbabilities(c.s,c.b)}))));"
r=subprocess.run(['node','--input-type=module','-e',js],input=json.dumps(cases),text=True,capture_output=True,check=True,cwd=root)
for c,o in zip(cases,json.loads(r.stdout)):
 np.testing.assert_allclose(o['s'],update_state(c['s'],c['dt'],c['intake'],c['disturb'],c['rest'],DEFAULT_PARAMETERS),rtol=0,atol=1e-12)
 np.testing.assert_allclose(o['p'],action_probabilities(c['s'],c['b'],np.ones(5)),rtol=0,atol=1e-12)
print('PASS: 100 JS/Python state and softmax comparisons, absolute tolerance 1e-12 (unit rates/coefficients)')
