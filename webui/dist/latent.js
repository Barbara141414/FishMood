// Mirrors fishwatch/simulation.py defaults. Engineering hypotheses, not physiology.
export const INITIAL=[[.72,.18,.75,.22],[.8,.25,.65,.3],[.4,.3,.72,.2]];
export const ACTION_NAMES=['探索','觅食','休息','躲避','急冲'];
export const clamp=v=>Math.max(0,Math.min(1,v));
export function updateState(state,minutes,intake=0,disturbance=0,resting=false){const [h,s,e,f]=state;return [clamp(h+.002*minutes-.12*intake),clamp(s+.18*disturbance-.004*s*minutes),e,clamp(f+(.001*Number(!resting)-.002*Number(resting))*minutes)];}
export function actionProbabilities([h,s,e,f],b){const scores=[1+3*e+b-2*f-s,4*h-2*s-f,3*f+1.2*s-1,3*s-e,2.5*s+b-f-1];const max=Math.max(...scores),weights=scores.map(v=>Math.exp(v-max)),sum=weights.reduce((a,b)=>a+b);return weights.map(v=>v/sum);}
export function chooseAction(state,b,rng=Math.random){let value=rng();const probabilities=actionProbabilities(state,b);for(let i=0;i<probabilities.length;i++){value-=probabilities[i];if(value<=0)return i;}return 4;}
export function intakeChance([h,s,e,f]){return h*(1-.8*s)*(1-.3*f);}

export function seededRandom(initialSeed=73){let seed=initialSeed>>>0;return ()=>{seed=(seed*1664525+1013904223)>>>0;return seed/4294967296;};}
// Frame-rate independent fixed simulation steps. Hidden tabs are paused by caller.
export function fixedClock(step,interval=.125){let accumulator=0;return {advance(seconds,speed=1,paused=false){if(!Number.isFinite(seconds)||seconds<0||!Number.isFinite(speed)||speed<=0)throw Error('Invalid simulation clock');if(paused)return;accumulator+=seconds*speed;while(accumulator+1e-10>=interval){step(interval);accumulator=Math.max(0,accumulator-interval);}},reset(){accumulator=0;}};}
