import assert from 'node:assert/strict';
import {INITIAL,updateState,actionProbabilities,intakeChance} from '../dist/latent.js';
const base=[.7,.5,.6,.4],near=(a,b)=>assert.ok(Math.abs(a-b)<1e-12);
near(updateState(base,0,0)[0],.7);
near(updateState(base,0,1)[0],.58);
near(updateState(base,0,0,1)[1],.68);
assert.ok(updateState(base,10,0,0,true)[3]<base[3]);
assert.ok(updateState(base,10)[1]<base[1]);
assert.equal(updateState(base,10)[2],base[2]);
for(const state of [...INITIAL,[0,0,0,0],[1,1,1,1]]){const p=actionProbabilities(state,.5);near(p.reduce((a,b)=>a+b),1);assert.ok(p.every(v=>v>=0&&v<=1));assert.ok(updateState(state,100,20,10).every(v=>v>=0&&v<=1));assert.ok(intakeChance(state)>=0&&intakeChance(state)<=1);}
console.log('PASS: intake-only satiety, stress gain/recovery, fixed E, fatigue/rest, state bounds, behavior probabilities');
const {seededRandom,fixedClock,chooseAction}=await import('../dist/latent.js');
assert.equal(INITIAL.length,3);
function run(fps,speed,duration){let state=[...INITIAL[0]],elapsed=0,next=0,action=0;const rng=seededRandom(73);const clock=fixedClock(dt=>{elapsed+=dt;if(elapsed>=next){action=chooseAction(state,.4,rng);next=elapsed+.75;}state=updateState(state,dt,0,0,action===2);});for(let i=0;i<fps*duration;i++)clock.advance(1/fps,speed);return{state,elapsed};}
assert.deepEqual(run(10,1,60),run(60,1,60));assert.deepEqual(run(30,2,30),run(60,1,60));assert.deepEqual(run(60,.5,120),run(60,1,60));assert.deepEqual(run(60,1,60),run(60,1,60));
let ticks=0;const timer=fixedClock(()=>ticks++);timer.advance(5,1,true);assert.equal(ticks,0);timer.advance(.1);timer.reset();timer.advance(.1);assert.equal(ticks,0);timer.advance(.025);assert.equal(ticks,1);
let state=[...INITIAL[1]];const e=state[2];for(let i=0;i<28800;i++){state=updateState(state,.125,i%61===0?1:0,i%79===0?1:0,i%3===0);assert.ok(state.every(v=>Number.isFinite(v)&&v>=0&&v<=1));assert.equal(state[2],e);}
console.log('PASS: 3 fish, 10/30/60 FPS and 0.5/1/2x equivalence, seeded repeatability, pause/reset accumulator, 28,800 bounded steps');
