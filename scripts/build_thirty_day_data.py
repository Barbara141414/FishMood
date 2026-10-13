"""Independent 30-day, three-fish synthetic experiment (engineering assumptions only).

Each fish has its own seeded minute-resolution hidden-state/action/position process.
Models estimate the *current* hourly hidden category from realized observable behavior;
this is not a prospective feeding-response predictor or a real-fish measurement.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from fishwatch.classical import fit_classical, probabilities

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'webui/dist/assets/thirty-days.json'
SEED = 173
FISH = (('huaban', '花斑'), ('hongding', '红顶'), ('lanyin', '蓝银'))
DIMENSIONS = ('H', 'S', 'E')
BACKENDS = ('logistic', 'gam', 'hmm')
FEATURES = ('explore_fraction', 'forage_fraction', 'rest_fraction',
            'hide_fraction', 'dart_fraction', 'speed_cm_per_minute',
            'food_zone_fraction', 'shelter_fraction', 'surface_fraction',
            'turn_fraction', 'displacement_cm')
FOOD = np.array([24., 32., 10.])
SHELTER = np.array([-26., 20., -12.])
LOW = np.array([-38., 11., -22.])
HIGH = np.array([38., 39., 24.])
# First eight days explicitly exercise every binary combination at the noon checkpoint.
# Remaining scenarios are fixed a priori, never selected using holdout scores.
SCENARIOS = ('LLL', 'HLL', 'LHL', 'LLH', 'HHL', 'HLH', 'LHH', 'HHH',
             'LHL', 'HLH', 'LLL', 'HHH', 'HLL', 'LHH', 'HHL', 'LLH',
             'HLH', 'LLL', 'HHH', 'LHL', 'HHL', 'LLH', 'HHH', 'HLL',
             'LHH', 'HLH', 'LLL', 'HHL', 'LHL', 'HHH')
PERSONALITIES = (
    {'boldness': .40, 'h_offset': .02, 's_offset': .01, 'e_offset': -.02,
     'description': '花斑：稳重、行动相对克制，惊扰后恢复较缓。'},
    {'boldness': .78, 'h_offset': .01, 's_offset': -.04, 'e_offset': .08,
     'description': '红顶：好奇、爱巡游，探索和靠近新事物的倾向较高。'},
    {'boldness': .18, 'h_offset': -.03, 's_offset': .07, 'e_offset': -.07,
     'description': '蓝银：敏感、安静，较常靠近遮蔽区。'},
)


def _action(state, personality, rng):
    h, s, e = state
    b = personality['boldness']
    logits = np.array([.3 + 3.5*e + 1.2*b - 1.4*s,
                       .2 + 3.8*h - 1.1*s + (1-b)*.3,
                       .6 + 2.2*(1-e) - .5*b,
                       -.8 + 4.2*s - .7*b,
                       -1.2 + 2.7*s + 1.0*b])
    weights = np.exp(logits-logits.max())
    return int(rng.choice(5, p=weights/weights.sum()))


def _observation(actions, speeds, positions, turns):
    counts = np.bincount(actions, minlength=5)/len(actions)
    loc = np.asarray(positions)
    return np.array([*counts, float(np.mean(speeds)),
                     float(np.mean(np.linalg.norm(loc-FOOD, axis=1) < 12)),
                     float(np.mean(np.linalg.norm(loc-SHELTER, axis=1) < 13)),
                     float(np.mean(loc[:, 1] > 33)), float(np.mean(turns)),
                     float(np.linalg.norm(loc[-1]-loc[0]))])


def _simulate_fish(index, events):
    fid, name = FISH[index]
    personality = PERSONALITIES[index]
    rng = np.random.default_rng(SEED + 1009*index)
    state = np.array([.19 + personality['h_offset'],
                      .19 + personality['s_offset'],
                      .19 + personality['e_offset']])
    position = np.array([(-20., 23., -2.), (0., 28., 11.),
                         (21., 19., -10.)][index])
    heading = float(rng.uniform(-math.pi, math.pi))
    action = _action(state, personality, rng)
    offered_until = -1
    intake_today = 0
    points, xs, rows = [], [], []
    actions, speeds, positions, turns = [action], [0.], [position.copy()], [0.]

    def snapshot(minute):
        features = _observation(actions, speeds, positions, turns)
        xs.append(features)
        rows.append({'analysis': fid, 'start_s': minute, 'end_s': minute+60,
                     'quality_ok': True})
        points.append({'minute': minute,
                       'state': dict(zip(DIMENSIONS, [round(float(v), 5) for v in state])),
                       'action': action, 'position': [round(float(v), 3) for v in position],
                       'features': [round(float(v), 6) for v in features],
                       'models': {}, 'quality_ok': True})

    snapshot(0)  # Initial instantaneous sample; hours 1..719 use trailing minutes.
    for minute in range(1, 43200):
        day, clock = divmod(minute, 1440)
        scenario = SCENARIOS[day]
        target = np.array([(.79 if bit == 'H' else .19) + personality[key]
                           for bit, key in zip(scenario, ('h_offset', 's_offset', 'e_offset'))])
        if clock == 465 + 17*index:
            offered_until = minute + 105
            intake_today = 0
            events.append({'minute': minute, 'fish_id': fid, 'type': 'food_offered'})
        disturbed = clock == 870 + 23*index
        if disturbed:
            events.append({'minute': minute, 'fish_id': fid, 'type': 'disturbance'})
        action = _action(state, personality, rng)
        previous = position.copy()
        old_heading = heading
        goal = FOOD if action == 1 and minute <= offered_until else SHELTER if action == 3 else None
        if goal is not None:
            direction = goal-position+rng.normal(0, 2.2, 3)
            heading = math.atan2(direction[2], direction[0])
            vertical = float(np.clip(direction[1]/max(np.linalg.norm(direction), 1), -.5, .5))
        else:
            heading += float(rng.normal(0, .23 if action != 4 else .8))
            vertical = float(rng.normal(0, .22))
        speed = [1.25, 1.7, .12, 1.1, 3.1][action] * (.8+.35*personality['boldness'])
        position = np.clip(position + np.array([math.cos(heading), vertical, math.sin(heading)])*speed,
                           LOW, HIGH)
        moved = float(np.linalg.norm(position-previous))
        consumed = (minute <= offered_until and intake_today < 2 and action == 1
                    and np.linalg.norm(position-FOOD) < 9
                    and rng.random() < .19*state[0]*(1-.65*state[1]))
        if consumed:
            intake_today += 1
            events.append({'minute': minute, 'fish_id': fid, 'type': 'food_consumed'})
        # H is contemporaneous simulated appetite drive; daily context changes its
        # baseline, and only actual simulated intake adds the separate satiation drop.
        state[0] = np.clip(state[0] + .008*(target[0]-state[0]) - .24*int(consumed), 0, 1)
        state[1] = np.clip(state[1] + (.006 if index == 0 else .008 if index == 1 else .0045)*(target[1]-state[1])
                           + .43*int(disturbed), 0, 1)
        diurnal = .035*math.sin(2*math.pi*(clock-360)/1440)
        state[2] = np.clip(state[2] + .006*(target[2]+diurnal-state[2])
                           + rng.normal(0, .0007), 0, 1)
        actions.append(action)
        speeds.append(moved)
        positions.append(position.copy())
        turns.append(float(abs(math.atan2(math.sin(heading-old_heading),
                                          math.cos(heading-old_heading))) > .65))
        if minute % 60 == 0:
            snapshot(minute)
            actions, speeds, positions, turns = [], [], [], []
    return {'id': fid, 'name': name, 'points': points}, np.asarray(xs), rows


def _metrics(y, p, prior):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    return {'points': int(len(y)), 'positive_fraction': round(float(y.mean()), 6),
            'accuracy_at_0_5': round(float(np.mean((p >= .5) == y)), 6),
            'brier': round(float(np.mean((p-y)**2)), 6),
            'baseline_train_prior': round(float(prior), 6),
            'baseline_majority_accuracy': round(float(np.mean((prior >= .5) == y)), 6),
            'baseline_prior_brier': round(float(np.mean((prior-y)**2)), 6)}


def _parameters(estimator, backend, train_x):
    if backend == 'hmm':
        return {'initial': estimator.initial.tolist(), 'transition': estimator.transition.tolist(),
                'standardized_means': estimator.means.tolist(),
                'standardized_variances': estimator.variances.tolist(),
                'preprocess_mean': estimator.preprocess[-1].mean_.tolist(),
                'preprocess_scale': estimator.preprocess[-1].scale_.tolist()}
    classifier = estimator[-1]
    result = {'classes': classifier.classes_.tolist(), 'coefficients': classifier.coef_.tolist(),
              'intercept': classifier.intercept_.tolist(),
              'preprocess_mean': estimator[1].mean_.tolist(),
              'preprocess_scale': estimator[1].scale_.tolist()}
    if backend == 'gam':
        result['spline_knot_vectors'] = [spline.t.tolist() for spline in estimator[2].bsplines_]
        result['spline_degree'] = estimator[2].degree
        # A conditional model-response display, holding other inputs at training medians.
        # These are fitted associations, not causal effects or calibrated state values.
        medians = np.median(train_x, axis=0)
        curves = {}
        for j, name in enumerate(FEATURES):
            lo, hi = np.quantile(train_x[:, j], [.05, .95])
            grid = np.linspace(lo, hi, 32)
            inputs = np.tile(medians, (len(grid), 1))
            inputs[:, j] = grid
            curves[name] = {'x': grid.round(6).tolist(),
                            'p_high': estimator.predict_proba(inputs)[:, 1].round(6).tolist()}
        result['partial_response_at_training_medians'] = curves
    return result


def build_data():
    events = []
    streams = [_simulate_fish(i, events) for i in range(3)]
    train_x = np.vstack([x[:480] for _, x, _ in streams])
    train_rows = [row for _, _, rows in streams for row in rows[:480]]
    metrics, parameters = {}, {}
    for backend in BACKENDS:
        metrics[backend], parameters[backend] = {}, {}
        for dimension in DIMENSIONS:
            train_y = np.array([int(p['state'][dimension] >= .5)
                                for fish, _, _ in streams for p in fish['points'][:480]])
            assert set(train_y) == {0, 1}, (backend, dimension)
            estimator = fit_classical(backend, train_x, train_y, train_rows)
            parameters[backend][dimension] = _parameters(estimator, backend, train_x)
            prior = float(train_y.mean())
            outcomes, estimates, by_fish, strata = [], [], {}, {}
            for fish, x, rows in streams:
                # Each HMM starts at this fish's first observation and filters forward.
                p = probabilities(estimator, x, rows)[:, list(estimator.classes_).index(1)]
                for point, prob in zip(fish['points'], p):
                    point['models'].setdefault(backend, {})[dimension] = round(float(prob), 6)
                held = fish['points'][480:]
                yy = [int(pt['state'][dimension] >= .5) for pt in held]
                pp = [pt['models'][backend][dimension] for pt in held]
                outcomes.extend(yy); estimates.extend(pp)
                by_fish[fish['id']] = _metrics(yy, pp, prior)
                own_strata = {}
                for point, y, value in zip(held, yy, pp):
                    for key in (f'label_{y}', f'scenario_{SCENARIOS[point["minute"]//1440]}'):
                        group = own_strata.setdefault(key, ([], []))
                        group[0].append(y); group[1].append(value)
                by_fish[fish['id']]['strata'] = {
                    key: _metrics(y, p, prior) for key, (y, p) in sorted(own_strata.items())}
                for point, y, value in zip(held, yy, pp):
                    hour = point['minute']//60
                    context = SCENARIOS[hour//24]
                    # Diagnostic slices fixed before fitting: label and scenario,
                    # including potential confounding by the other two dimensions.
                    for key in (f'label_{y}', f'scenario_{context}'):
                        group = strata.setdefault(key, ([], []))
                        group[0].append(y); group[1].append(value)
            metrics[backend][dimension] = {
                'overall': _metrics(outcomes, estimates, prior), 'fish': by_fish,
                'strata': {key: _metrics(y, p, prior) for key, (y, p) in sorted(strata.items())}}
    # After honest chronological evaluation, refit a separate descriptive model
    # on all 30 days. Never use it to score the held-out days above.
    full_x = np.vstack([x for _, x, _ in streams])
    full_rows = [row for _, _, rows in streams for row in rows]
    full_parameters = {}
    for backend in BACKENDS:
        full_parameters[backend] = {}
        for dimension in DIMENSIONS:
            full_y = np.array([int(p['state'][dimension] >= .5)
                               for fish, _, _ in streams for p in fish['points']])
            final_estimator = fit_classical(backend, full_x, full_y, full_rows)
            full_parameters[backend][dimension] = _parameters(final_estimator, backend, full_x)
    events.sort(key=lambda e: (e['minute'], e['fish_id'], e['type']))
    return {'domain': 'synthetic', 'kind': 'thirty_day_three_fish', 'seed': SEED,
            'step_minutes': 60, 'days': 30, 'threshold': .5,
            'fish': [fish for fish, _, _ in streams], 'events': events,
            'model_meta': {
                'notice': 'Synthetic engineering assumptions only; hidden H/S/E are NOT validated real-fish psychological or biological readings. Holdout measures same-generator generalization only.',
                'fit': 'Same eleven behavior/position features and >=0.5 hidden labels for all three backends. Fish hours 0-479 (days 1-20) train; hours 480-719 (days 21-30) chronological holdout. No fitting or tuning on holdout.',
                'train_minutes': [0, 28740], 'holdout_minutes': [28800, 43140],
                'label_threshold': .5, 'features': list(FEATURES),
                'feature_source': 'Realized actions and positions over trailing 60 minutes (hour 0 instantaneous); no hidden states, scenarios, event indicators, simulated intake, or future feeding response in X.',
                'hour_zero': 'Initial instantaneous observation is part of training, not a complete 60-minute window.',
                'simulation': 'Independent seeded 1-minute trajectories per virtual fish; fixed daily contexts drive current H/S/E baseline; actual simulated intake additionally reduces H. Offering food is an opportunity, not confirmation of intake. Scheduled disturbance adds transient S. All dynamics, coefficients, and scenarios are engineering assumptions.',
                'scenarios': [{'day': i+1, 'target_HSE': bits} for i, bits in enumerate(SCENARIOS)],
                'scenario_checkpoint': 'Noon (hour 12) on days 1-8 deliberately covers all eight high/low H/S/E combinations for each fish; letters describe generator target, not observed facts or model predictions.',
                'personalities': {fish_id: personality for (fish_id, _), personality in zip(FISH, PERSONALITIES)},
                'events': 'Per-fish daily food opportunity at minute-of-day 465+17*index (105-minute window); disturbance at 870+23*index; simulated intake is a separate stochastic event.',
                'hmm': 'Supervised label-anchored Gaussian HMM; each fish filtered independently from its first hour, never smoothing or seeing later observations.',
                'probabilities': 'P(synthetic hidden current-hour state >= 0.5), not continuous state regression, future feeding response, or real-fish inference.',
                'model_parameters': parameters,
                'full_30_day_parameters': full_parameters,
                'parameter_stages': 'model_parameters are 20-day evaluation fits used for displayed probabilities and holdout scores; full_30_day_parameters are separate all-30-day refits for descriptive use only. They do NOT score the holdout.',
                'holdout_metrics': metrics}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = build_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n',
                           encoding='utf-8')
    print(f"Wrote {sum(len(f['points']) for f in data['fish'])} hourly points to {args.output}")


if __name__ == '__main__':
    main()
