"""Generate a new hourly, three-fish synthetic experiment without rendering video.

All dynamics and observation coefficients below are engineering assumptions, not
biological estimates. This is independent of the historical 48-second simulator.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from fishwatch.classical import fit_classical, probabilities

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'webui/dist/assets/ten-days.json'
SEED = 73
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


def _choose_action(state, boldness, rng):
    h, s, e = state
    logits = np.array([.6 + 2.8*e + boldness - 1.5*s,
                       .4 + 3*h - 1.5*s, .7 + 1.8*(1-e),
                       -.7 + 3.5*s, -1 + 3*s + boldness])
    weights = np.exp(logits - logits.max())
    return int(rng.choice(5, p=weights/weights.sum()))


def _observation(acts, speeds, positions, turns):
    # Only realized actions and positions enter X; neither latent state nor events
    # nor food intake are supplied to any estimator.
    counts = np.bincount(acts, minlength=5)/len(acts)
    loc = np.asarray(positions)
    return np.array([*counts, float(np.mean(speeds)),
                     float(np.mean(np.linalg.norm(loc-FOOD, axis=1) < 12)),
                     float(np.mean(np.linalg.norm(loc-SHELTER, axis=1) < 13)),
                     float(np.mean(loc[:, 1] > 33)), float(np.mean(turns)),
                     float(np.linalg.norm(loc[-1]-loc[0]))])


def _simulate_fish(index, events):
    fish_id, name = FISH[index]
    rng = np.random.default_rng(SEED + index*1009)
    state = np.array([(.42, .64, .28)[index], (.19, .25, .32)[index],
                      (.42, .57, .68)[index]], dtype=float)
    boldness = (.27, .48, .72)[index]
    position = np.array([(-20., 23., -2.), (0., 28., 11.), (21., 19., -10.)][index])
    heading = float(rng.uniform(-math.pi, math.pi))
    action = _choose_action(state, boldness, rng)
    offered_until = -1
    intake_today = 0
    points, xs, rows = [], [], []
    actions, speeds, positions, turns = [action], [0.], [position.copy()], [0.]

    def snapshot(minute):
        features = _observation(actions, speeds, positions, turns)
        xs.append(features)
        rows.append({'analysis': fish_id, 'start_s': minute, 'end_s': minute+60,
                     'quality_ok': True})
        points.append({'minute': minute,
                       'state': dict(zip(DIMENSIONS, [round(float(v), 5) for v in state])),
                       'action': action,
                       'position': [round(float(v), 3) for v in position],
                       'models': {}, 'quality_ok': True})

    snapshot(0)  # Initial instantaneous observation; later points use the trailing hour.
    for minute in range(1, 14341):
        day, clock = divmod(minute, 1440)
        # One opportunity and one disturbance per fish/day, with individual offsets.
        if clock == 480 + 17*index:
            offered_until = minute + 100
            intake_today = 0
            events.append({'minute': minute, 'fish_id': fish_id, 'type': 'food_offered'})
        disturbed = clock == 850 + 23*index
        if disturbed:
            events.append({'minute': minute, 'fish_id': fish_id, 'type': 'disturbance'})
        action = _choose_action(state, boldness, rng)
        previous = position.copy()
        old_heading = heading
        if action == 1 and minute <= offered_until:
            target = FOOD
        elif action == 3:
            target = SHELTER
        else:
            target = None
        if target is not None:
            direction = target-position + rng.normal(0, 2.2, 3)
            heading = math.atan2(direction[2], direction[0])
            vertical = float(np.clip(direction[1]/max(np.linalg.norm(direction), 1), -.5, .5))
        else:
            heading += float(rng.normal(0, .23 if action != 4 else .8))
            vertical = float(rng.normal(0, .22))
        speed = [1.25, 1.7, .12, 1.1, 3.1][action] * (.8 + .35*boldness)
        velocity = np.array([math.cos(heading), vertical, math.sin(heading)])
        position = np.clip(position + velocity*speed, LOW, HIGH)
        moved = float(np.linalg.norm(position-previous))
        consumed = (minute <= offered_until and intake_today < 2 and action == 1
                    and np.linalg.norm(position-FOOD) < 9
                    and rng.random() < .23*state[0]*(1-.65*state[1]))
        if consumed:
            intake_today += 1
            events.append({'minute': minute, 'fish_id': fish_id, 'type': 'food_consumed'})
        state[0] = np.clip(state[0] + .00075 - .35*int(consumed), 0, 1)
        state[1] = np.clip(state[1] - .005*state[1] + .72*int(disturbed), 0, 1)
        # New ten-day hypothesis: diurnal target plus slow mean reversion and noise.
        # Unlike the old short session, E is deliberately NOT fixed.
        diurnal = .52 + .18*math.sin(2*math.pi*(clock-360)/1440) + (.0, .06, -.06)[index]
        state[2] = np.clip(state[2] + .003*(diurnal-state[2]) + rng.normal(0, .0012), 0, 1)
        actions.append(action)
        speeds.append(moved)
        positions.append(position.copy())
        turns.append(float(abs(math.atan2(math.sin(heading-old_heading),
                                          math.cos(heading-old_heading))) > .65))
        if minute % 60 == 0:
            snapshot(minute)
            actions, speeds, positions, turns = [], [], [], []
    return {'id': fish_id, 'name': name, 'points': points}, np.asarray(xs), rows


def build_data():
    events = []
    streams = [_simulate_fish(i, events) for i in range(3)]
    fish = [stream[0] for stream in streams]
    train_x = np.vstack([x[:168] for _, x, _ in streams])
    train_rows = [row for _, _, rows in streams for row in rows[:168]]
    metrics = {}
    for backend in BACKENDS:
        metrics[backend] = {}
        for dimension in DIMENSIONS:
            train_y = np.array([int(point['state'][dimension] >= .5)
                                for item, _, _ in streams for point in item['points'][:168]])
            assert len(np.unique(train_y)) == 2, (backend, dimension)
            estimator = fit_classical(backend, train_x, train_y, train_rows)
            holdout_prob, holdout_truth = [], []
            for item, x, rows in streams:
                # Run the forward filter once from the start of each fish's history.
                p = probabilities(estimator, x, rows)[:, list(estimator.classes_).index(1)]
                holdout_prob.extend(p[168:])
                holdout_truth.extend(int(pt['state'][dimension] >= .5)
                                     for pt in item['points'][168:])
                for pt, prob in zip(item['points'], p):
                    pt['models'].setdefault(backend, {})[dimension] = round(float(prob), 5)
            metrics[backend][dimension] = {
                'holdout_accuracy_at_0_5': round(float(np.mean((np.asarray(holdout_prob) >= .5) == holdout_truth)), 4),
                'holdout_points': len(holdout_truth),
            }
    events.sort(key=lambda event: (event['minute'], event['fish_id'], event['type']))
    return {'domain': 'synthetic', 'kind': 'ten_day_three_fish', 'seed': SEED,
            'step_minutes': 60, 'days': 10,
            'model_origin': 'newly_fitted_hourly_synthetic_features',
            'threshold': .7, 'fish': fish, 'events': events,
            'model_meta': {
                'notice': 'Synthetic engineering hypotheses; not real fish psychology or biological validation.',
                'fit': 'First 7 days per fish (hours 0-167); last 3 days chronological holdout (hours 168-239).',
                'train_minutes': [0, 10020], 'holdout_minutes': [10080, 14340],
                'label_threshold': .5, 'acceptance_threshold': .7,
                'features': list(FEATURES),
                'feature_source': 'Realized action and 3D position over trailing 60 minutes; hour 0 is instantaneous. No latent state, event, or intake feature.',
                'simulation': 'Independent seeded fish, 1-minute latent/action/position updates; scheduled daily food opportunities and disturbances; H rises 0.00075/min and drops 0.35 only on simulated intake; S decays 0.005*S/min and gains 0.72 per disturbance; E follows a new noisy diurnal mean-reverting target (not the old fixed-E short-session rule).',
                'hmm': 'Supervised label-anchored Gaussian HMM; per-fish forward filtering only, no smoothing or future observations.',
                'probabilities': 'P(synthetic high state >= 0.5), not continuous state regression; trained on this new hourly mechanism, not old 4-second windows.',
                'holdout_metrics': metrics}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = build_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
    print(f"Wrote {sum(len(f['points']) for f in data['fish'])} hourly points to {args.output}")


if __name__ == '__main__':
    main()
