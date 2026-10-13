"""Contract, chronology, leakage and metric checks for the independent 30-day export."""
import json
from collections import Counter

import numpy as np
import pytest

from scripts import build_thirty_day_data as generator
from fishwatch.classical import probabilities


@pytest.fixture(scope='module')
def built():
    observed = {}
    original = generator.fit_classical

    def capture(backend, x, y, rows):
        model = original(backend, x, y, rows)
        observed[(backend, len(observed))] = (model, x.copy(), y.copy(),
                                               [r.copy() for r in rows])
        return model

    generator.fit_classical = capture
    try:
        data = generator.build_data()
    finally:
        generator.fit_classical = original
    assert json.loads(generator.DEFAULT_OUTPUT.read_text(encoding='utf-8')) == data
    # Re-run both stochastic generation and fitting with the same fixed seed.
    assert generator.build_data() == data
    return data, observed


def test_hourly_contract_scenarios_and_events(built):
    data, _ = built
    assert (data['domain'], data['kind'], data['days'], data['step_minutes']) == (
        'synthetic', 'thirty_day_three_fish', 30, 60)
    assert [f['id'] for f in data['fish']] == ['huaban', 'hongding', 'lanyin']
    assert len(data['model_meta']['features']) == 11
    assert len(data['model_meta']['scenarios']) == 30
    assert {s['target_HSE'] for s in data['model_meta']['scenarios'][:8]} == {
        ''.join(bits) for bits in __import__('itertools').product('LH', repeat=3)}
    assert data['model_meta']['train_minutes'] == [0, 28740]
    assert data['model_meta']['holdout_minutes'] == [28800, 43140]
    assert 'NOT validated real-fish' in data['model_meta']['notice']
    assert 'future feeding response' in data['model_meta']['feature_source']
    paths = []
    for fish in data['fish']:
        points = fish['points']
        assert len(points) == 720
        assert [p['minute'] for p in points] == list(range(0, 43141, 60))
        paths.append(tuple(tuple(p['position']) for p in points))
        for p in points:
            assert {'minute', 'state', 'action', 'position', 'models', 'quality_ok'} <= set(p)
            assert p['quality_ok'] is True and p['action'] in range(5)
            assert all(0 <= p['state'][d] <= 1 for d in 'HSE')
            assert len(p['features']) == 11 and all(np.isfinite(p['features']))
            assert all(lo <= v <= hi for v, lo, hi in zip(
                p['position'], (-38, 11, -22), (38, 39, 24)))
            assert set(p['models']) == set(generator.BACKENDS)
            assert all(set(p['models'][b]) == set('HSE') and
                       all(0 <= v <= 1 for v in p['models'][b].values())
                       for b in generator.BACKENDS)
        # Fixed early noon contexts, not a post-hoc selection on predictions.
        for day, scenario in enumerate(data['model_meta']['scenarios'][:8]):
            state = points[24*day+12]['state']
            assert ''.join('H' if state[d] >= .5 else 'L' for d in 'HSE') == scenario['target_HSE']
        own = [e for e in data['events'] if e['fish_id'] == fish['id']]
        by_type = {kind: [e['minute'] for e in own if e['type'] == kind]
                   for kind in ('food_offered', 'food_consumed', 'disturbance')}
        assert Counter(m//1440 for m in by_type['food_offered']) == Counter(range(30))
        assert Counter(m//1440 for m in by_type['disturbance']) == Counter(range(30))
        assert all(any(0 <= t-offer <= 105 for offer in by_type['food_offered'])
                   for t in by_type['food_consumed'])
        assert len(by_type['food_consumed']) <= 60
        # Opportunities are not equivalent to eating; at least one opportunity
        # for this fish has no intake in its window.
        assert any(not any(0 <= t-offer <= 105 for t in by_type['food_consumed'])
                   for offer in by_type['food_offered'])
    assert len(set(paths)) == 3
    assert all(set(e) == {'minute', 'fish_id', 'type'} for e in data['events'])
    assert all(0 <= e['minute'] <= 43140 for e in data['events'])


def test_training_inputs_and_forward_only(built):
    data, observed = built
    assert len(observed) == 18
    forbidden = {'H', 'S', 'E', 'intake', 'food_consumed', 'food_offered',
                 'disturbance', 'scenario', 'future_response'}
    assert not forbidden.intersection(data['model_meta']['features'])
    assert len(set(data['model_meta']['features'])) == 11
    # Exported features are rounded for JSON; inference uses the unrounded arrays.
    full_x = [generator._simulate_fish(i, [])[1] for i in range(3)]
    for (backend, _), (model, train_x, train_y, rows) in list(observed.items())[:9]:
        assert train_x.shape == (1440, 11)
        assert len(train_y) == len(rows) == 1440
        assert set(train_y) == {0, 1}
        assert np.allclose(train_x, np.vstack([x[:480] for x in full_x]), atol=5e-7)
        assert [rows[i]['analysis'] for i in (0, 480, 960)] == [f['id'] for f in data['fish']]
        assert all(row['start_s'] <= 28740 and row['end_s'] <= 28800 for row in rows)
        for fish, x in zip(data['fish'], full_x):
            inference_rows = [{'analysis': fish['id'], 'start_s': p['minute'],
                               'end_s': p['minute']+60, 'quality_ok': True}
                              for p in fish['points']]
            p = probabilities(model, x, inference_rows)[:, list(model.classes_).index(1)]
            # Each estimator is shared between H/S/E fits but unique per backend/dimension.
            dimension = generator.DIMENSIONS[list(observed).index((backend, _)) % 3]
            assert np.allclose(p, [pt['models'][backend][dimension] for pt in fish['points']],
                               atol=5.01e-7)
            if backend == 'hmm':
                changed = x.copy()
                changed[501:] += 1000  # Later observations must not alter the past.
                shorter = probabilities(model, x[:501], inference_rows[:501])
                future_changed = probabilities(model, changed, inference_rows)
                assert np.allclose(shorter, future_changed[:501], atol=1e-12)
                # Filtering another fish first must not affect this fish's initial prior.
                assert np.allclose(probabilities(model, x[:1], inference_rows[:1]),
                                   probabilities(model, x, inference_rows)[:1])


def test_parameter_and_holdout_metrics_consistency(built):
    data, observed = built
    meta = data['model_meta']
    for backend in generator.BACKENDS:
        for dimension in 'HSE':
            index = generator.DIMENSIONS.index(dimension)
            model, _, train_y, _ = observed[(backend, 3*generator.BACKENDS.index(backend)+index)]
            params = meta['model_parameters'][backend][dimension]
            if backend == 'hmm':
                assert np.allclose(params['transition'], model.transition)
                assert np.allclose(params['standardized_means'], model.means)
                assert np.allclose(params['initial'], model.initial)
                assert np.allclose(params['standardized_variances'], model.variances)
                assert np.allclose(np.sum(params['transition'], axis=1), 1)
            else:
                assert np.allclose(params['coefficients'], model[-1].coef_)
                assert np.allclose(params['intercept'], model[-1].intercept_)
                if backend == 'gam':
                    assert all(np.allclose(knots, spline.t) for knots, spline in zip(
                        params['spline_knot_vectors'], model[2].bsplines_))
                    curves = params['partial_response_at_training_medians']
                    assert set(curves) == set(generator.FEATURES)
                    assert all(len(v['x']) == len(v['p_high']) == 32 and
                               all(0 <= p <= 1 for p in v['p_high'])
                               for v in curves.values())
            report = meta['holdout_metrics'][backend][dimension]
            prior = round(float(train_y.mean()), 6)
            sets = {'overall': [p for f in data['fish'] for p in f['points'][480:]]}
            sets.update({fid: next(f for f in data['fish'] if f['id'] == fid)['points'][480:]
                         for fid in report['fish']})
            for key, points in sets.items():
                metric = report['overall'] if key == 'overall' else report['fish'][key]
                y = np.asarray([int(p['state'][dimension] >= .5) for p in points])
                prob = np.asarray([p['models'][backend][dimension] for p in points])
                assert metric['points'] == len(points)
                assert metric['positive_fraction'] == round(float(y.mean()), 6)
                assert metric['accuracy_at_0_5'] == round(float(np.mean((prob >= .5) == y)), 6)
                assert metric['brier'] == round(float(np.mean((prob-y)**2)), 6)
                assert metric['baseline_train_prior'] == prior
                assert metric['baseline_majority_accuracy'] == round(float(np.mean((prior >= .5) == y)), 6)
                assert metric['baseline_prior_brier'] == round(float(np.mean((prior-y)**2)), 6)
            assert report['overall']['points'] == 720
            assert sum(report['fish'][fid]['points'] for fid in report['fish']) == 720
            for scope, points in [('overall', sets['overall'])] + [
                    (fid, sets[fid]) for fid in report['fish']]:
                strata = report['strata'] if scope == 'overall' else report['fish'][scope]['strata']
                assert {'label_0', 'label_1'} <= set(strata)
                assert strata['label_0']['points'] + strata['label_1']['points'] == len(points)
                for stratum, metric in strata.items():
                    selected = [p for p in points if (
                        int(p['state'][dimension] >= .5) == int(stratum[-1])
                        if stratum.startswith('label_') else
                        data['model_meta']['scenarios'][p['minute']//1440]['target_HSE'] ==
                        stratum.removeprefix('scenario_'))]
                    y = np.asarray([int(p['state'][dimension] >= .5) for p in selected])
                    prob = np.asarray([p['models'][backend][dimension] for p in selected])
                    assert metric['points'] == len(selected) > 0
                    assert metric['positive_fraction'] == round(float(y.mean()), 6)
                    assert metric['accuracy_at_0_5'] == round(float(np.mean((prob >= .5) == y)), 6)
                    assert metric['brier'] == round(float(np.mean((prob-y)**2)), 6)


def test_full_30_day_refits_are_separate_from_holdout(built):
    data, observed = built
    meta = data['model_meta']
    assert 'do NOT score the holdout' in meta['parameter_stages']
    for backend_index, backend in enumerate(generator.BACKENDS):
        for dim_index, dimension in enumerate(generator.DIMENSIONS):
            full_model, x, y, rows = observed[(backend, 9 + 3*backend_index + dim_index)]
            assert x.shape == (2160, 11)
            assert len(y) == len(rows) == 2160
            assert rows[0]['analysis'] == 'huaban'
            assert rows[720]['analysis'] == 'hongding'
            assert rows[1440]['analysis'] == 'lanyin'
            assert rows[-1]['start_s'] == 43140
            parameters = meta['full_30_day_parameters'][backend][dimension]
            if backend == 'hmm':
                assert np.allclose(parameters['transition'], full_model.transition)
                assert np.allclose(parameters['initial'], full_model.initial)
            else:
                assert np.allclose(parameters['coefficients'], full_model[-1].coef_)
                assert np.allclose(parameters['intercept'], full_model[-1].intercept_)
            # The displayed timeline still comes from the 20-day evaluation fit.
            evaluation_model = observed[(backend, 3*backend_index + dim_index)][0]
            fish = data['fish'][0]
            sample_x = generator._simulate_fish(0, [])[1]
            sample_rows = [{'analysis': fish['id'], 'start_s': p['minute'],
                            'end_s': p['minute']+60, 'quality_ok': True}
                           for p in fish['points']]
            evaluation_prob = probabilities(evaluation_model, sample_x, sample_rows)[:, 1]
            assert np.allclose(evaluation_prob,
                               [p['models'][backend][dimension] for p in fish['points']],
                               atol=5.01e-7)
