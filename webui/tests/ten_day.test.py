"""Contract and causal checks for the new ten-day synthetic hourly export."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts import build_ten_day_data as generator


@pytest.fixture(scope='module')
def data():
    generated = generator.build_data()
    output = generator.DEFAULT_OUTPUT
    assert json.loads(output.read_text(encoding='utf-8')) == generated
    assert output.stat().st_size < 500_000
    # Repeat the entire simulation and fit, not merely the seeded RNG.
    assert generator.build_data() == generated
    return generated


def test_schema_bounds_and_distinct_trajectories(data):
    assert {k: data[k] for k in ('domain', 'kind', 'seed', 'step_minutes', 'days',
                                'model_origin', 'threshold')} == {
        'domain': 'synthetic', 'kind': 'ten_day_three_fish', 'seed': 73,
        'step_minutes': 60, 'days': 10,
        'model_origin': 'newly_fitted_hourly_synthetic_features', 'threshold': .7}
    assert [f['id'] for f in data['fish']] == ['huaban', 'hongding', 'lanyin']
    paths = []
    for fish in data['fish']:
        points = fish['points']
        assert len(points) == 240
        assert [p['minute'] for p in points] == list(range(0, 14341, 60))
        path = tuple(tuple(p['position']) for p in points)
        paths.append(path)
        for p in points:
            assert set(p) == {'minute', 'state', 'action', 'position', 'models', 'quality_ok'}
            assert set(p['state']) == set('HSE')
            assert all(0 <= p['state'][dim] <= 1 for dim in 'HSE')
            assert p['action'] in range(5) and p['quality_ok'] is True
            assert len(p['position']) == 3
            assert all(lo <= v <= hi for v, lo, hi in zip(p['position'],
                       (-38, 11, -22), (38, 39, 24)))
            assert set(p['models']) == {'logistic', 'gam', 'hmm'}
            for model in p['models'].values():
                assert set(model) == set('HSE')
                assert all(0 <= value <= 1 for value in model.values())
    assert len(set(paths)) == 3
    assert 'old' in data['model_meta']['simulation']
    assert 'not real fish' in data['model_meta']['notice']


def test_events_and_state_causality(data):
    events = data['events']
    assert all(e['type'] in ('food_offered', 'food_consumed', 'disturbance')
               and e['fish_id'] in {f['id'] for f in data['fish']}
               and 0 <= e['minute'] <= 14340 for e in events)
    for fish in data['fish']:
        fid = fish['id']
        own = [e for e in events if e['fish_id'] == fid]
        by_type = {t: [e['minute'] for e in own if e['type'] == t]
                   for t in ('food_offered', 'food_consumed', 'disturbance')}
        assert Counter(t//1440 for t in by_type['food_offered']) == Counter(range(10))
        assert Counter(t//1440 for t in by_type['disturbance']) == Counter(range(10))
        assert by_type['food_consumed']
        for t in by_type['food_consumed']:
            assert any(0 <= t-offer <= 100 for offer in by_type['food_offered'])
        points = fish['points']
        assert any(abs(p['state']['E'] - points[0]['state']['E']) > .03 for p in points)
        for before, after in zip(points, points[1:]):
            start, end = before['minute'], after['minute']
            ate = any(start < t <= end for t in by_type['food_consumed'])
            disturbed = any(start < t <= end for t in by_type['disturbance'])
            if not ate:
                assert after['state']['H'] + 1e-5 >= before['state']['H']
            if after['state']['H'] < before['state']['H'] - 1e-5:
                assert ate
            if disturbed:
                assert after['state']['S'] > before['state']['S']
            elif before['state']['S'] > .01:
                assert after['state']['S'] <= before['state']['S'] + 1e-5


def test_temporal_fit_boundary_and_feature_contract(monkeypatch, data):
    observed = []
    original = generator.fit_classical

    def capture(backend, x, y, rows):
        observed.append((backend, x.copy(), y.copy(), [row.copy() for row in rows]))
        return original(backend, x, y, rows)

    monkeypatch.setattr(generator, 'fit_classical', capture)
    assert generator.build_data() == data
    assert len(observed) == 9
    assert data['model_meta']['train_minutes'] == [0, 10020]
    assert data['model_meta']['holdout_minutes'] == [10080, 14340]
    assert all(name not in ('H', 'S', 'E', 'intake', 'disturbance')
               for name in data['model_meta']['features'])
    for backend, x, y, rows in observed:
        assert backend in ('logistic', 'gam', 'hmm')
        assert x.shape == (3*168, len(generator.FEATURES))
        assert len(y) == len(rows) == 3*168
        assert all(row['start_s'] <= 10020 and row['end_s'] <= 10080 for row in rows)
        assert [rows[i]['analysis'] for i in (0, 168, 336)] == [f['id'] for f in data['fish']]
        assert set(y) == {0, 1}
    assert all(v['holdout_points'] == 216 for backend in data['model_meta']['holdout_metrics'].values()
               for v in backend.values())
