import joblib
import numpy as np
import pytest
from fishwatch.classical import fit_classical, probabilities, INPUTS


def dataset():
    rng = np.random.default_rng(23)
    y = np.tile(np.repeat([0, 1], 20), 4)
    x = rng.normal(size=(len(y), len(INPUTS))) + y[:, None]*2
    rows = [{'analysis': str(i//40), 'start_s': i%40, 'end_s': i%40+1} for i in range(len(y))]
    x[0, 0] = np.nan
    return x, y, rows


@pytest.mark.parametrize('backend', ['logistic', 'gam', 'hmm'])
def test_roundtrip_and_missing(backend, tmp_path):
    x, y, rows = dataset()
    estimator = fit_classical(backend, x, y, rows)
    p = probabilities(estimator, x, rows)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert (p.argmax(axis=1) == y).mean() > .9
    path = tmp_path/'model.joblib'; joblib.dump(estimator, path)
    np.testing.assert_allclose(p, probabilities(joblib.load(path), x, rows))


def test_hmm_causal_and_recording_reset():
    x, y, rows = dataset()
    estimator = fit_classical('hmm', x, y, rows)
    full = estimator.predict_proba(x, rows)
    np.testing.assert_allclose(full[:25], estimator.predict_proba(x[:25], rows[:25]))
    np.testing.assert_allclose(full[40:80], estimator.predict_proba(x[40:80], rows[40:80]))
    # Missing quality resets history for the next valid observation.
    rows[19]['quality_ok'] = False
    p = estimator.predict_proba(x, rows)
    np.testing.assert_allclose(p[20], estimator.predict_proba(x[20:21], rows[20:21])[0])


def test_hmm_does_not_learn_transitions_across_gaps():
    x, y, rows = dataset()
    rows = [dict(r, analysis=str(i)) for i, r in enumerate(rows)]
    estimator = fit_classical('hmm', x, y, rows)
    np.testing.assert_allclose(estimator.transition, .5)
