"""Small supervised models; HMM inference uses forward filtering, never future data."""
import numpy as np
from scipy.special import logsumexp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from .features import FEATURES

# Compact behavior-only inputs. No post-feeding outcomes are input features.
INPUTS = ['speed_mean', 'acceleration_mean', 'displacement_ratio', 'turn_rate',
          'feeding_zone_fraction', 'surface_fraction', 'dispersion',
          'activity_fraction', 'wall_fraction', 'shelter_fraction',
          'speed_baseline_z', 'feeding_baseline_z']


def vectors(sequences):
    return sequences[:, -1, [FEATURES.index(f) for f in INPUTS]]


def contiguous(a, b):
    return a['analysis'] == b['analysis'] and abs(a['end_s']-b['start_s']) < 1e-4


class BehavioralHMM:
    """Label-anchored Gaussian HMM with diagonal emissions and Laplace transitions."""
    def fit(self, x, y, rows):
        good = y >= 0
        self.preprocess = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler())
        z = self.preprocess.fit_transform(x[good])
        labels = y[good]
        self.classes_ = np.unique(labels)
        k = len(self.classes_)
        self.means = np.array([z[labels == c].mean(axis=0) for c in self.classes_])
        # Shrink per-class variance toward global variance; keep constant columns finite.
        global_var = np.maximum(z.var(axis=0), 0.05)
        self.variances = np.array([np.maximum(.9*z[labels == c].var(axis=0)+.1*global_var, .05) for c in self.classes_])
        transitions, starts = np.ones((k, k)), np.ones(k)
        for i, label in enumerate(y):
            if label < 0:
                continue
            if i and y[i-1] >= 0 and contiguous(rows[i-1], rows[i]):
                transitions[y[i-1], label] += 1
            else:
                starts[label] += 1
        self.transition = transitions/transitions.sum(axis=1, keepdims=True)
        self.initial = starts/starts.sum()
        return self

    def predict_proba(self, x, rows):
        z = self.preprocess.transform(x)
        emissions = -.5*np.sum(np.log(2*np.pi*self.variances)[None] +
                    (z[:, None]-self.means[None])**2/self.variances[None], axis=2)
        result, posterior = [], None
        for i, emission in enumerate(emissions):
            reset = i == 0 or not contiguous(rows[i-1], rows[i]) or not rows[i-1].get('quality_ok', True)
            prior = self.initial if reset else posterior @ self.transition
            if not rows[i].get('quality_ok', True):
                posterior = prior
            else:
                logp = np.log(np.maximum(prior, 1e-300)) + emission
                posterior = np.exp(logp-logsumexp(logp))
            result.append(posterior.copy())
        return np.asarray(result)


def fit_classical(backend, x, y, rows):
    if backend == 'hmm':
        return BehavioralHMM().fit(x, y, rows)
    steps = [SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()]
    if backend == 'gam':
        # Additive univariate cubic spline bases; L2 coefficient penalty (not a curvature penalty).
        steps.append(SplineTransformer(n_knots=4, degree=3, include_bias=False, extrapolation='constant'))
    steps.append(LogisticRegression(C=1.0, max_iter=3000))
    estimator = make_pipeline(*steps)
    valid = y >= 0
    return estimator.fit(x[valid], y[valid])


def probabilities(estimator, x, rows):
    return estimator.predict_proba(x, rows) if isinstance(estimator, BehavioralHMM) else estimator.predict_proba(x)


def explain(estimator, backend, training_x):
    result = {'inputs': INPUTS, 'backend': backend, 'interpretation': '关联而非因果；所有参数仅在训练集拟合'}
    if backend == 'hmm':
        result.update(initial=estimator.initial.tolist(), transition=estimator.transition.tolist(),
                      standardized_means=estimator.means.tolist(), standardized_variances=estimator.variances.tolist())
    else:
        clf = estimator[-1]
        result.update(classes=clf.classes_.tolist(), coefficients=clf.coef_.tolist(), intercept=clf.intercept_.tolist())
        if backend == 'gam':
            # Partial response curves at training medians, explicitly not causal effects.
            base = estimator[0].transform(training_x)
            median = np.median(base, axis=0)
            curves = {}
            for j, name in enumerate(INPUTS):
                grid = np.linspace(*np.quantile(base[:, j], [.05, .95]), 40)
                x = np.tile(median, (len(grid), 1)); x[:, j] = grid
                curves[name] = {'x': grid.tolist(), 'probabilities': estimator.predict_proba(x).tolist()}
            result['partial_response_at_training_medians_uncalibrated'] = curves
    return result
