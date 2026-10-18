"""Shared deterministic metrics and fixed non-neural controls, CPU only."""
import numpy as np

DISCOUNTS = 1.0 / np.log2(np.arange(2, 12))

def ranking(scores, k=10):
    scores = np.asarray(scores)
    assert scores.ndim == 2 and np.isfinite(scores).all()
    return np.argsort(-scores, axis=1, kind='stable')[:, :k]

def metrics(top, targets):
    assert len(top) == len(targets)
    result = np.empty((len(top), 2), np.float64)
    for i, (pred, target) in enumerate(zip(top, targets)):
        truth = set(map(int, target))
        assert truth and len(set(pred)) == len(pred)
        hit = np.asarray([int(p) in truth for p in pred], dtype=np.float64)
        result[i] = (hit @ DISCOUNTS[:len(pred)] / DISCOUNTS[:min(len(truth), 10)].sum(),
                     hit.sum() / len(truth))
    return result

def target_sets(q, indices):
    return [q['bank_members'][q['target_block'][i], :q['bank_size'][q['target_block'][i]]]
            for i in indices]

def cohort_groups(q, indices):
    groups = []
    for i in indices:
        history = set()
        for b in q['history_blocks'][i, :q['length'][i]]:
            history.update(q['bank_members'][b, :q['bank_size'][b]])
        target = set(q['bank_members'][q['target_block'][i], :q['bank_size'][q['target_block'][i]]])
        groups.append('repeat_only' if target <= history else 'unseen_only' if not target & history else 'mixed')
    return np.asarray(groups)

def fit_counts(q, n_codes):
    pop = np.zeros(n_codes, np.float64)
    transition = np.zeros((n_codes, n_codes), np.float64)
    for i in q['fit']:
        t = q['target_block'][i]
        targets = q['bank_members'][t, :q['bank_size'][t]]
        b = q['history_blocks'][i, q['length'][i] - 1]
        sources = q['bank_members'][b, :q['bank_size'][b]]
        weight = q['prefix_weight'][i]
        pop[targets] += weight / len(targets)
        transition[np.ix_(sources, targets)] += weight / (len(sources) * len(targets))
    assert np.isclose(pop.sum(), len(np.unique(q['pid'][q['fit']])))
    pop /= pop.sum()
    transition += pop[None, :]
    transition /= transition.sum(1, keepdims=True)
    return pop, transition

def count_scores(q, indices, name, pop, transition):
    scores = np.tile(pop, (len(indices), 1))
    if name == 'Popularity':
        return scores
    for row, i in enumerate(indices):
        blocks = q['history_blocks'][i, :q['length'][i]]
        if name == 'QuarterTransition':
            b = blocks[-1]
            codes = q['bank_members'][b, :q['bank_size'][b]]
            scores[row] = transition[codes].mean(0)
        elif name == 'PersonalRecency':
            personal = np.zeros(len(pop), np.float64)
            for age, b in enumerate(blocks[::-1]):
                codes = q['bank_members'][b, :q['bank_size'][b]]
                personal[codes] += .8 ** age / len(codes)
            scores[row] = .8 * personal / personal.sum() + .2 * pop
        else:
            raise ValueError(name)
    return scores

def neighbour_votes(neighbours, similarities, q, pop):
    scores = np.zeros((len(neighbours), len(pop)), np.float64)
    for row, (indices, sims) in enumerate(zip(neighbours, similarities)):
        for i, sim in zip(indices, sims):
            b = q['target_block'][i]
            codes = q['bank_members'][b, :q['bank_size'][b]]
            scores[row, codes] += max(float(sim), 0.) * q['prefix_weight'][i] / len(codes)
        total = scores[row].sum()
        scores[row] = scores[row] / total if total > 0 else pop
    return scores

def cohort_contrast(neural_metrics, control_metrics, groups, seed=20261018, draws=5000):
    difference = np.asarray(neural_metrics) - np.asarray(control_metrics)
    unseen = np.asarray(groups) != 'repeat_only'
    assert difference.ndim == 1 and unseen.any()
    observed = np.asarray([difference.mean(), difference[unseen].mean()])
    observed = np.r_[observed, observed[1] - observed[0]]
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, 3), np.float64)
    for j in range(draws):
        ix = rng.integers(0, len(difference), len(difference))
        selected = unseen[ix]
        if not selected.any():
            raise ValueError('Bootstrap draw has no unseen-positive resumes')
        a = difference[ix].mean()
        u = difference[ix][selected].mean()
        boot[j] = (a, u, u-a)
    radius = np.quantile(np.max(np.abs(boot-observed), axis=1), .95)
    intervals = np.c_[observed-radius, observed+radius]
    reversal = bool((intervals[0, 1] < -.01 and intervals[1, 0] > .01)
                    or (intervals[1, 1] < -.01 and intervals[0, 0] > .01))
    shift = bool(intervals[2, 0] > .01 or intervals[2, 1] < -.01)
    return {'estimates': dict(zip(['G_all', 'G_unseen_positive', 'Delta_cohort'], observed.tolist())),
            'simultaneous_95_intervals': dict(zip(['G_all', 'G_unseen_positive', 'Delta_cohort'], intervals.tolist())),
            'bootstrap_draws': draws, 'bootstrap_seed': seed, 'simultaneous_radius': float(radius),
            'material_rank_reversal_supported': reversal, 'material_contrast_shift_supported': shift}
