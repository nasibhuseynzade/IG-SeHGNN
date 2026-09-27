from collections import Counter

import numpy as np
import torch

from tadmps.probe import score_cv


# fixed cv folds
def make_folds(idx, n_folds=5, seed=0):
    idx = np.array(idx).copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    parts = np.array_split(idx, n_folds)
    folds = []
    for i in range(n_folds):
        va = parts[i]
        tr = np.concatenate([parts[j] for j in range(n_folds) if j != i])
        folds.append((torch.LongTensor(np.sort(tr)), torch.LongTensor(np.sort(va))))
    return folds


def hops(key):
    return len(key) - 1


def depth_summary(sel):
    return dict(sorted(Counter(hops(k) for k in sel).items()))


# greedy info-gain selection
def greedy_select(channels, y, folds, num_classes,
                  min_select=3, max_select=12, eps=0.001, patience=2,
                  verbose=True, **probe_kw):
    remaining = list(channels.keys())
    selected, path = [], []

    cur = score_cv([], y, folds, num_classes, **probe_kw)  # empty-set score
    best_score, best_len, no_improve = cur, 0, 0
    if verbose:
        print(f"  <empty>         score={cur:.4f}")

    while remaining and len(selected) < max_select:
        best_k, best_s = None, float('-inf')  # best candidate
        for k in remaining:
            xs = [channels[q] for q in selected] + [channels[k]]
            s = score_cv(xs, y, folds, num_classes, **probe_kw)
            if s > best_s:
                best_s, best_k = s, k

        selected.append(best_k)
        remaining.remove(best_k)
        path.append((best_k, best_s))
        if verbose:
            print(f"  + {best_k:14s} score={best_s:.4f}  gain={best_s - cur:+.4f}")
        cur = best_s

        # stopping rule
        if best_s > best_score + eps:
            best_score, best_len, no_improve = best_s, len(selected), 0
        else:
            no_improve += 1
            if no_improve >= patience and len(selected) >= min_select:
                break

    keep = max(best_len, min_select)  # best-cv prefix
    final = selected[:keep]
    if verbose:
        print(f"  -> keep best-CV prefix of {keep}: {final}  (cv={best_score:.4f})")
    return final, path
