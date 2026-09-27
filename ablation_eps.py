import contextlib
import io
import json
import os
import sys

import numpy as np
import torch

import main_A
import tadmps.selector as selector
from tadmps.probe import score_cv

# ablation grid
EPS_GRID = [0.0005, 0.001, 0.005, 0.01]
MIN_GRID = [1, 3]
PATIENCE, MAX_SEL = 2, 12
PROBE_KW = dict(probe='linear', hidden=128, steps=100, metric='ce')

TGT = {'DBLP': 'A', 'ACM': 'P', 'IMDB': 'M'}
# dataset configs
ARGV = {
    'DBLP': ['--dataset', 'DBLP', '--num-hops', '4', '--num-label-hops', '4', '--label-feats', '--residual',
             '--n-fp-layers', '2', '--n-task-layers', '3', '--hidden', '512', '--embed-size', '512',
             '--dropout', '0.5', '--input-drop', '0.5', '--epoch', '200', '--seeds', '1'],
    'ACM':  ['--dataset', 'ACM', '--num-hops', '4', '--num-label-hops', '4', '--label-feats',
             '--n-fp-layers', '2', '--n-task-layers', '1', '--hidden', '512', '--embed-size', '512',
             '--dropout', '0.5', '--input-drop', '0.5', '--epoch', '200', '--seeds', '1'],
    'IMDB': ['--dataset', 'IMDB', '--num-hops', '4', '--num-label-hops', '4', '--label-feats',
             '--n-fp-layers', '2', '--n-task-layers', '4', '--hidden', '512', '--embed-size', '512',
             '--dropout', '0.5', '--input-drop', '0.', '--epoch', '200', '--seeds', '1'],
}


def make_args(ds, extra=()):
    args = main_A.parse_args(ARGV[ds] + list(extra))
    if ds == 'ACM':
        args.ACM_keep_F = False
    return args


# full greedy path
def full_path(ds):
    args = make_args(ds)
    with contextlib.redirect_stdout(io.StringIO()):
        g, adjs, init_labels, num_classes, dl, trainval_nid, test_nid = main_A.load_dataset(args)
        g = main_A.hg_propagate_feat_dgl(g, TGT[ds], args.num_hops, args.num_hops + 1, [], echo=False)
    feats = {k: g.nodes[TGT[ds]].data.pop(k) for k in list(g.nodes[TGT[ds]].data.keys())}
    y = init_labels.long()
    folds = selector.make_folds(trainval_nid, n_folds=5, seed=0)

    torch.manual_seed(0); np.random.seed(0)
    empty = score_cv([], y, folds, num_classes, **PROBE_KW)
    remaining, chosen, path = list(feats.keys()), [], []
    print(f'[{ds}] pool={len(remaining)}  empty={empty:.4f}', flush=True)
    while remaining and len(chosen) < MAX_SEL:
        best_k, best_s = None, -1e9
        for k in remaining:
            s = score_cv([feats[q] for q in chosen] + [feats[k]], y, folds, num_classes, **PROBE_KW)
            if s > best_s:
                best_s, best_k = s, k
        chosen.append(best_k); remaining.remove(best_k); path.append((best_k, best_s))
        prev = path[-2][1] if len(path) > 1 else empty
        print(f'[{ds}]  + {best_k:8s} score={best_s:.4f}  gain={best_s - prev:+.4f}', flush=True)
    return empty, path


# stopping-rule replay
def replay(empty, path, eps, m):
    best_score, best_len, no_improve = empty, 0, 0
    for i, (_, s) in enumerate(path):
        n = i + 1
        if s > best_score + eps:
            best_score, best_len, no_improve = s, n, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE and n >= m:
                break
    return [k for k, _ in path[:max(best_len, m)]]


# train fixed set
def train_fixed(ds, keys):
    selector.greedy_select = lambda *a, **kw: (list(keys), [])
    for lst in (main_A.RESULTS, main_A.TIMES, main_A.NPARAMS, main_A.NCHAN):
        lst.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        main_A.main(make_args(ds, ['--select', 'tadmps']))
    r = np.array(main_A.RESULTS[0]) * 100
    return dict(k=main_A.NCHAN[0], params=main_A.NPARAMS[0], s_epoch=float(np.mean(main_A.TIMES)),
                micro=float(r[0]), macro=float(r[1]))


# epsilon/min ablation
def main():
    datasets = sys.argv[1:] or ['DBLP', 'ACM', 'IMDB']
    out = {}
    for ds in datasets:
        empty, path = full_path(ds)
        configs = {(e, m): replay(empty, path, e, m) for e in EPS_GRID for m in MIN_GRID}
        trained = {}
        for (e, m), keys in configs.items():
            sig = tuple(keys)
            if sig not in trained:
                print(f'[{ds}] training eps={e} m={m} -> {len(keys)} selected {keys}', flush=True)
                trained[sig] = train_fixed(ds, keys)
                print(f'[{ds}]   {trained[sig]}', flush=True)
        out[ds] = dict(empty=empty, path=path,
                       rows=[dict(eps=e, m=m, selected=keys, **trained[tuple(keys)])
                             for (e, m), keys in configs.items()])
        with open('ablation_eps_results.json', 'w') as f:
            json.dump(out, f, indent=1)

    for ds, d in out.items():
        print(f'\n===== {ds}  (target self-channel {TGT[ds]} is always added) =====')
        print(f"{'eps':>7} {'m':>2} {'#sel':>4} {'k':>3} {'params':>11} {'s/ep':>6} {'micro':>6} {'macro':>6}  selected")
        for r in d['rows']:
            print(f"{r['eps']:>7} {r['m']:>2} {len(r['selected']):>4} {r['k']:>3} {r['params']:>11,} "
                  f"{r['s_epoch']:>6.3f} {r['micro']:>6.2f} {r['macro']:>6.2f}  {r['selected']}")


if __name__ == '__main__':
    main()
