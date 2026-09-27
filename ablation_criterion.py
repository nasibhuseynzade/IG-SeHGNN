import contextlib
import io
import json
import sys
import time

import numpy as np
import torch

import main_A
import tadmps.selector as selector
from tadmps.probe import score_cv
from ablation_eps import make_args, train_fixed, PROBE_KW, TGT

# metapath budgets
BUDGETS = [2, 4]
N_RANDOM = 5
CKA_NODES = 2000


# candidate pool
def load_pool(ds):
    args = make_args(ds)
    with contextlib.redirect_stdout(io.StringIO()):
        g, adjs, init_labels, num_classes, dl, trainval_nid, test_nid = main_A.load_dataset(args)
        g = main_A.hg_propagate_feat_dgl(g, TGT[ds], args.num_hops, args.num_hops + 1, [], echo=False)
    feats = {k: g.nodes[TGT[ds]].data.pop(k) for k in list(g.nodes[TGT[ds]].data.keys())}
    folds = selector.make_folds(trainval_nid, n_folds=5, seed=0)
    return feats, init_labels.long(), folds, num_classes


# conditional greedy
def greedy(feats, cands, y, folds, nc, K, metric):
    torch.manual_seed(0); np.random.seed(0)
    chosen, remaining = [], list(cands)
    kw = dict(PROBE_KW, metric=metric)
    for _ in range(K):
        scores = {k: score_cv([feats[q] for q in chosen] + [feats[k]], y, folds, nc, **kw) for k in remaining}
        best = max(scores, key=scores.get)
        chosen.append(best); remaining.remove(best)
    return chosen


# unconditional ranking
def unconditional(feats, cands, y, folds, nc, K):
    torch.manual_seed(0); np.random.seed(0)
    scores = {k: score_cv([feats[k]], y, folds, nc, **PROBE_KW) for k in cands}
    return sorted(scores, key=scores.get, reverse=True)[:K], scores


# label-free cka
def cka_farthest(feats, cands, tgt, K):
    rng = np.random.RandomState(0)
    n = next(iter(feats.values())).shape[0]
    idx = torch.as_tensor(rng.choice(n, size=min(CKA_NODES, n), replace=False))
    H = torch.eye(len(idx)) - 1.0 / len(idx)

    def gram(x):
        x = x[idx].float()
        return H @ (x @ x.t()) @ H

    grams = {k: gram(feats[k]) for k in [tgt] + list(cands)}
    norms = {k: torch.linalg.norm(G) for k, G in grams.items()}
    cka = lambda a, b: float((grams[a] * grams[b]).sum() / (norms[a] * norms[b] + 1e-12))
    selected, remaining = [tgt], list(cands)
    for _ in range(K):
        best = min(remaining, key=lambda c: max(cka(c, s) for s in selected))
        selected.append(best); remaining.remove(best)
    return selected[1:]


# criterion ablation
def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else 'DBLP'
    tgt = TGT[ds]
    feats, y, folds, nc = load_pool(ds)
    cands = [k for k in feats if k != tgt]
    Kmax = max(BUDGETS)
    print(f'[{ds}] pool={len(cands)} candidates (+ target {tgt})', flush=True)

    sel, sel_time = {}, {}
    t = time.time(); path = greedy(feats, cands, y, folds, nc, Kmax, 'ce'); sel_time['ours'] = time.time() - t
    print(f'[{ds}] ours path   {path}  ({sel_time["ours"]:.0f}s)', flush=True)
    t = time.time(); f1path = greedy(feats, cands, y, folds, nc, Kmax, 'f1'); sel_time['f1'] = time.time() - t
    print(f'[{ds}] f1 path     {f1path}  ({sel_time["f1"]:.0f}s)', flush=True)
    t = time.time(); _, uscores = unconditional(feats, cands, y, folds, nc, Kmax); sel_time['uncond'] = time.time() - t
    uorder = sorted(uscores, key=uscores.get, reverse=True)
    print(f'[{ds}] uncond rank {uorder[:6]}  ({sel_time["uncond"]:.0f}s)', flush=True)
    t = time.time(); ckapath = cka_farthest(feats, cands, tgt, Kmax); sel_time['cka'] = time.time() - t
    print(f'[{ds}] cka path    {ckapath}  ({sel_time["cka"]:.0f}s)', flush=True)
    shortest = sorted(cands, key=lambda k: (len(k), cands.index(k)))

    rng = np.random.RandomState(0)
    for K in BUDGETS:
        sel[('ours', K)] = path[:K]
        sel[('f1', K)] = f1path[:K]
        sel[('uncond', K)] = uorder[:K]
        sel[('cka', K)] = ckapath[:K]
        sel[('shortest', K)] = shortest[:K]
        for r in range(N_RANDOM):
            sel[(f'random{r}', K)] = list(rng.choice(cands, size=K, replace=False))

    cache, rows = {}, []
    for (method, K), keys in sel.items():
        sig = tuple(sorted(keys))
        if sig not in cache:
            print(f'[{ds}] train {method:9s} K={K}  {keys}', flush=True)
            cache[sig] = train_fixed(ds, keys)
            print(f'[{ds}]   {cache[sig]}', flush=True)
        rows.append(dict(method=method, K=K, selected=list(keys), **cache[sig]))
        with open(f'ablation_criterion_{ds}.json', 'w') as f:
            json.dump(dict(rows=rows, sel_time=sel_time, uncond_scores=uscores), f, indent=1)

    print(f'\n===== {ds}: selection-criterion ablation (target {tgt} always added) =====')
    for K in BUDGETS:
        print(f'--- budget K={K} ---')
        for m in ['ours', 'uncond', 'f1', 'cka', 'shortest']:
            r = next(r for r in rows if r['method'] == m and r['K'] == K)
            print(f"{m:9s} k={r['k']:2d} {r['params']:>11,} {r['s_epoch']:.3f}s  "
                  f"{r['micro']:.2f} / {r['macro']:.2f}  {r['selected']}")
        rr = [r for r in rows if r['method'].startswith('random') and r['K'] == K]
        mi, ma = np.array([r['micro'] for r in rr]), np.array([r['macro'] for r in rr])
        print(f"random    ({N_RANDOM} draws)      {mi.mean():.2f}+/-{mi.std():.2f} / {ma.mean():.2f}+/-{ma.std():.2f}  "
              f"min {mi.min():.2f} max {mi.max():.2f}")
    print('selection time (s):', {k: round(v, 1) for k, v in sel_time.items()})


if __name__ == '__main__':
    main()
