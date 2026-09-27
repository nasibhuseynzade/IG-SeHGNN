# IG-SeHGNN

**Information-Gain-based metapath selection for SeHGNN** (TAD-MPS: Task-aware, Automatic-Depth Meta-Path Selection).

[SeHGNN](https://github.com/ICT-GIMLab/SeHGNN) (AAAI 2023) pre-computes one feature channel per metapath and fuses
all channels with a transformer. It uses every metapath up to a hand-chosen hop budget, and the cost of its semantic
fusion grows quadratically with the number of metapaths.

This repository adds a **metapath selection step in front of the unchanged SeHGNN**:

1. **Candidate generation** – all feature and label metapath channels up to a hop budget `L` are pre-computed with
   SeHGNN's own propagation.
2. **Task-aware selection** – each feature metapath is scored by its *conditional information gain* about the labels,
   `gain(P | S) = CE(S) − CE(S ∪ {P}) ≈ I(X^P; Y | X^S)`, estimated with a small bag-of-channels probe on cross-validation
   folds of the training/validation nodes. Metapaths are added greedily until the gain vanishes, so redundant metapaths
   are skipped and the metapath depth is chosen automatically.
3. **Training** – the original SeHGNN is trained on the selected feature channels plus all label channels, with the
   same hyper-parameters.

The information-gain idea is inspired by
[MP-GNN (Ferrini et al., 2023)](https://arxiv.org/abs/2309.17113).

## Repository structure

```
IG-SeHGNN/
├── model.py               # SeHGNN model (unchanged)
├── utils.py               # data loading, metapath propagation, training step
├── main.py                # original SeHGNN training
├── main_A.py              # SeHGNN + metapath selection (proposed model)
├── tadmps/
│   ├── probe.py           # bag-of-channels probe, cross-validated score
│   └── selector.py        # greedy information-gain selection, stopping rule
├── ablation_eps.py        # ablation: stopping threshold / minimum size
├── ablation_criterion.py  # ablation: selection criteria
└── data/
    └── data_loader.py     # HGB data loader
```

## Requirements

Tested with Python 3.11, PyTorch 2.2.1, DGL 2.2.0, torch-sparse 0.6.18, scikit-learn 1.5.1 (CPU).

```bash
pip install -r requirements.txt
```

`torch-sparse` and `dgl` must match your PyTorch (and CUDA) version; see the
[PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and
[DGL](https://www.dgl.ai/pages/start.html) installation guides.

## Data

Download `DBLP.zip`, `ACM.zip` and `IMDB.zip` from the [HGB repository](https://github.com/THUDM/HGB) and extract them
into `data/`:

```
data/
├── data_loader.py
├── DBLP/   (node.dat, link.dat, label.dat, label.dat.test, info.dat)
├── ACM/
└── IMDB/
```

A different location can be given with `--root /path/to/data`.

## Usage

All commands can be run from the repository root. Checkpoints are written to `output/`.

### Original SeHGNN

```bash
# DBLP
python main.py --dataset DBLP --epoch 200 --n-fp-layers 2 --n-task-layers 3 --num-hops 2 --num-label-hops 4 \
    --label-feats --residual --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 --seeds 1 2 3 4 5

# ACM
python main.py --dataset ACM --epoch 200 --n-fp-layers 2 --n-task-layers 1 --num-hops 4 --num-label-hops 4 \
    --label-feats --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 --seeds 1 2 3 4 5

# IMDB
python main.py --dataset IMDB --epoch 200 --n-fp-layers 2 --n-task-layers 4 --num-hops 4 --num-label-hops 4 \
    --label-feats --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0. --seeds 1 2 3 4 5
```

### SeHGNN with metapath selection (proposed)

Same SeHGNN hyper-parameters, a hop-4 candidate pool, and `--select tadmps`:

```bash
# DBLP
python main_A.py --dataset DBLP --epoch 200 --n-fp-layers 2 --n-task-layers 3 --num-hops 4 --num-label-hops 4 \
    --label-feats --residual --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 --seeds 1 2 3 4 5 \
    --select tadmps

# ACM
python main_A.py --dataset ACM --epoch 200 --n-fp-layers 2 --n-task-layers 1 --num-hops 4 --num-label-hops 4 \
    --label-feats --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0.5 --seeds 1 2 3 4 5 \
    --select tadmps

# IMDB
python main_A.py --dataset IMDB --epoch 200 --n-fp-layers 2 --n-task-layers 4 --num-hops 4 --num-label-hops 4 \
    --label-feats --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0. --seeds 1 2 3 4 5 \
    --select tadmps
```

`main_A.py --select none` runs the original SeHGNN with the same pipeline. The selection is done once, before the seed
loop, and does not depend on the training seed.

At the end of a run the script prints the number of channels, the number of parameters, the average training time per
epoch and the mean / standard deviation of the test Micro-F1 and Macro-F1 over the seeds.

### Selection options (`main_A.py`)

| Option | Default | Meaning |
|---|---|---|
| `--select` | `none` | `none` = original SeHGNN, `tadmps` = with metapath selection |
| `--select-metric` | `ce` | `ce` = information gain (cross-entropy reduction), `f1` = F1 gain |
| `--probe` | `linear` | probe type: `linear` or `mlp` |
| `--probe-hidden` | `128` | hidden size of the `mlp` probe |
| `--probe-steps` | `100` | optimisation steps per probe |
| `--probe-seed` | `0` | seed of the probe initialisation |
| `--n-folds` | `5` | cross-validation folds |
| `--fold-seed` | `0` | seed of the fold split |
| `--sel-eps` | `0.001` | minimum score improvement (nats) |
| `--sel-patience` | `2` | steps without improvement before stopping |
| `--min-select` | `3` | minimum number of selected metapaths |
| `--max-select` | `12` | maximum number of selected metapaths |

### Ablations

```bash
python ablation_eps.py DBLP ACM IMDB     # stopping threshold and minimum size
python ablation_criterion.py DBLP        # selection criteria (also: IMDB)
```

Results are printed and saved as JSON in the working directory.

## Notes

- Only **feature** metapaths are selected; label metapaths are always kept. The target node's own features are always
  kept.
- In the SeHGNN code, ACM subject nodes are labelled `C` and term nodes `K` (term nodes are not used, as in the original
  SeHGNN setting). A selected key such as `PCP` therefore means paper–subject–paper.
- The selection runs on CPU and its cost grows with the number of candidate metapaths, folds and probe steps.

## Acknowledgements

The SeHGNN model, data loading and metapath propagation code come from the official SeHGNN implementation:
<https://github.com/ICT-GIMLab/SeHGNN>. Please also cite the original work:

```bibtex
@inproceedings{yang2023simple,
  title     = {Simple and Efficient Heterogeneous Graph Neural Network},
  author    = {Yang, Xiaocheng and Yan, Mingyu and Pan, Shirui and Ye, Xiaochun and Fan, Dongrui},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages     = {10816--10824},
  year      = {2023}
}
```
