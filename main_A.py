import os
import gc
import time
import uuid
import argparse
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse import remove_diag, set_diag

from model import *
from utils import *


def main(args):
    g, adjs, init_labels, num_classes, dl, trainval_nid, test_nid = load_dataset(args)

    # row-normalized adjacency
    for k in adjs.keys():
        adjs[k].storage._value = None
        adjs[k].storage._value = torch.ones(adjs[k].nnz()) / adjs[k].sum(dim=-1)[adjs[k].storage.row()]

    # target node types
    if args.dataset == 'DBLP':
        tgt_type = 'A'; node_types = ['A', 'P', 'T', 'V']; extra_metapath = []
    elif args.dataset == 'ACM':
        tgt_type = 'P'; node_types = ['P', 'A', 'C']; extra_metapath = []
    elif args.dataset == 'IMDB':
        tgt_type = 'M'; node_types = ['M', 'A', 'D', 'K']; extra_metapath = []
    else:
        assert 0
    extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_hops + 1]

    print(f'Current num hops = {args.num_hops} for feature propagation')
    prop_device = 'cpu'; store_device = 'cpu'

    prop_tic = datetime.datetime.now()
    if len(extra_metapath):
        max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
    else:
        max_length = args.num_hops + 1

    # feature propagation
    g = hg_propagate_feat_dgl(g, tgt_type, args.num_hops, max_length, extra_metapath, echo=True)
    raw_feats = {}
    for k in list(g.nodes[tgt_type].data.keys()):
        raw_feats[k] = g.nodes[tgt_type].data.pop(k)

    print(f'For tgt type {tgt_type}, feature keys (num={len(raw_feats)}):')
    for k, v in raw_feats.items():
        print(k, v.size())
    print()

    # metapath selection
    if args.select == 'tadmps':
        from tadmps.selector import greedy_select, make_folds, depth_summary
        y_sel = init_labels.long()
        folds = make_folds(trainval_nid, n_folds=args.n_folds, seed=args.fold_seed)
        torch.manual_seed(args.probe_seed)  # reproducible probes
        np.random.seed(args.probe_seed)
        print(f'[TAD-MPS] CV info-gain selection (metric={args.select_metric}, '
              f'probe={args.probe}) over {len(raw_feats)} feature candidates')
        selected, _ = greedy_select(
            raw_feats, y_sel, folds, num_classes,
            min_select=args.min_select, max_select=args.max_select,
            eps=args.sel_eps, patience=args.sel_patience,
            probe=args.probe, hidden=args.probe_hidden, steps=args.probe_steps,
            metric=args.select_metric, verbose=True)
        if tgt_type not in selected:  # keep target channel
            selected = [tgt_type] + selected
        raw_feats = {k: raw_feats[k] for k in selected}
        print(f'[TAD-MPS] selected {len(selected)} feature metapaths: {selected} '
              f'depth={depth_summary(selected)}\n')

    data_size = {k: v.size(-1) for k, v in raw_feats.items()}
    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()

    scalar = None
    labels = init_labels.clone()
    device = 'cpu'
    if args.dataset != 'IMDB':
        labels_cpu = labels.long().to(device)
    else:
        labels = labels.float(); labels_cpu = labels.to(device)

    for seed in args.seeds:
        print('Restart with seed =', seed)
        args.seed = seed
        set_random_seed(args.seed)

        checkpt_folder = f'./output/{args.dataset}/'
        if not os.path.exists(checkpt_folder):
            os.makedirs(checkpt_folder)
        checkpt_file = checkpt_folder + uuid.uuid4().hex
        print('checkpt_file', checkpt_file)

        # train/val split
        val_ratio = 0.2
        train_nid = trainval_nid.copy()
        np.random.shuffle(train_nid)
        split = int(train_nid.shape[0] * val_ratio)
        val_nid = train_nid[:split]
        train_nid = train_nid[split:]
        train_nid = np.sort(train_nid)
        val_nid = np.sort(val_nid)

        train_node_nums = len(train_nid)
        valid_node_nums = len(val_nid)
        test_node_nums = len(test_nid)
        trainval_point = train_node_nums
        valtest_point = trainval_point + valid_node_nums
        print(f'#Train {train_node_nums}, #Val {valid_node_nums}, #Test {test_node_nums}')

        labeled_nid = np.concatenate((train_nid, val_nid, test_nid))
        labeled_num_nodes = len(labeled_nid)
        num_nodes = dl.nodes['count'][0]

        if labeled_num_nodes < num_nodes:
            flag = np.ones(num_nodes, dtype=bool)
            flag[train_nid] = 0; flag[val_nid] = 0; flag[test_nid] = 0
            extra_nid = np.where(flag)[0]
            print(f'Find {len(extra_nid)} extra nid for dataset {args.dataset}')
        else:
            extra_nid = np.array([])

        feats = {k: v.detach().clone() for k, v in raw_feats.items()}

        # label propagation
        label_feats = {}
        if args.label_feats:
            if args.dataset != 'IMDB':
                label_onehot = torch.zeros((num_nodes, num_classes))
                label_onehot[train_nid] = F.one_hot(init_labels[train_nid], num_classes).float()
            else:
                label_onehot = torch.zeros((num_nodes, num_classes))
                label_onehot[train_nid] = init_labels[train_nid].float()

            extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_label_hops + 1]
            if len(extra_metapath):
                max_length = max(args.num_label_hops + 1, max([len(ele) for ele in extra_metapath]))
            else:
                max_length = args.num_label_hops + 1

            print(f'Current num hops = {args.num_label_hops} for label propagation')
            meta_adjs = hg_propagate_sparse_pyg(
                adjs, tgt_type, args.num_label_hops, max_length, extra_metapath,
                prop_feats=False, echo=True, prop_device=prop_device)

            for k, v in tqdm(meta_adjs.items()):
                label_feats[k] = remove_diag(v) @ label_onehot
            gc.collect()

            if args.dataset == 'IMDB':
                condition = lambda ra, rb, rc, k: True
                check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=False, loss_type='bce')
            else:
                condition = lambda ra, rb, rc, k: True
                check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=True)
            print('Involved label keys', label_feats.keys())

        # data loaders
        train_loader = torch.utils.data.DataLoader(
            train_nid, batch_size=args.batch_size, shuffle=True, drop_last=False)

        with_mask = False
        eval_loader, full_loader = [], []
        eval_batch_size = 2 * args.batch_size

        for batch_idx in range((labeled_num_nodes - 1) // eval_batch_size + 1):
            batch_start = batch_idx * eval_batch_size
            batch_end = min(labeled_num_nodes, (batch_idx + 1) * eval_batch_size)
            batch = torch.LongTensor(labeled_nid[batch_start:batch_end])
            batch_feats = {k: x[batch] for k, x in feats.items()}
            batch_labels_feats = {k: x[batch] for k, x in label_feats.items()}
            eval_loader.append((batch, batch_feats, batch_labels_feats, None))

        for batch_idx in range((len(extra_nid) - 1) // eval_batch_size + 1):
            batch_start = batch_idx * eval_batch_size
            batch_end = min(len(extra_nid), (batch_idx + 1) * eval_batch_size)
            batch = torch.LongTensor(extra_nid[batch_start:batch_end])
            batch_feats = {k: x[batch] for k, x in feats.items()}
            batch_labels_feats = {k: x[batch] for k, x in label_feats.items()}
            full_loader.append((batch, batch_feats, batch_labels_feats, None))

        # unchanged SeHGNN
        gc.collect()
        model = SeHGNN(args.dataset, args.embed_size, args.hidden, num_classes, feats.keys(), label_feats.keys(), tgt_type,
                       args.dropout, args.input_drop, args.att_drop, args.n_fp_layers, args.n_task_layers, args.act,
                       args.residual, data_size=data_size)
        model = model.to(device)
        NPARAMS.append(get_n_params(model))
        NCHAN.append(len(feats) + len(label_feats))
        if args.seed == args.seeds[0]:
            print(model)
            print('# Params:', get_n_params(model))

        loss_fcn = nn.BCEWithLogitsLoss() if args.dataset == 'IMDB' else nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_epoch, best_val_loss = -1, 1000000
        best_test_loss = 0; best_val = (0, 0); best_test = (0, 0)
        best_pred = None
        train_times = []

        # training loop
        for epoch in tqdm(range(args.epoch)):
            gc.collect()
            start = time.time()
            loss, acc = train(model, feats, label_feats, labels_cpu, loss_fcn, optimizer, train_loader, evaluator, scalar=scalar)
            train_times.append(time.time() - start)

            with torch.no_grad():
                model.eval()
                raw_preds = []
                for batch, batch_feats, batch_labels_feats, batch_mask in eval_loader:
                    batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
                    batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
                    raw_preds.append(model(batch, batch_feats, batch_labels_feats, None).cpu())
                raw_preds = torch.cat(raw_preds, dim=0)
                loss_val = loss_fcn(raw_preds[trainval_point:valtest_point], labels[val_nid]).item()
                loss_test = loss_fcn(raw_preds[valtest_point:labeled_num_nodes], labels[test_nid]).item()

            preds = raw_preds.argmax(dim=-1) if args.dataset != 'IMDB' else (raw_preds > 0.).int()
            val_acc = evaluator(preds[trainval_point:valtest_point], labels[val_nid])
            test_acc = evaluator(preds[valtest_point:labeled_num_nodes], labels[test_nid])

            if loss_val < best_val_loss:  # best checkpoint
                best_epoch = epoch
                best_val_loss = loss_val
                best_test_loss = loss_test
                best_val = val_acc
                best_test = test_acc
                best_pred = raw_preds
                torch.save(model.state_dict(), f'{checkpt_file}.pkl')

            if epoch - best_epoch > args.patience:  # early stopping
                break
            if epoch > 0 and epoch % 10 == 0:
                print(f'Epoch {epoch}: Val loss {loss_val:.4f} val_acc ({val_acc[0]*100:.2f},{val_acc[1]*100:.2f}) '
                      f'| best@{best_epoch} val ({best_val[0]*100:.2f},{best_val[1]*100:.2f}) '
                      f'test ({best_test[0]*100:.2f},{best_test[1]*100:.2f})')

        print(f'Best Epoch {best_epoch}: Val ({best_val[0]*100:.4f},{best_val[1]*100:.4f}) '
              f'Test ({best_test[0]*100:.4f},{best_test[1]*100:.4f})')

        # full-graph prediction
        all_pred = torch.empty((num_nodes, num_classes))
        all_pred[labeled_nid] = best_pred
        if len(full_loader):
            model.load_state_dict(torch.load(f'{checkpt_file}.pkl', map_location='cpu'), strict=True)
            with torch.no_grad():
                model.eval()
                raw_preds = []
                for batch, batch_feats, batch_labels_feats, batch_mask in full_loader:
                    batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
                    batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
                    raw_preds.append(model(batch, batch_feats, batch_labels_feats, None).cpu())
                raw_preds = torch.cat(raw_preds, dim=0)
            all_pred[extra_nid] = raw_preds

        predict_prob = all_pred.softmax(dim=1) if args.dataset != 'IMDB' else torch.sigmoid(all_pred)
        preds = predict_prob.argmax(dim=1, keepdim=True) if args.dataset != 'IMDB' else (predict_prob > 0.5).int()
        test_acc = evaluator(preds[test_nid], labels[test_nid])
        print(f'seed {seed} FINAL test_acc micro/macro = {test_acc[0]*100:.2f}/{test_acc[1]*100:.2f}')
        RESULTS.append((test_acc[0], test_acc[1]))
        TIMES.append(float(np.mean(train_times)))
        del model


# result buffers
RESULTS = []
TIMES = []
NPARAMS = []
NCHAN = []


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='SeHGNN + TAD-MPS (Model A)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1])
    parser.add_argument('--dataset', type=str, default='DBLP', choices=['DBLP', 'ACM', 'IMDB'])
    parser.add_argument('--root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--embed-size', type=int, default=512)
    parser.add_argument('--num-hops', type=int, default=2)
    parser.add_argument('--label-feats', action='store_true', default=False)
    parser.add_argument('--num-label-hops', type=int, default=2)
    parser.add_argument('--n-fp-layers', type=int, default=2)
    parser.add_argument('--n-task-layers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--input-drop', type=float, default=0.1)
    parser.add_argument('--att-drop', type=float, default=0.)
    parser.add_argument('--act', type=str, default='none', choices=['none', 'relu', 'leaky_relu', 'sigmoid'])
    parser.add_argument('--residual', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=50)
    # selection options
    parser.add_argument('--select', type=str, default='none', choices=['none', 'tadmps'],
                        help='none = original SeHGNN; tadmps = proposed Model A')
    parser.add_argument('--select-metric', type=str, default='ce', choices=['ce', 'f1'],
                        help='ce = information-gain (val CE reduction); f1 = val macro-F1')
    parser.add_argument('--probe', type=str, default='linear', choices=['linear', 'mlp'])
    parser.add_argument('--probe-hidden', type=int, default=128)
    parser.add_argument('--probe-steps', type=int, default=100)
    parser.add_argument('--probe-seed', type=int, default=0)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--fold-seed', type=int, default=0)
    parser.add_argument('--min-select', type=int, default=3)
    parser.add_argument('--max-select', type=int, default=12)
    parser.add_argument('--sel-eps', type=float, default=0.001)
    parser.add_argument('--sel-patience', type=int, default=2)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'ACM':
        args.ACM_keep_F = False
    print(args)
    main(args)

    # summary
    R = np.array(RESULTS) * 100
    tag = 'SeHGNN (original)' if args.select == 'none' else f'Model A / TAD-MPS ({args.select_metric})'
    print(f'\n================ {args.dataset}: {tag} over {len(args.seeds)} seeds ================')
    print(f'#channels(k) = {NCHAN[0]}   #params = {NPARAMS[0]:,}   '
          f'avg s/epoch = {np.mean(TIMES):.3f}')
    print(f'test micro = {R[:,0].mean():.2f} +/- {R[:,0].std():.2f}   '
          f'macro = {R[:,1].mean():.2f} +/- {R[:,1].std():.2f}')
