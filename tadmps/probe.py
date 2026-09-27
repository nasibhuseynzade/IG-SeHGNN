import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


# bag-of-channels probe
class BagProbe(nn.Module):
    def __init__(self, dims, num_classes, hidden=None, dropout=0.0):
        super().__init__()
        self.hidden = hidden
        if hidden is None:  # linear probe
            self.maps = nn.ModuleList([nn.Linear(d, num_classes, bias=False) for d in dims])
        else:  # mlp probe
            self.maps = nn.ModuleList([nn.Linear(d, hidden, bias=False) for d in dims])
            self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden, num_classes))

    def forward(self, xs):
        out = 0  # summed logits
        for lin, x in zip(self.maps, xs):
            out = out + lin(x)
        return out if self.hidden is None else self.head(out)


def fit_and_score(xs, y, tr_idx, va_idx, num_classes,
                  probe='mlp', hidden=128, dropout=0.0, metric='f1',
                  steps=150, lr=0.05, weight_decay=1e-4, device='cpu'):
    multilabel = (y.dim() == 2)
    yf = y.float()

    # class prior
    if len(xs) == 0:
        if multilabel:
            prior = yf[tr_idx].mean(0).clamp(1e-6, 1 - 1e-6)
            P = prior.unsqueeze(0).expand(len(va_idx), -1)
            if metric == 'ce':
                return float(-F.binary_cross_entropy(P, yf[va_idx]))
            pred = (P > 0.5).cpu().numpy()
            return f1_score(y[va_idx].cpu().numpy(), pred, average='micro', zero_division=0)
        cnt = torch.bincount(y[tr_idx], minlength=num_classes).float()
        if metric == 'ce':
            prior = (cnt / cnt.sum()).clamp_min(1e-9)
            return float(-F.nll_loss(torch.log(prior).expand(len(va_idx), -1), y[va_idx]))
        pred = np.full(len(va_idx), cnt.argmax().item())
        return f1_score(y[va_idx].cpu().numpy(), pred, average='macro', zero_division=0)

    dims = [x.size(1) for x in xs]
    h = None if probe == 'linear' else hidden
    net = BagProbe(dims, num_classes, hidden=h, dropout=dropout).to(device)
    xs = [x.to(device).float() for x in xs]
    y = y.to(device); yf = yf.to(device)
    xtr = [x[tr_idx] for x in xs]
    loss_fn = (lambda o: F.binary_cross_entropy_with_logits(o, yf[tr_idx])) if multilabel \
        else (lambda o: F.cross_entropy(o, y[tr_idx]))

    # train probe
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(net(xtr)).backward()
        opt.step()

    # held-out score
    net.eval()
    with torch.no_grad():
        logits = net([x[va_idx] for x in xs])
        if multilabel:
            if metric == 'ce':
                return float(-F.binary_cross_entropy_with_logits(logits, yf[va_idx]))
            pred = (logits > 0).cpu().numpy()
            return f1_score(y[va_idx].cpu().numpy(), pred, average='micro', zero_division=0)
        if metric == 'ce':
            return float(-F.cross_entropy(logits, y[va_idx]))
        pred = logits.argmax(1).cpu().numpy()
    return f1_score(y[va_idx].cpu().numpy(), pred, average='macro', zero_division=0)


# fold-averaged score
def score_cv(xs, y, folds, num_classes, **kw):
    return float(np.mean([fit_and_score(xs, y, tr, va, num_classes, **kw)
                          for tr, va in folds]))
