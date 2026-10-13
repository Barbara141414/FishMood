"""Optional CPU-friendly causal TCN, shared trunk and independent task heads."""
import numpy as np


def torch_modules():
    try:
        import torch
        from torch import nn
        return torch, nn
    except ImportError as exc:
        raise ValueError('TCN需要: pip install -e ".[temporal]"') from exc


def make_network(input_size, classes, history=6):
    torch, nn = torch_modules()

    class Block(nn.Module):
        def __init__(self, inp, channels, dilation):
            super().__init__()
            self.padding = 2*dilation
            self.conv1 = nn.Conv1d(inp, channels, 3, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation)
            self.skip = nn.Conv1d(inp, channels, 1) if inp != channels else nn.Identity()
            self.drop = nn.Dropout(0.1)

        def forward(self, x):
            h = self.drop(torch.relu(self.conv1(nn.functional.pad(x, (self.padding, 0)))))
            h = self.drop(torch.relu(self.conv2(nn.functional.pad(h, (self.padding, 0)))))
            return torch.relu(h+self.skip(x))

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            dilations = [1, 2, 4]
            while 1+4*sum(dilations) < history:
                dilations.append(dilations[-1]*2)
            self.trunk = nn.Sequential(*[Block(input_size if i == 0 else 32, 32, d) for i, d in enumerate(dilations)])
            self.heads = nn.ModuleDict({name: nn.Linear(32, len(values)) for name, values in classes.items()})

        def forward(self, x):
            h = self.trunk(x.transpose(1, 2))[:, :, -1]
            return {name: head(h) for name, head in self.heads.items()}

    return Network()


def fit_preprocess(x):
    flat = x.reshape(-1, x.shape[-1])
    center, scale = [], []
    for col in flat.T:
        v = col[np.isfinite(col)]
        center.append(float(np.mean(v)) if len(v) else 0.0)
        scale.append(max(float(np.std(v)), 1e-4) if len(v) else 1.0)
    return {"center": np.array(center), "scale": np.array(scale)}


def transform(x, prep):
    valid = np.isfinite(x)
    z = np.where(valid, (x-prep["center"])/prep["scale"], 0)
    return np.concatenate([np.clip(z, -12, 12), valid.astype(float)], axis=-1).astype(np.float32)


def fit_tcn(train_x, train_y, val_x, val_y, classes, epochs, seed):
    import copy
    torch, nn = torch_modules()
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    prep = fit_preprocess(train_x)
    x = torch.from_numpy(transform(train_x, prep))
    vx = torch.from_numpy(transform(val_x, prep))
    ys = {k: torch.tensor(v, dtype=torch.long) for k, v in train_y.items()}
    vys = {k: torch.tensor(v, dtype=torch.long) for k, v in val_y.items()}
    net = make_network(x.shape[-1], classes, x.shape[1])
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.01)
    weights = {}
    for task, labels in train_y.items():
        counts = np.bincount(labels[labels >= 0], minlength=len(classes[task]))
        w = 1/np.maximum(counts, 1)
        weights[task] = torch.tensor(w/w.mean(), dtype=torch.float32)

    def loss_for(logits, labels, selected=None):
        losses = []
        for name in classes:
            yy = labels[name] if selected is None else labels[name][selected]
            mask = yy >= 0
            if mask.any():
                losses.append(nn.functional.cross_entropy(logits[name][mask], yy[mask], weight=weights[name]))
        return torch.stack(losses).mean() if losses else None

    best, state, stale = float("inf"), None, 0
    log = []
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(len(x))
        for idx in order.split(64):
            loss = loss_for(net(x[idx]), ys, idx)
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2)
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = loss_for(net(vx), vys)
            score = float(val_loss) if val_loss is not None else float("inf")
        log.append({"epoch": epoch+1, "validation_loss": score})
        if score < best-1e-5:
            best, stale = score, 0
            state = copy.deepcopy(net.state_dict())
        else:
            stale += 1
        if stale >= 12:
            break
    if state is None:
        raise ValueError("TCN无有效验证标签，无法训练")
    return {"state": {k: v.numpy() for k, v in state.items()}, "prep": prep, "training_log": log}


def tcn_logits(payload, sequences, classes):
    torch, _ = torch_modules()
    torch.set_num_threads(2)
    x = transform(sequences, payload["prep"])
    net = make_network(x.shape[-1], classes, x.shape[1])
    net.load_state_dict({k: torch.from_numpy(v) for k, v in payload["state"].items()})
    net.eval()
    chunks = {k: [] for k in classes}
    with torch.no_grad():
        for start in range(0, len(x), 256):
            pred = net(torch.from_numpy(x[start:start+256]))
            for name, logits in pred.items():
                chunks[name].append(logits.numpy())
    return {k: np.concatenate(v) for k, v in chunks.items()}
