"""Matched episodes, sealed test scoring and paired player-level comparisons."""
from collections import Counter
import time

import numpy as np
import torch
from torch.nn import functional as F

from aicup_pid.local_choice import ConditionalChoice, phase_permutation
from aicup_pid.reference_policy import episodic_loss, interval, owner_metrics, played_scores
from pid_choice_manifest import atomic_json, source_record
from pid_local_choice_features import get_basis
from pid_reference_policy import split_indices


class Arm:
    def __init__(self, arm, data):
        self.name = arm
        self.products = arm != "A"
        self.context = data["local_key" if arm in ("L", "A") else "global_key"]
        self.scale = data["scale", self.products]
        self.values = data["residual", self.products]
        self.model = ConditionalChoice(self.context.size(-1), len(self.scale), uniform=arm == "U").cuda()

    def predict(self, data, game_ids, turns, keys, values):
        features = get_basis(data, game_ids, turns, self.products) / self.scale
        base = data["base"][game_ids, turns]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp, null = self.model(self.context[game_ids, turns], features, base, keys, values)
        return lp, null


@torch.no_grad()
def evaluate(arm, data, indices, ablation="intact"):
    """One score per candidate and query game, using the same seven supports in all arms."""
    arm.model.eval()
    p = len(indices)
    refs = torch.as_tensor(indices[:, :7], device="cuda")
    keys = arm.context[refs]
    values = arm.values[refs]
    if ablation == "permute":
        order = phase_permutation(data["t"][refs].cpu().numpy(), 509)
        idx = torch.as_tensor(order, device="cuda")[..., None]
        values = torch.gather(values, 2, idx.expand_as(values))
    elif ablation == "zero":
        values = torch.zeros_like(values)
    elif ablation != "intact":
        raise ValueError("unknown ablation")
    qgames = indices[:, 7:].ravel()
    scores, base_scores, null_mass = [], [], []
    for i in range(0, len(qgames), 2):
        games = torch.as_tensor(qgames[i:i + 2], device="cuda")
        ids = games.repeat_interleave(32)
        turns = torch.arange(32, device="cuda").repeat(len(games))
        y = data["played"][ids, turns]
        lp, null = arm.predict(data, ids, turns, keys, values)
        observed = played_scores(lp, y)
        baseline = data["base"][ids, turns].gather(1, y[:, None])
        scores.append((observed - baseline).reshape(len(games), 32, p).mean(1).cpu().numpy())
        base_scores.append(baseline.reshape(len(games), 32).mean(1).cpu().numpy())
        null_mass.append(null.mean().item())
    score = np.concatenate(scores).reshape(p, 3, p)
    base_ll = np.concatenate(base_scores).reshape(p, 3)
    if not (np.isfinite(score).all() and np.isfinite(base_ll).all()):
        raise ValueError("nonfinite game scores")
    ix = np.arange(p)
    own = score[ix, :, ix]
    per, metrics = owner_metrics(score)
    nll = -(own + base_ll).mean()
    ce = F.cross_entropy(torch.tensor(score.mean(1)) * 10, torch.arange(p))
    metrics.update(move_nll=float(nll), base_nll=float(-base_ll.mean()), null=float(np.mean(null_mass)),
                   objective=float(ce.item() + .1 * nll))
    return score, per, metrics, base_ll


def fit_one(data, train, dev, label, seed, dest, protocol):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    arm = Arm(label, data)
    opt = torch.optim.AdamW(arm.model.parameters(), lr=3e-4, weight_decay=1e-4)
    target = torch.arange(16, device="cuda").repeat_interleave(4)
    path = dest / f"{label}_{seed}.pt"
    best = float("inf")
    history = []
    start = time.monotonic()
    for step in range(1, 4001):
        arm.model.train()
        owners = rng.choice(len(train), 16, replace=False)
        slots = np.stack([rng.choice(10, 4, replace=False) for _ in owners])
        refs = torch.as_tensor(train[owners[:, None], slots[:, :3]], device="cuda")
        ids = torch.as_tensor(train[owners, slots[:, 3]], device="cuda").repeat_interleave(4)
        turns = torch.as_tensor(np.concatenate([rng.choice(32, 4, replace=False) for _ in owners]), device="cuda")
        lp, _ = arm.predict(data, ids, turns, arm.context[refs], arm.values[refs])
        loss = episodic_loss(lp, data["played"][ids, turns], target)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(arm.model.parameters(), 1.)
        opt.step()
        if step % 1000 == 0:
            _, _, validation, _ = evaluate(arm, data, dev)
            history.append({"step": step, **validation})
            if validation["objective"] < best:
                best = validation["objective"]
                torch.save({"protocol": protocol["id"], "seed": seed, "arm": label,
                            "state": arm.model.state_dict(), "step": step}, path)
            print(f"{label} seed {seed} step {step} dev CE+NLL {validation['objective']:.5f} "
                  f"{time.monotonic() - start:.1f}s", flush=True)
    atomic_json(dest / f"{label}_{seed}_dev.json", history)
    return path


def per_size(scores, n):
    rows = scores if n == 1 else scores[:, :n].sum(1, keepdims=True)
    p = len(rows)
    owner = np.arange(p)[:, None, None]
    own = np.take_along_axis(rows, owner, axis=-1)
    rank = ((rows > own) | ((rows == own) & (np.arange(p)[None, None] < owner))).sum(-1)
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.).mean(1), rank


def support_rivals(data, indices, games):
    # Rank tags and mean spatial residuals from support only. True query labels are used
    # later for evaluation, after every candidate has received its score.
    support = indices[:, :7]
    ranks = [Counter(games[i]["key"].split(":")[0] for i in ids).most_common(1)[0][0] for ids in support]
    profile = data["residual", True][torch.as_tensor(support, device="cuda")][..., :384].mean((1, 2))
    profile = F.normalize(profile, dim=-1)
    sim = (profile @ profile.T).cpu().numpy()
    allowed = np.equal.outer(ranks, ranks) & ~np.eye(len(ranks), dtype=bool)
    if not allowed.any(1).all():
        raise ValueError("no same-rank reference rival for a test owner")
    return np.where(allowed, sim, -np.inf).argmax(1)


def compare(scores, left, right, rivals):
    a, b = scores[left], scores[right]
    ids = np.arange(len(a))
    delta = a - b
    result = {"equal_size_rank": interval(owner_metrics(a)[0] - owner_metrics(b)[0]),
              "same_rank_reference_rival_ll_margin": interval((delta[ids, :, ids] - delta[ids, :, rivals]).mean(1))}
    diff = {}
    for size in (1, 2, 3):
        sa, ra = per_size(a, size)
        sb, rb = per_size(b, size)
        diff[size] = sa - sb
        if size == 1:
            result["one_game_top1_changes"] = {"games": int(ra.size), "rescued": int(((ra == 0) & (rb != 0)).sum()),
                                               "broken": int(((ra != 0) & (rb == 0)).sum()),
                                               "rescued_from_rank2_to_5": int(((ra == 0) & (rb > 0) & (rb < 5)).sum()),
                                               "previously_rank2_to_5": int(((rb > 0) & (rb < 5)).sum())}
    result["rank_by_size"] = {str(n): interval(values) for n, values in diff.items()}
    result["one_minus_three_rank_change"] = interval(diff[1] - diff[3])
    return result


def experiment(data, games, dest, protocol):
    _, train = split_indices(games, "train")
    _, dev = split_indices(games, "dev")
    paths = []
    for seed in protocol["seeds"]:
        for label in protocol["arms"]:
            paths.append(fit_one(data, train, dev, label, seed, dest, protocol))
    # Open test outcomes only after all eight checkpoints have been fixed by development scores.
    owners, test = split_indices(games, "test")
    idx = np.arange(len(test))
    rivals = support_rivals(data, test, games)
    runs, scores, comparisons = {}, {}, {}
    for path in paths:
        ck = torch.load(path, weights_only=True)
        if ck["protocol"] != protocol["id"]:
            raise ValueError("checkpoint protocol mismatch")
        arm = Arm(ck["arm"], data)
        arm.model.load_state_dict(ck["state"])
        for mode in (("intact", "permute", "zero") if ck["arm"] == "L" else ("intact",)):
            name = path.stem if mode == "intact" else path.stem + "_" + mode
            score, _, metrics, base_ll = evaluate(arm, data, test, mode)
            scores[name] = score
            runs[name] = {"checkpoint_step": ck["step"], "parameters": sum(p.numel() for p in arm.model.parameters()),
                          "metrics": metrics, "own_minus_cyclic_ll": interval((score[idx, :, idx] - score[idx, :, (idx + 1) % len(idx)]).mean(1)),
                          "own_minus_near_same_rank_ll": interval((score[idx, :, idx] - score[idx, :, rivals]).mean(1))}
            if mode == "zero" and float(np.abs(score).max()) > 2e-6:
                raise ValueError("zero memory failed to recover base")
    for seed in protocol["seeds"]:
        comparisons[str(seed)] = {key: compare(scores, f"L_{seed}", other, rivals) for key, other in {
            "L-G": f"G_{seed}", "L-U": f"U_{seed}", "L-A": f"A_{seed}", "L-permuted": f"L_{seed}_permute"}.items()}
    ok = all(by_arm[key]["equal_size_rank"]["ci95"][0] > 0 for by_arm in comparisons.values()
             for key in ("L-G", "L-U", "L-permuted"))
    personalized = all(runs[f"L_{s}"]["own_minus_cyclic_ll"]["ci95"][0] > 0 for s in protocol["seeds"])
    verdict = "local_retrieval_supported" if ok else "personalization_only" if personalized else "mechanism_not_established"
    turns = data["t"][torch.as_tensor(test[:, :7], device="cuda")].cpu().numpy()
    perm = phase_permutation(turns, 509)
    np.savez_compressed(dest / "scores.npz", **scores, base_ll=base_ll, rivals=rivals, owners=np.array(owners))
    return {"protocol": protocol["id"], "data_id": protocol["data_id"], "scope": protocol["scope"],
            "verdict": verdict, "owner_counts": [len(train), len(dev), len(test)],
            "fraction_reference_tokens_permuted": float((perm != np.arange(32)).mean()),
            "checkpoints": [source_record(path) for path in paths], "runs": runs, "comparisons": comparisons}
