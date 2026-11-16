"""Fixed-budget conditional policy training; validation selects, test only reports."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from behavior_model import BehaviorPolicy
from behavior_data import digest, episode_games, immutable_json, validate_features


def player_metrics(scores, truth):
    scores, truth = np.asarray(scores), np.asarray(truth)
    c = scores.shape[1]
    if c < 2 or scores.shape[0] != len(truth) or not np.isfinite(scores).all():
        raise ValueError('invalid candidate scores')
    own = scores[np.arange(len(truth)), truth]
    wrong = (scores.sum(1) - own) / (c - 1)
    values = {'nll': -own, 'margin': own - wrong, 'top1': (scores.argmax(1) == truth).astype(float)}
    out = {}
    for k, v in values.items():
        out[k] = np.array([v[truth == i].mean() for i in range(c)])
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError('each candidate needs heldout queries')
    return out


def paired_ci(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('expected seed x player differences')
    per_player = a.mean(0)
    rng = np.random.default_rng(20261116)
    b = per_player[rng.integers(0, len(per_player), (2000, len(per_player)))].mean(1)
    return {'mean': float(per_player.mean()), 'ci95': np.quantile(b, [.025, .975]).tolist()}


def mechanism_verdict(background_gain, correct_margin, global_gains, permutation_gains):
    if background_gain['ci95'][0] <= 0 or correct_margin['ci95'][0] <= 0:
        return 'not_supported'
    if min(global_gains) <= 0 or min(permutation_gains) <= 0:
        return 'personalization_without_context_binding'
    return 'mechanism_supported_in_pilot'


def toy_batch(rng, device, candidates=8):
    """New random half-plane per task, antipodal reference pairs: equal action marginals."""
    w = rng.normal(size=(candidates, 2)); w /= np.linalg.norm(w, axis=1, keepdims=True)
    r = rng.normal(size=(candidates, 6, 2)); r = np.concatenate([r, -r], 1)
    x = rng.normal(size=(candidates, 8, 2))
    yr = (np.einsum('cmd,cd->cm', r, w) > 0).astype(int)
    yq = (np.einsum('cqd,cd->cq', x, w) > 0).astype(int)
    state = np.zeros((candidates*8, 388), np.float32); state[:, :2] = x.reshape(-1, 2)
    rs = np.zeros((candidates, 12, 388), np.float32); rs[:, :, :2] = r
    rv = np.zeros((candidates, 12, 386), np.float32); rv[:, :, 0] = yr * 2 - 1; rv[:, :, -1] = -np.log(2)
    phi = np.zeros((candidates*8, 2, 385), np.float32); phi[:, 0, 0] = -1; phi[:, 1, 0] = 1
    arrays = [phi, np.full((candidates*8, 2), -np.log(2), np.float32), state,
              np.zeros((candidates*8, 8, 6), np.float32), rs, np.zeros((candidates, 12, 8, 6), np.float32), rv,
              np.ones((candidates, 12), bool)]
    return [torch.tensor(a, device=device) for a in arrays], torch.tensor(yq.reshape(-1), device=device), torch.arange(candidates, device=device).repeat_interleave(8)


def toy_probe(root):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(1); torch.manual_seed(817)
    net = BehaviorPolicy('contextual').to(device); opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(817)
    for step in range(500):
        x, y, who = toy_batch(rng, device)
        lp = net(*x); own = lp[torch.arange(len(y), device=device), who]
        loss = F.nll_loss(own, y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2); opt.step()
    net.eval(); hits = []; margins = []
    with torch.no_grad():
        # Separate RNG, unseen continuous functions and query samples.
        for _ in range(32):
            x, y, who = toy_batch(rng, device)
            lp = net(*x); own = lp[torch.arange(len(y), device=device), who]
            wrong = lp[torch.arange(len(y), device=device), (who+1)%8]
            hits.extend((own.argmax(-1) == y).cpu().tolist())
            margins.extend((own.gather(1, y[:, None]) - wrong.gather(1, y[:, None])).flatten().cpu().tolist())
    result = {'accuracy': float(np.mean(hits)), 'correct_reference_margin': float(np.mean(margins)),
              'queries': len(hits), 'passed': np.mean(hits) >= .70 and np.mean(margins) > 0,
              'scope': 'Controlled synthetic mechanism, not Go performance'}
    root.mkdir(parents=True, exist_ok=True)
    (root/'toy.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Toy', result, flush=True)
    if not result['passed']:
        raise RuntimeError('synthetic mechanism gate failed')


def load_data(root, protocol, device):
    complete = json.loads((root/'extraction_complete.json').read_text())
    assert complete['protocol_sha256'] == digest(root/'protocol.json')
    data = {}
    for rec in complete['records']:
        path = root/'games'/f"{rec['index']:04d}.npz"
        assert digest(path) == rec['sha256']
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}; validate_features(a)
        assert protocol['rows'][rec['index']]['key'] == rec['key']
        data[rec['key']] = {k: torch.as_tensor(a[k], device=device) for k in ('phi', 'logp', 'state', 'history', 'ref_value', 'played')}
    assert set(data) == {g['key'] for g in protocol['rows']}
    return data


def references(data, groups, rng=None):
    obs = []
    for games in groups:
        selected = []
        for k in games:
            n = len(data[k]['played'])
            ix = np.arange(n) if rng is None else rng.choice(n, min(8, n), replace=False)
            selected.append({field: data[k][field][ix] for field in ('state', 'history', 'ref_value')})
        obs.append({field: torch.cat([x[field] for x in selected]) for field in ('state', 'history', 'ref_value')})
    maximum = max(len(x['state']) for x in obs)
    device = obs[0]['state'].device
    output = []
    for field in ('state', 'history', 'ref_value'):
        shape = obs[0][field].shape[1:]
        a = torch.zeros((len(obs), maximum, *shape), device=device)
        for i, x in enumerate(obs): a[i, :len(x[field])] = x[field]
        output.append(a)
    mask = torch.zeros((len(obs), maximum), device=device, dtype=torch.bool)
    for i, x in enumerate(obs): mask[i, :len(x['state'])] = True
    return output + [mask]


@torch.no_grad()
def evaluate(net, data, rows, owners, ablation=None):
    net.eval(); device = next(net.parameters()).device
    ref = [[g['key'] for g in rows if g['owner'] == p and g['role'] == 'reference'] for p in owners]
    mem = references(data, ref)
    score, truth = [], []
    for i, p in enumerate(owners):
        for g in rows:
            if g['owner'] != p or g['role'] != 'heldout': continue
            x = data[g['key']]; n = len(x['played'])
            lp = net(x['phi'].float(), x['logp'], x['state'], x['history'], *mem, ablation=ablation)
            vals = lp.gather(2, x['played'][:, None, None].expand(n, len(owners), 1)).squeeze(-1)
            score.append(vals.mean(0).cpu().numpy()); truth.append(i)
    scores, truth = np.stack(score), np.array(truth)
    return player_metrics(scores, truth), scores, truth


def train(root):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('use bounded Slurm allocation')
    if not json.loads((root/'toy.json').read_text())['passed']:
        raise RuntimeError('toy prerequisite missing')
    p = json.loads((root/'protocol.json').read_text())
    script_dir = Path(__file__).parent
    code = {str(script_dir/n): digest(script_dir/n) for n in ('behavior_model.py', 'behavior_data.py', 'pid_behavior_train.py')}
    immutable_json(root/'training_protocol.json', {'data_protocol_sha256': digest(root/'protocol.json'), 'code': code,
                    'selection': 'minimum validation own-action NLL; checkpoint selection completed before all test evaluation'})
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = True
    device = 'cuda'; started = time.monotonic()
    data = load_data(root, p, device)
    groups = {owner: [g['key'] for g in p['rows'] if g['owner'] == owner] for owner in p['split']['train']}
    directory = root/'models'; directory.mkdir(exist_ok=True)
    selections = {}; curves = {}
    for seed in p['seeds']:
        for arm in p['arms']:
            name = f'{arm}_{seed}'
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            net = BehaviorPolicy(arm).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
            best = float('inf'); curve = []
            for step in range(p['steps'] + 1):
                if step % p['validation_interval'] == 0:
                    stats, _, _ = evaluate(net, data, p['rows'], p['split']['val'])
                    nll = float(stats['nll'].mean()); entry = {'step': step, 'val_nll': nll, 'val_margin': float(stats['margin'].mean()), 'val_top1': float(stats['top1'].mean())}
                    if nll < best:
                        best = nll; selections[name] = entry
                        torch.save(net.state_dict(), directory/f'{name}.pt')
                    curve.append(entry); curves[name] = curve
                    (root/'training_progress.json').write_text(json.dumps({'selection': selections, 'curves': curves}, indent=2)+'\n')
                    print(name, entry, 'elapsed', round(time.monotonic()-started, 1), flush=True)
                if step == p['steps']: break
                net.train()
                owners, refgames, querygames = episode_games(groups, rng)
                mem = references(data, refgames, rng)
                items = []
                for key in querygames:
                    x = data[key]; ix = rng.choice(len(x['played']), 4, replace=len(x['played']) < 4)
                    items.append({k: x[k][ix] for k in x})
                q = {k: torch.cat([x[k] for x in items]) for k in items[0]}
                lp = net(q['phi'].float(), q['logp'], q['state'], q['history'], *mem)
                values = lp.gather(2, q['played'][:, None, None].expand(len(q['played']), 16, 1)).squeeze(-1)
                who = torch.arange(16, device=device).repeat_interleave(4)
                nll = -values[torch.arange(len(who), device=device), who].mean()
                identity = F.cross_entropy(values.reshape(16, 4, 16).mean(1)/p['identity_temperature'], torch.arange(16, device=device)) if arm != 'background' else nll.new_zeros(())
                loss = nll + p['identity_weight'] * identity
                if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2); opt.step()
            del net, opt
    # Selection lock is committed BEFORE any test score is computed.
    immutable_json(root/'selection.json', {'selections': selections, 'checkpoint_sha256': {path.name:digest(path) for path in directory.glob('*.pt')}})
    results = {}; per_seed = []
    for seed in p['seeds']:
        metrics = {}
        for arm in p['arms']:
            name = f'{arm}_{seed}'; net = BehaviorPolicy(arm).to(device)
            net.load_state_dict(torch.load(directory/f'{name}.pt', map_location=device, weights_only=True))
            for ablation in ([None, 'permute', 'no_actions'] if arm == 'contextual' else [None]):
                label = arm if ablation is None else f'{arm}_{ablation}'
                stats, scores, truth = evaluate(net, data, p['rows'], p['split']['test'], ablation)
                metrics[label] = stats
                np.savez_compressed(root/f'test_{label}_{seed}.npz', scores=scores, truth=truth, owners=np.array(p['split']['test']))
                results[f'{label}_{seed}'] = {k: v.tolist() for k, v in stats.items()}
            del net
        per_seed.append(metrics)
    gains = {a: np.stack([m[a]['nll'] - m['contextual']['nll'] for m in per_seed])
             for a in ('background', 'global', 'contextual_permute', 'contextual_no_actions')}
    reports = {a: paired_ci(v) for a, v in gains.items()}
    margin = paired_ci(np.stack([m['contextual']['margin'] for m in per_seed]))
    result = {'scope': p['scope'], 'selection': selections, 'nll_improvement': reports, 'correct_reference_margin': margin,
              'per_seed_gain': {a: v.mean(1).tolist() for a, v in gains.items()}, 'per_player': results,
              'verdict': mechanism_verdict(reports['background'], margin, gains['global'].mean(1), gains['contextual_permute'].mean(1)),
              'elapsed_seconds': time.monotonic()-started, 'protocol_sha256': digest(root/'protocol.json'),
              'training_protocol_sha256': digest(root/'training_protocol.json')}
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_player', 'selection')}, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run-dir', type=Path, required=True); ap.add_argument('--toy', action='store_true')
    a = ap.parse_args(); toy_probe(a.run_dir) if a.toy else train(a.run_dir)
