"""Deterministic dataset boundaries, immutable protocol and feature validation."""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def player_split(rows):
    if any(g['pool'] != 'calibration' or g['fold'] not in (3, 4) for g in rows):
        raise ValueError('only existing calibration folds 3/4 are allowed')
    if len({g['key'] for g in rows}) != len(rows) or len({g['sgf_sha256'] for g in rows}) != len(rows):
        raise ValueError('duplicate key or SGF')
    owners = sorted({g['owner'] for g in rows}, key=lambda p: hashlib.sha256(('behavior-player-v1|' + p).encode()).hexdigest())
    if len(owners) != 100:
        raise ValueError('expected 100 calibration players')
    return {'train': owners[:64], 'val': owners[64:80], 'test': owners[80:]}


def prefix_history(moves, t, target):
    if t < 0 or t > len(moves):
        raise ValueError('invalid prefix endpoint')
    out = np.zeros((8, 6), dtype=np.float32)
    before = moves[max(0, t - 8):t]
    for j, (color, x, y) in enumerate(before, 8 - len(before)):
        out[j] = [0 if x is None else x / 18, 0 if y is None else y / 18,
                  float(x is None), float(color == 'W'), float(color == target), 1]
    return out


def position_indices(n):
    if n < 1:
        raise ValueError('empty game')
    return np.linspace(0, n - 1, min(16, n), dtype=int)


def episode_games(groups, rng):
    owners = rng.choice(sorted(groups), 16, replace=False).tolist()
    draws = [rng.choice(groups[p], 4, replace=False).tolist() for p in owners]
    return owners, [x[:3] for x in draws], [x[3] for x in draws]


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('refusing to replace a different protocol')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'x') as f:
            json.dump(value, f, sort_keys=True, indent=2)
            f.write('\n')


def validate_features(a):
    n = len(a['played'])
    shapes = {'phi': (n, 362, 385), 'logp': (n, 362), 'state': (n, 388),
              'history': (n, 8, 6), 'ref_value': (n, 386), 't': (n,)}
    if not 1 <= n <= 16 or any(a[k].shape != sh for k, sh in shapes.items()):
        raise ValueError('invalid feature shapes')
    if any(not np.isfinite(a[k]).all() for k in ('phi', 'state', 'history', 'ref_value')):
        raise ValueError('nonfinite features')
    if not (a['phi'][:, 361, -1] == 1).all() or a['phi'][:, 361, :-1].any() or a['phi'][:, :361, -1].any():
        raise ValueError('invalid pass feature encoding')
    lp = a['logp'].astype(float)
    if np.isnan(lp).any() or np.isposinf(lp).any() or not np.allclose(np.exp(lp).sum(1), 1, atol=1e-5):
        raise ValueError('invalid normalized legal policy')
    played = a['played']
    if not np.issubdtype(played.dtype, np.integer) or (played < 0).any() or (played > 361).any():
        raise ValueError('invalid action index')
    if not np.isfinite(lp[np.arange(n), played]).all():
        raise ValueError('illegal observed target')
    if not (np.diff(a['t']) > 0).all():
        raise ValueError('positions must be distinct and chronological')
