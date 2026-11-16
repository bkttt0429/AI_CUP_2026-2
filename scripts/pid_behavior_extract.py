"""Full-channel, deterministic candidate-action features for the behavior pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from behavior_data import digest, immutable_json, player_split, position_indices, prefix_history, validate_features
from pid_behavior_submit import assert_allocation

SOURCE = Path('/work/u2507115/pid_v2/choice_probe_20261116/2993db0b3ed8/manifest.json')
LEGACY = Path('/home/u2507115/.cache/pidchoice_wt')
ROOT = Path(__file__).resolve().parents[1]


def prepare(root):
    m = json.loads(SOURCE.read_text())
    rows = [g for g in m['games'] if g['pool'] == 'calibration']
    split = player_split(rows)
    assert len(rows) == 1000
    for p in sum(split.values(), []):
        gs = [g for g in rows if g['owner'] == p]
        assert sum(g['role'] == 'reference' for g in gs) == 7
        assert sum(g['role'] == 'heldout' for g in gs) == 3
    for g in rows:
        assert hashlib.sha256(g['sgf'].encode()).hexdigest() == g['sgf_sha256']
    sources = {str(p): digest(p) for p in [SOURCE, Path(__file__), ROOT/'scripts/behavior_data.py',
               ROOT/'scripts/pid_behavior_submit.py', LEGACY/'scripts/pid_choice_extract.py',
               LEGACY/'scripts/pid_choice_data_audit.py', LEGACY/'scripts/pid_trunk_extract.py']}
    protocol = {'schema': 1, 'split': split, 'rows': rows, 'sources': sources,
                'checkpoint': m['assets']['checkpoint'], 'positions_per_game': 16, 'max_ply': None,
                'feature_channels': 385, 'legality': 'simple ko, no superko',
                'seeds': [20261116, 20261117], 'arms': ['background', 'global', 'contextual'],
                'steps': 1200, 'validation_interval': 200, 'reference_games': 3,
                'reference_positions': 8, 'query_positions': 4, 'episode_players': 16,
                'lr': .001, 'weight_decay': .0001, 'identity_weight': .1, 'identity_temperature': .1,
                'max_gpu_minutes': 60, 'max_cache_bytes': int(5.5 * 2**30), 'min_free_bytes': 8 * 2**30,
                'scope': 'Held-out players for this model only; previously explored calibration population, not fresh efficacy confirmation.'}
    immutable_json(root/'protocol.json', protocol)
    print('Frozen protocol', root/'protocol.json', {k: len(v) for k, v in split.items()}, flush=True)


def init_worker(config):
    r1 = Path(os.environ.get('AICUP1_ROOT', '/home/u2507115/BK_AICUP_2026-1'))
    sys.path[:0] = [str(r1/'scripts'), str(r1/'src')]
    import trunk_extract as te
    te.init_worker(config)


def replay(g):
    import trunk_extract as te
    stones, moves = te.setup_and_moves(g['sgf'])
    own = [t for t, (c, _, _) in enumerate(moves) if c == g['color']]
    selected = {own[i] for i in position_indices(len(own))}
    gs = te.start_state(stones, te.RULES)
    out = {k: [] for k in ('bins', 'globs', 'played', 'mover', 't', 'legal', 'history')}
    for t, (c, x, y) in enumerate(moves):
        pla = te.Board.BLACK if c == 'B' else te.Board.WHITE
        if pla != gs.board.pla:
            raise ValueError('recorded color does not match board turn')
        loc = te.Board.PASS_LOC if x is None else gs.board.loc(x, y)
        if not gs.board.would_be_legal(pla, loc):
            raise ValueError('illegal actual move')
        if t in selected:
            b, glob = gs.get_input_features(te._FEATURES)
            legal = np.zeros(362, bool)
            for yy in range(19):
                for xx in range(19):
                    pos = gs.board.loc(xx, yy)
                    legal[te._FEATURES.loc_to_tensor_pos(pos, gs.board)] = gs.board.would_be_legal(pla, pos)
            legal[361] = True
            played = 361 if x is None else te._FEATURES.loc_to_tensor_pos(loc, gs.board)
            assert legal[played]
            for name, value in [('bins', b[0].astype(np.uint8)), ('globs', glob[0].astype(np.float32)),
                                ('played', played), ('mover', int(c == 'W')), ('t', t), ('legal', legal),
                                ('history', prefix_history(moves, t, g['color']))]:
                out[name].append(value)
        gs.play(pla, loc)
    return {k: np.stack(v) for k, v in out.items()}


def extract(root):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('use a bounded Slurm allocation')
    assert_allocation(root, 'extract')
    p = json.loads((root/'protocol.json').read_text())
    for path, sha in p['sources'].items():
        assert digest(path) == sha, path
    checkpoint = p['checkpoint']['path']
    assert digest(checkpoint) == p['checkpoint']['sha256']
    if shutil.disk_usage(root).free < p['min_free_bytes'] + 5 * 2**30:
        raise RuntimeError('insufficient scratch headroom')
    import torch
    sys.path[:0] = [str(LEGACY/'scripts'), str(LEGACY/'src')]
    from pid_choice_extract import forward_choices
    from aicup.humansl import load_humansl
    torch.set_num_threads(1)
    config = torch.load(checkpoint, map_location='cpu', weights_only=True)['config']
    model, _ = load_humansl(checkpoint, 'cuda'); model.eval()
    projection = np.eye(384, dtype=np.float32)
    directory = root/'games'; directory.mkdir(exist_ok=True)
    started = time.monotonic(); records = []
    with ProcessPoolExecutor(max_workers=7, mp_context=mp.get_context('spawn'), initializer=init_worker, initargs=(config,)) as ex:
        for start in range(0, len(p['rows']), 28):
            todo = []
            for i in range(start, min(start+28, len(p['rows']))):
                path = directory/f'{i:04d}.npz'; meta = path.with_suffix('.json')
                if path.exists() and meta.exists():
                    rec = json.loads(meta.read_text())
                    assert rec['key'] == p['rows'][i]['key'] and rec['protocol_sha256'] == digest(root/'protocol.json')
                    assert digest(path) == rec['sha256']
                    records.append(rec)
                else:
                    todo.append(i)
            replays = ex.map(replay, [p['rows'][i] for i in todo], chunksize=1)
            for i, r in zip(todo, replays):
                x = forward_choices(model, r, projection, batch=32)
                phi = x['psi'].float()
                lp = x['logp'].float().log_softmax(-1)
                prob = lp.exp()
                mu = (prob[..., None] * phi).sum(1)
                played = torch.tensor(r['played'], device='cuda', dtype=torch.long)
                chosen = phi[torch.arange(len(phi), device='cuda'), played]
                entropy = -(prob * lp.masked_fill(~torch.isfinite(lp), 0)).sum(1)
                scalar = torch.stack([x['value'], torch.tensor(r['t']/256, device='cuda'),
                                      torch.tensor(r['mover'], device='cuda'), entropy], -1).float()
                state = torch.cat([phi[:, :361, :384].mean(1), scalar], -1)
                rv = torch.cat([chosen - mu, lp[torch.arange(len(phi), device='cuda'), played, None]], -1)
                arrays = {'phi': phi.cpu().numpy().astype(np.float16), 'logp': lp.cpu().numpy(),
                          'state': state.cpu().numpy(), 'history': r['history'], 'ref_value': rv.cpu().numpy(),
                          'played': r['played'].astype(np.int64), 't': r['t']}
                validate_features(arrays)
                next_bytes = sum(a.nbytes for a in arrays.values()) + 4096
                if sum(rec['bytes'] for rec in records) + next_bytes > p['max_cache_bytes']:
                    raise RuntimeError('next record would exceed cache limit')
                if shutil.disk_usage(root).free < p['min_free_bytes'] + next_bytes:
                    raise RuntimeError('insufficient headroom before next write')
                path = directory/f'{i:04d}.npz'; tmp = path.with_suffix('.tmp')
                with open(tmp, 'wb') as f:
                    np.savez(f, **arrays)
                os.replace(tmp, path)
                rec = {'index': i, 'key': p['rows'][i]['key'], 'n': len(played), 'bytes': path.stat().st_size,
                       'sha256': digest(path), 'protocol_sha256': digest(root/'protocol.json')}
                immutable_json(path.with_suffix('.json'), rec); records.append(rec)
            size = sum(rec['bytes'] for rec in records)
            if size > p['max_cache_bytes'] or shutil.disk_usage(root).free < p['min_free_bytes']:
                raise RuntimeError('cache budget or free-space limit reached')
            print(f'extracted {len(records)}/1000, {size/2**30:.3f} GiB, {time.monotonic()-started:.1f}s', flush=True)
    assert len(records) == 1000 and len({r['index'] for r in records}) == 1000
    immutable_json(root/'extraction_complete.json', {'protocol_sha256': digest(root/'protocol.json'),
                   'records': sorted(records, key=lambda r:r['index']), 'bytes': sum(r['bytes'] for r in records)})
    print('Extraction complete', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--prepare', action='store_true'); args = ap.parse_args()
    prepare(args.run_dir) if args.prepare else extract(args.run_dir)
