"""In-allocation extraction of global and pre-move local choice features."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

import numpy as np
import torch
from torch.nn import functional as F

from pid_reference_policy import _GraphExtraOutputs, load_humansl, neutral_rows, position_indices, te
from aicup_pid.local_choice import action_basis, local_geometry


def replay_positions(game):
    """Check every recorded move, compute full KataGo features only for the 32 selected states."""
    stones, moves = te.setup_and_moves(game["sgf"])
    moves = moves[:256]
    turns = [i for i, (color, _, _) in enumerate(moves) if color == game["color"]]
    selected = {turns[i] for i in position_indices(len(turns))}
    state = te.start_state(stones, te.RULES)
    locations = [state.board.loc(x, y) for y in range(19) for x in range(19)]
    rows = {name: [] for name in ("bins", "globs", "legal", "played", "mover", "t")}
    for t, (color, x, y) in enumerate(moves):
        board = state.board
        player = te.Board.BLACK if color == "B" else te.Board.WHITE
        loc = te.Board.PASS_LOC if x is None else board.loc(x, y)
        if board.pla != player or not board.would_be_legal(player, loc):
            raise ValueError(f"{game['key']}: invalid recorded turn at {t}")
        if t in selected:
            planes, glob = state.get_input_features(te._FEATURES)
            legal = np.asarray([board.would_be_legal(player, loc_a) for loc_a in locations] + [True], dtype=bool)
            action = te._FEATURES.loc_to_tensor_pos(loc, board)
            if not legal[action]:
                raise ValueError("played point is masked")
            rows["bins"].append(planes[0].astype(np.uint8))
            rows["globs"].append(glob[0])
            rows["legal"].append(legal)
            rows["played"].append(action)
            rows["mover"].append(int(color == "W"))
            rows["t"].append(t)
        state.play(player, loc)
    return {k: np.stack(v) for k, v in rows.items()}


def get_basis(data, games, plies, products):
    return action_basis(data["trunk"][games, plies].float(), data["shape"][games, plies].float(),
                        data["previous"][games, plies].float(), data["relative"][games, plies].float(),
                        products=products)


def standardized(array, fit):
    reference = array[fit].flatten(0, 1)
    mean = reference.mean(0)
    sd = reference.std(0, unbiased=False).clamp_min(.05)
    return (array - mean) / sd, mean, sd


@torch.no_grad()
def extract(m, device="cuda"):
    start = time.monotonic()
    games = m["games"]
    n = len(games)
    dims = {"trunk": (362, 385), "shape": (362, 45), "previous": (45,), "relative": (362, 5),
            "global": (388,), "local": (96,), "base": (362,), "played": (), "t": ()}
    half = {"trunk", "shape", "previous", "relative"}
    data = {}
    for name, dim in dims.items():
        dtype = torch.float16 if name in half else torch.int64 if name in {"played", "t"} else torch.float32
        data[name] = torch.empty(n, 32, *dim, dtype=dtype, device=device)
    action_dim = 385 + 4 * 45 + 5
    scales = {}
    fit = torch.tensor([g["split"] == "train" and g["slot"] < 7 for g in games], device=device)
    for products in (True, False):
        data["residual", products] = torch.empty(n, 32, action_dim, device=device)
        scales[products] = torch.zeros(action_dim, dtype=torch.float64, device=device)
    ckpt = m["checkpoint"]["path"]
    backbone, _ = load_humansl(ckpt, device)
    backbone.eval()
    profiles = torch.as_tensor(neutral_rows(), device=device, dtype=torch.float32)
    config = backbone.config
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
                             initializer=te.init_worker, initargs=(config,)) as pool:
        for start_g in range(0, n, 64):
            part = list(pool.map(replay_positions, games[start_g:start_g + 64]))
            for j in range(0, len(part), 4):
                records = part[j:j + 4]
                sl = slice(start_g + j, start_g + j + len(records))
                inputs = {k: torch.as_tensor(np.concatenate([r[k] for r in records]), device=device) for k in records[0]}
                planes = inputs["bins"].float()
                side, played, turns = (inputs[k].long() for k in ("mover", "played", "t"))
                extra = _GraphExtraOutputs(["trunkfinal"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = backbone(planes, inputs["globs"].float(),
                                      input_meta=profiles[side] if backbone.get_has_metadata_encoder() else None,
                                      extra_outputs=extra)
                    post = backbone.postprocess_output(output)[0]
                spatial = extra.returned["trunkfinal"].float().flatten(2).transpose(1, 2)
                if spatial.shape[1:] != (361, 384):
                    raise ValueError("unexpected backbone shape")
                trunk = F.pad(spatial, (0, 1, 0, 1))
                trunk[:, 361, -1] = 1
                base = F.log_softmax(post[0][:, 0].float().masked_fill(~inputs["legal"], -torch.inf), -1)
                prob = base.exp()
                shape, previous, relative, summary = local_geometry(planes, prob)
                win = F.softmax(post[1].float(), -1)
                entropy = -(prob * base.masked_fill(~inputs["legal"], 0)).sum(-1)
                global_ctx = torch.cat([spatial.mean(1), (win[:, 0] - win[:, 1])[:, None], entropy[:, None],
                                        turns[:, None] / 256, side[:, None]], -1)
                if not base.gather(-1, played[:, None]).isfinite().all():
                    raise ValueError("nonfinite observed move")
                batch = {"trunk": trunk, "shape": shape, "previous": previous, "relative": relative,
                         "global": global_ctx, "local": summary, "base": base, "played": played, "t": turns}
                for key, value in batch.items():
                    data[key][sl] = value.reshape(len(records), 32, *dims[key])
                for products in (True, False):
                    raw = action_basis(*(data[k][sl].flatten(0, 1).float() for k in ("trunk", "shape", "previous", "relative")),
                                       products=products)
                    expected = torch.einsum("qa,qad->qd", prob, raw)
                    observed = raw[torch.arange(len(played), device=device), played]
                    residual = observed - expected
                    if not residual.isfinite().all():
                        raise ValueError("invalid reference moment")
                    data["residual", products][sl] = residual.reshape(len(records), 32, -1)
                    weights = prob * fit[sl].repeat_interleave(32)[:, None]
                    scales[products] += torch.einsum("qa,qad->d", weights, raw.square()).double()
            if start_g % 256 == 0 or start_g + len(part) == n:
                print(f"extracted {start_g + len(part)}/{n} games in {time.monotonic() - start:.1f}s", flush=True)
    del backbone
    norm = {}
    for products in (True, False):
        scale = (scales[products] / (32 * fit.sum())).sqrt().clamp_min(.05).float()
        data["scale", products] = scale
        data["residual", products] /= scale
        norm[f"action_rms_{int(products)}"] = scale
    gz, norm["global_mean"], norm["global_sd"] = standardized(data.pop("global"), fit)
    lz, norm["local_mean"], norm["local_sd"] = standardized(data.pop("local"), fit)
    gen = torch.Generator().manual_seed(1138)
    projection = torch.randn(gz.size(-1), lz.size(-1), generator=gen).to(device) / gz.size(-1) ** .5
    control, norm["control_mean"], norm["control_sd"] = standardized(torch.tanh(gz @ projection), fit)
    norm["fixed_global_projection"] = projection
    data["global_key"] = torch.cat([gz, control], -1)
    data["local_key"] = torch.cat([gz, lz], -1)
    return data, {k: v.cpu().numpy() for k, v in norm.items()}
