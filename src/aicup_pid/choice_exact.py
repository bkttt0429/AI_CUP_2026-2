"""Full action normalizer for a fixed candidate coefficient estimate."""
import torch


def exact_scores(logp, phi, played, coefficients, *, time_chunk=64, candidate_chunk=600):
    lp, x, z = logp.double(), phi.double(), coefficients.double()
    p = played.long()
    if lp.shape != x.shape[:2] or z.ndim != 2 or z.shape[1] != x.shape[2] or p.shape != lp.shape[:1]:
        raise ValueError("invalid exact score shapes")
    if time_chunk <= 0 or candidate_chunk <= 0 or not torch.isfinite(x).all() or not torch.isfinite(z).all():
        raise ValueError("nonfinite exact parameters or invalid chunks")
    if torch.isnan(lp).any() or torch.isposinf(lp).any() or not torch.isfinite(lp.gather(1, p[:, None])).all():
        raise ValueError("invalid or illegal observation")
    out = torch.zeros(len(z), dtype=torch.float64, device=x.device)
    with torch.inference_mode():
        for t in range(0, len(x), time_chunk):
            xx, ll, pp = x[t:t + time_chunk], lp[t:t + time_chunk], p[t:t + time_chunk]
            normalizer = torch.logsumexp(ll, 1)
            for c in range(0, len(z), candidate_chunk):
                zz = z[c:c + candidate_chunk]
                tilt = (xx.reshape(-1, x.shape[2]) @ zz.T).reshape(len(xx), x.shape[1], len(zz))
                actual = tilt[torch.arange(len(xx), device=x.device), pp]
                out[c:c + len(zz)] += (actual - torch.logsumexp(ll[:, :, None] + tilt, 1) + normalizer[:, None]).sum(0)
    if not torch.isfinite(out).all():
        raise ValueError("nonfinite exact score")
    return out
