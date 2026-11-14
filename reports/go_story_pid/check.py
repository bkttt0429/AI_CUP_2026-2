"""Static checks of the assembled page (no browser): runs every ANIM and CHARTS function under node with a stub DOM, then
  - each SVG parses as XML; every <animate> has as many values as keyTimes, keyTimes rise from 0 to 1
  - text inside each SVG stays within the viewBox width (CJK ~1.0 em, other ~0.58 em; warns only)
  - HTML: unique ids; every data-sec has a lesson with id sec-N; every data-anim has an ANIM; every chart div has a CHARTS entry
  python3 reports/go_story/check.py
"""
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

D = Path(__file__).resolve().parent; P = D / "parts"
STUB = """
const window = {matchMedia: () => ({matches: false})};
const document = {querySelectorAll: () => [], getElementById: () => null, body: {classList: {add(){}, remove(){}}}};
"""


def run_node():
    import shutil
    if shutil.which("node") is None:   # compute nodes have no node: skip the animation run, keep the HTML checks
        print("WARN node not found: animation functions not executed (HTML checks only)")
        return {"anims": {}, "charts": {}, "errors": [], "skipped": True}
    num = (P / "10_num.js").read_text()
    head = (P / "01_engine_head.js").read_text().replace("<script>", "")
    scripts = "\n".join(s.read_text() for s in sorted(P.glob("a*.js")))
    test = """
  const OUT = {anims: {}, charts: {}, errors: []};
  Object.keys(ANIM).forEach(k => { const f = ANIM[k]; try { AD = (f && f.dur) || 9; HOLD = AD - 0.7; RAW = !!(f && f.raw); const an = f(); OUT.anims[k] = an ? {svg: an.svg, steps: an.steps || [], tech: an.tech || '', note: an.note || '', dur: AD} : null; } catch (e) { OUT.errors.push('ANIM.' + k + ': ' + e.message); } });
  Object.keys(CHARTS).forEach(k => { try { OUT.charts[k] = CHARTS[k](); } catch (e) { OUT.errors.push('CHARTS.' + k + ': ' + e.message); } });
  console.log(JSON.stringify(OUT));
})();
"""
    src = STUB + num.replace("window.NUM", "window.NUM") + "\n" + head + "\n" + scripts + "\n" + test
    tmp = D / ".check_run.js"; tmp.write_text(src)
    r = subprocess.run(["node", str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        print("NODE FAILED:\n" + r.stderr[-3000:]); sys.exit(1)
    return json.loads(r.stdout)


def text_width(s, fs):
    w = 0.0
    for ch in s:
        o = ord(ch)
        w += fs * (1.0 if (0x2E80 <= o <= 0x9FFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F) else 0.58)
    return w


CLASS_FS = {"an-k": 13, "an-n": 13, "an-t": 14, "an-l": 12, "an-v": 15, "an-s": 11.5, "an-w": 12, "an-b": 13, "an-h": 13, "an-cl": 12, "an-sv": 16, "an-sk": 11, "sift-t": 13, "sift-c": 11, "vd": 14}


def check_svg(name, svg, warn, err):
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        err.append(f"ANIM.{name}: SVG does not parse: {e}"); return
    vb = root.get("viewBox")
    W = float(vb.split()[2]) if vb else None
    ns = "{http://www.w3.org/2000/svg}"
    for el in root.iter():
        tag = el.tag.replace(ns, "")
        if tag == "animate":
            vals, kts = el.get("values", "").split(";"), el.get("keyTimes", "").split(";")
            if len(vals) != len(kts):
                err.append(f"ANIM.{name}: animate values {len(vals)} != keyTimes {len(kts)}")
            else:
                k = [float(x) for x in kts]
                if abs(k[0]) > 1e-9 or abs(k[-1] - 1) > 1e-9 or any(b < a - 1e-9 for a, b in zip(k, k[1:])):
                    err.append(f"ANIM.{name}: keyTimes not 0..1 rising: {kts[:3]}...{kts[-2:]}")
        if tag == "text" and W and el.get("x") is not None:
            s = "".join(el.itertext()).strip()
            if not s:
                continue
            cls = (el.get("class") or "").split(); fs = max([CLASS_FS.get(c, 0) for c in cls] + [0]) or 13
            m = re.search(r"font-size:\s*([\d.]+)px", el.get("style") or "")
            if m:
                fs = float(m.group(1))
            try:
                x = float(el.get("x"))
            except ValueError:
                continue
            w = text_width(s, fs); anchor = el.get("text-anchor", "start")
            x0, x1 = (x, x + w) if anchor == "start" else ((x - w / 2, x + w / 2) if anchor == "middle" else (x - w, x))
            if x0 < -2 or x1 > W + 2:
                warn.append(f"ANIM.{name}: text may overflow ({x0:.0f}..{x1:.0f} of {W:.0f}): {s[:40]}")


class Ids(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids, self.secs, self.lessons, self.anims, self.charts, self.lesson_ids = [], [], 0, [], [], set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        cls = (a.get("class") or "").split()
        if "deriv-open" in cls and "data-sec" in a:
            self.secs.append(int(a["data-sec"]))
        if tag == "section" and "lesson" in cls:
            self.lessons += 1; self.lesson_ids.add(a.get("id"))
        if "data-anim" in a:
            self.anims.append(a["data-anim"])
        if "bars" in cls and "id" in a:
            self.charts.append(a["id"])


def main():
    out = run_node(); warn, err = [], list(out["errors"])
    for k, an in out["anims"].items():
        if an is None:
            warn.append(f"ANIM.{k} returned null"); continue
        check_svg(k, an["svg"], warn, err)
    html = (D / "index.html").read_text(); p = Ids(); p.feed(html)
    dup = sorted({i for i in p.ids if p.ids.count(i) > 1})
    if dup:
        err.append(f"duplicate ids: {dup}")
    for s in p.secs:   # the engine opens the lesson whose id is sec-N (98_engine_tail.js secIdx), so check ids, not positions
        if f"sec-{s}" not in p.lesson_ids:
            err.append(f"data-sec {s} has no lesson with id sec-{s} (lessons: {p.lessons})")
    for a in ([] if out.get("skipped") else p.anims):
        if a not in out["anims"]:
            err.append(f"data-anim '{a}' has no ANIM")
    for c in ([] if out.get("skipped") else p.charts):
        if c not in out["charts"]:
            warn.append(f"chart div '{c}' has no CHARTS entry")
    print(f"anims {len(out['anims'])}, charts {len(out['charts'])}, cards with derivation buttons {len(p.secs)}, lessons {p.lessons}")
    for w in warn:
        print("WARN", w)
    for e in err:
        print("ERROR", e)
    print("OK" if not err else f"{len(err)} errors")
    (D / ".check_out.json").write_text(json.dumps(out, ensure_ascii=False))
    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()
