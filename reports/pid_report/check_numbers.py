#!/usr/bin/env python3
"""Number gate for the player-identification report text.

Usage: python3 check_numbers.py [--selftest] [--cards] FILE...
Every decimal number with >= 3 digits after the point in FILE must occur in the facts files.
"""
import re
import sys

FACTS = ["/home/u2507115/.cache/pidrpt_wt/reports/pid_report/facts_A.md",
         "/home/u2507115/.cache/pidrpt_wt/reports/pid_report/facts_B.md",
         "/home/u2507115/.cache/pidrpt_wt/reports/pid_report/facts_C.md",
         "/home/u2507115/.cache/pidrpt_wt/reports/pid_report/facts_D.md"]
EXTRA = ["+0.0713", "+0.0583", "0.82", "+0.0104", "+0.0148", "+0.0227", "+0.0037", "+0.0068"]  # the corrected calibration totals

NUM_RE = re.compile(r"[+\-−]?\d+\.\d{3,}")


def _blank(m):
    """Replace matched text by spaces, keeping newlines so line numbers survive."""
    return re.sub(r"[^\n]", " ", m.group(0))


def _strip(text, is_html=True):
    text = re.sub(r"<!--.*?-->", _blank, text, flags=re.S)
    text = re.sub(r"/\*.*?\*/", _blank, text, flags=re.S)
    text = re.sub(r"(?m)(^|(?<=\s))//[^\n]*", _blank, text)
    if is_html:
        text = re.sub(r"<svg\b[^>]*>", _blank, text, flags=re.S)  # svg root attributes
        # every tag's attributes (this also removes coordinate attributes of svg children)
        text = re.sub(r"</?[A-Za-z][^<>]*>", _blank, text, flags=re.S)
    return text


def numbers(text, is_html=True):
    """(line number, token) for every decimal number with >= 3 digits after the point."""
    out = []
    for i, line in enumerate(_strip(text, is_html).split("\n"), 1):
        for m in NUM_RE.finditer(line):
            out.append((i, m.group(0).replace("−", "-")))
    return out


def traceable(token, facts_text):
    """True when the token (with or without its sign) occurs as a whole number in facts_text or EXTRA."""
    bare = token.lstrip("+-−")
    hay = facts_text + "\n" + "\n".join(EXTRA)
    return re.search(r"(?<![\d.])" + re.escape(bare) + r"(?!\d)", hay) is not None


def card_limits(html):
    """One message per violation in cards whose id starts with card-pid2-."""
    msgs = []
    starts = [m for m in re.finditer(r'id="(card-pid2-[^"]*)"', html)]
    for k, m in enumerate(starts):
        end = html.find("</section>", m.end())  # the card itself only, never the lesson after it
        if end < 0:
            end = len(html)
        seg = html[m.end():end]
        cid = m.group(1)
        for ul in re.finditer(r'<ul\b[^>]*class="[^"]*\bkey\b[^"]*"[^>]*>(.*?)</ul>', seg, flags=re.S):
            lis = re.findall(r"<li\b[^>]*>(.*?)</li>", ul.group(1), flags=re.S)
            if len(lis) > 4:
                msgs.append("%s: ul.key has %d li (max 4)" % (cid, len(lis)))
            for li in lis:
                t = re.sub(r"<[^>]*>", "", li).strip()
                if len(t) > 64:
                    msgs.append("%s: li of %d characters (max 64): %s" % (cid, len(t), t[:30]))
        for tb in re.finditer(r"<table\b.*?</table>", seg, flags=re.S):
            rows = re.findall(r"<tr\b.*?</tr>", tb.group(0), flags=re.S)
            body = [r for r in rows if "<th" not in r]
            if len(rows) > len(body):
                n = len(body)
            else:
                n = max(len(rows) - 1, 0)
            if n > 6:
                msgs.append("%s: table has %d body rows (max 6)" % (cid, n))
    return msgs


def selftest():
    assert traceable("0.9069275", "…Public 0.9069275…")
    assert not traceable("0.906927", "…0.9069275…")
    assert traceable("+0.0035", "… +0.0035 [+0.0020, +0.0049] …")
    assert len(numbers('<td class="n">0.9035072</td>')) == 1
    assert numbers('<rect x="12.500" width="3.250">') == []
    assert numbers("a −0.0123 b") == [(1, "-0.0123")]
    assert numbers("<!-- 0.1234 -->\nx 0.5678") == [(2, "0.5678")]
    assert numbers("for(i=0;i<n;i++){v=x>0.1234}", False) == [(1, "0.1234")]
    assert card_limits('<div id="card-pid2-x"><ul class="key">' + "<li>a</li>" * 5 + "</ul></div>")
    card = ('<section class="page card" id="card-pid2-y"><ul class="key">' + "<li>a</li>" * 3 + "</ul><table>"
            + "<tr><td>x</td></tr>" * 4 + "</table></section>")
    lesson = '<section class="lesson" id="sec-y"><table>' + "<tr><td>x</td></tr>" * 10 + "</table></section>"
    assert not card_limits(card + lesson)
    print("selftest OK")


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    if "--selftest" in argv:
        selftest()
        if not args:
            return 0
    facts = ""
    for f in FACTS:
        with open(f, encoding="utf-8") as fh:
            facts += fh.read() + "\n"
    bad = 0
    for path in args:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        lines = text.split("\n")
        for ln, tok in numbers(text, path.endswith(".html")):
            if not traceable(tok, facts):
                ctx = lines[ln - 1]
                i = ctx.find(tok.lstrip("+-−"))
                print("%s:%d: %s not in facts: %s" % (path, ln, tok, ctx[max(i - 10, 0):max(i - 10, 0) + 30].strip()))
                bad += 1
        if "--cards" in argv:
            for msg in card_limits(text):
                print("%s: %s" % (path, msg))
                bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
