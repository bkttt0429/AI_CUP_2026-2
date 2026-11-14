"""Assemble the Go report page from parts/ (head, chapter cards c*.html, footer, NUM, engine, chapter scripts a*.js, engine tail).
  python3 reports/go_story/assemble.py            -> reports/go_story/index.html
"""
from pathlib import Path

D = Path(__file__).resolve().parent; P = D / "parts"


def main():
    cards = sorted(P.glob("c*.html")); scripts = sorted(P.glob("a*.js"))
    out = [(P / "00_head.html").read_text()]
    out += [c.read_text() for c in cards]
    out.append((P / "99_foot.html").read_text())
    out.append("<script>\n" + (P / "10_num.js").read_text() + "\n</script>\n")
    out.append((P / "01_engine_head.js").read_text())
    out += ["\n  // ===== " + s.name + " =====\n" + s.read_text() for s in scripts]
    out.append((P / "98_engine_tail.js").read_text())
    html = "\n".join(out)
    (D / "index.html").write_text(html)
    print(f"index.html: {len(html):,} chars; {len(cards)} card files, {len(scripts)} script files")


if __name__ == "__main__":
    main()
