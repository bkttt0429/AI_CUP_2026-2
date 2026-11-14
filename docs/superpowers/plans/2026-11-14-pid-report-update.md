# Player-Identification Report Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the player-identification part of the report page (`go_story`) up to date with the state of 2026-11-14 (Public 0.9069275): add chapters 13–19 for the added pools, correct the stale statements of chapters 8–12, and leave a facts library and an outline from which the organisers' final written report can be written after 2026-11-24.

**Architecture:** All new content goes into two NEW part files (`c09_pidv2.html` for cards and derivation sections, `a08b_pidv2.js` for numbers and charts), so the existing parts of the other competition are not touched. The old player-ID part (`c08_pid.html`) gets a short list of pinned edits. Every number comes from the facts library (three Markdown files extracted from the working notes); a small checker script fails the build when a number in the player-ID parts cannot be found there.

**Tech Stack:** static HTML + JS parts concatenated by `assemble.py`; `check.py` (python3 + node) lints the built page; the checker is python3 standard library only. No framework, no new dependency.

**Spec:** the facts library `/home/u2507115/BK_AICUP_2026-2/reports/pid_report/facts_A.md`, `facts_B.md`, `facts_C.md` (read all three before Task 2) plus the "Content spec" inside each task below. The working notes `/home/u2507115/BK_AICUP_2026-2/METHOD_NOTES.md` are the source of the facts; open them only to resolve a doubt, at the line range a fact cites.

## Global Constraints

- Report text is Traditional Chinese (Taiwan usage). No Simplified characters. Internal names (`pid_v2c2_bs.csv`, `o12F`, …) stay in ASCII exactly as in the facts.
- Every number written into `c08_pid.html`, `c09_pidv2.html` or `a08b_pidv2.js` is copied verbatim from a fact (all digits, sign, interval brackets). Never round, average, sum or otherwise compute a number that no fact states. If a number given in this plan is not found in the facts, do NOT write it: use the facts' value if the same quantity is there, otherwise leave the sentence out and list it in the task report under "not found".
- Files you may create: `/home/u2507115/BK_AICUP_2026-1/reports/go_story/parts/c09_pidv2.html`, `/home/u2507115/BK_AICUP_2026-1/reports/go_story/parts/a08b_pidv2.js`, `/home/u2507115/BK_AICUP_2026-2/reports/pid_report/check_numbers.py`, `/home/u2507115/BK_AICUP_2026-2/reports/pid_report/build_log.md`, `/home/u2507115/.cache/report_audit/job_pid_build.sh`.
- Files you may edit, and only in the places a task names: `/home/u2507115/BK_AICUP_2026-1/reports/go_story/parts/c08_pid.html`, `.../parts/a09_terms.js` (append entries only), `/home/u2507115/BK_AICUP_2026-1/reports/go_story/講稿_關鍵.md` (only the part under the heading `## 競賽一　玩家辨識`).
- Do not edit any other file in `/home/u2507115/BK_AICUP_2026-1` (it belongs to another working session). In particular do not edit `10_num.js`, `c00`–`c07`, `a01`–`a07`, `assemble.py`, `check.py`.
- No git command in `/home/u2507115/BK_AICUP_2026-1` at all. In `/home/u2507115/BK_AICUP_2026-2` do not commit either; the plan's owner commits after review.
- Cluster rule: nothing is computed on the login node. `assemble.py`, `check.py` and `check_numbers.py` are run ONLY through the job script of Task 1 (`sbatch ... scripts/nano4/run.sbatch bash ~/.cache/report_audit/job_pid_build.sh`, submitted from `/home/u2507115/BK_AICUP_2026-1`). Editing files and reading logs on the login node is fine.
- Card limits (the page has a presentation mode that overflowed before): at most 4 bullets in `ul.key`, at most 44 full-width characters per bullet, at most 6 table rows on a card. Longer material goes into the derivation section (`section.lesson`).
- Status tags: a bullet that states a measured number ends with `<span class="tag t-m">實測</span>`; an interpretation ends with `<span class="tag t-i">推論</span>`; anything that used leaderboard feedback carries `<span class="tag t-p">用了 Public 回饋</span>`.
- No personal data: no personal names, e-mail addresses, tokens or account identifiers. Do not state a leaderboard rank except in the exact form the facts give it (for example `當時 1／81`).
- Wording about the unused data must stay exactly: 「fold 0 從未用於訓練或選模型」「沒有人工標註測試資料」「由排行榜回饋得知的答案只用於評估，沒有用於訓練」 — and only where a fact supports the sentence.

## Review Focus

1. A number copied wrongly or rounded differently from its source: the checker of Task 1 must fail on it; Task 1 Step 2 pins this with a self-test.
2. A stale sentence left in chapters 8–12 (an upload "planned for 11-07", "最終檔" used for a file that is no longer final): Task 9 lists every such sentence; its verification greps for the stale strings.
3. An id collision (`sec-N`, `card-…`, `chart-…`) or a derivation button without its section: `check.py` reports these as errors; Task 10 requires "0 errors".
4. A claim stronger than its evidence (for example "the original pools are solved exactly", a calibration ratio stated as a constant, a rank): the Content spec gives the exact wording; do not strengthen it. Task 8's verification greps for forbidden phrases.
5. An external resource without all five disclosure fields (source URL, version or file name, licence, how it is used, measured contribution): Task 7's table has these five columns and its verification counts non-empty cells.
6. A card that overflows in presentation mode: Task 10 counts bullets and bullet lengths with the checker's `--cards` option.

---

## Shared reference for all tasks

**Page conventions (copy them from an existing card, do not invent markup).** Open `/home/u2507115/BK_AICUP_2026-1/reports/go_story/parts/c08_pid.html` and use the card `id="card-pid-assign"` together with its `section.lesson id="sec-18"` as the model. A card is:

```html
<section class="page card" id="card-pid2-NAME" data-sec="N" data-ch="C" data-title="短標題">
  <div class="report">
    <div class="txt">
      <div class="num">第 C 章 · 章名 · k／m</div>
      <h3>一句話的結論當標題</h3>
      <ul class="key"><li>…</li></ul>
      <div class="cause"><span><span class="k">問題</span><b>…</b></span><span class="arrow">→</span><span><span class="k go">作法</span><b>…</b></span><span class="arrow">→</span><span><span class="k stuck">限制</span><b>…</b></span><button class="deriv-open" data-sec="N">推導 ▸</button></div>
    </div>
    <div class="fig">
      <div class="bars" id="chart-pid2-NAME"></div>
      <div class="chips"><span class="chip"><b>來源</b>…</span></div>
    </div>
  </div>
</section>
<section class="lesson" id="sec-N" data-title="C.k 標題">
  <h3>C.k 標題</h3>
  <p class="proves">對應報告卡 N+1：…</p>
  <h4>…</h4><table>…</table><ul>…</ul>
  <p class="src">出處：…</p>
</section>
```

The first card of a chapter also carries `data-chtitle="章名"`. A table on a card is wrapped in `<div class="table-wrap">`; numeric cells are `<td class="n">`. A chart is registered in a JS part as `CHARTS['chart-id'] = () => '<h4>標題</h4>' + barRows(rows, lo, hi);` with `rows = [[label, value, displayText, color]]`; read `a08_pid.js` for two worked examples (`chart-pid-score` for bars, `chart-pid-arch` for an SVG diagram inside `<div class="pid-fig" id="…">`).

**Id allocation (fixed; do not change):**

| Chapter (`data-ch`) | `data-chtitle` | Card id | `data-sec` / lesson id | Chart id |
|---|---|---|---|---|
| 13 | 玩家辨識：新增題池與 v2 流程 | `card-pid2-task` | 25 / `sec-25` | (table) |
| 13 | | `card-pid2-arch` | 26 / `sec-26` | `chart-pid2-arch` |
| 14 | 玩家辨識：分數的每一步 | `card-pid2-climb` | 27 / `sec-27` | `chart-pid2-climb` |
| 15 | 玩家辨識：讓序列網路變強的做法 | `card-pid2-data` | 28 / `sec-28` | (table) |
| 15 | | `card-pid2-long` | 29 / `sec-29` | `chart-pid2-fam` |
| 15 | | `card-pid2-sup` | 30 / `sec-30` | `chart-pid2-single` |
| 16 | 玩家辨識：驗證工具與選檔規則 | `card-pid2-valid` | 31 / `sec-31` | (table) |
| 17 | 玩家辨識：沒有通過的方向 | `card-pid2-neg` | 32 / `sec-32` | (table) |
| 18 | 玩家辨識：外部資源與揭露 | `card-pid2-ext` | 33 / `sec-33` | (table) |
| 19 | 玩家辨識：進行中 | `card-pid2-next` | 34 / `sec-34` | (none) |

**Facts library layout.** Each file has sections `1 時間軸`, `2 排行榜讀數` (a table), `3 方法與成分`, `4 通過的比較`, `5 沒有通過的方向`, `6 驗證工具與事先規則`, `7 外部資源`, `8 更正與教訓`, `9 待確認`; `facts_A.md` also has `0 任務與規則`. Facts are list items `- **[A-T01]** …｜出處：…`. In a derivation section's `<p class="src">` cite the fact ids you used and their METHOD_NOTES sections.

**Known correction to apply over the facts:** if `facts_C.md` contains the sentence that the five calibration readings add up to "擬題池合計約 +0.031、新增那一半合計約 +0.033", it is the uncorrected version. The corrected statement (METHOD_NOTES §2.114 as of 2026-11-14 23:30) is: 五次的擬題池增益 +0.0132、+0.0196、+0.0268、+0.0082、+0.0035 合計 +0.0713；對應的 Public 新增那一半 +0.0104、+0.0148、+0.0227、+0.0037、+0.0068 合計 +0.0583，比例 0.82. Use the corrected numbers.

---

### Task 1: Number checker and build job

**Files:**
- Create: `/home/u2507115/BK_AICUP_2026-2/reports/pid_report/check_numbers.py`
- Create: `/home/u2507115/.cache/report_audit/job_pid_build.sh`
- Create: `/home/u2507115/BK_AICUP_2026-2/reports/pid_report/build_log.md` (one line per build: time, job id, result)

**Interfaces:**
- Produces: `python3 check_numbers.py [--selftest] [--cards] FILE...` — exit status 0 when every checked number is traceable, 1 otherwise; prints one line per untraceable number: `FILE:LINE: NUMBER not in facts: <30 characters of context>`.
- Produces: `bash ~/.cache/report_audit/job_pid_build.sh` — runs assemble, check.py and the checker, prints their outputs and a last line `PID_BUILD OK` or `PID_BUILD FAILED`.

- [ ] **Step 1: Write `check_numbers.py`**

Signature and behaviour (the body is yours):

```python
FACTS = ["/home/u2507115/BK_AICUP_2026-2/reports/pid_report/facts_A.md",
         "/home/u2507115/BK_AICUP_2026-2/reports/pid_report/facts_B.md",
         "/home/u2507115/BK_AICUP_2026-2/reports/pid_report/facts_C.md"]
EXTRA = ["+0.0713", "+0.0583", "0.82", "+0.0104", "+0.0148", "+0.0227", "+0.0037", "+0.0068"]  # the corrected calibration totals

def numbers(text: str) -> list[tuple[int, str]]:
    """(line number, token) for every decimal number with at least three digits after the point
    (regex [+-−]?\d+\.\d{3,}) in the text, after removing <svg>…</svg> coordinate attributes,
    HTML tags' attributes, and JS/HTML comments. The Unicode minus '−' counts as '-'."""

def traceable(token: str, facts_text: str) -> bool:
    """True when the token, with and without its sign, occurs in facts_text or EXTRA as a whole number
    (not as a prefix of a longer number)."""

def card_limits(html: str) -> list[str]:
    """One message per violation in cards whose id starts with card-pid2-: more than 4 <li> in ul.key,
    an <li> longer than 44 characters after stripping tags, a card table with more than 6 <tr> besides the header."""
```

`--selftest` runs these assertions and exits 0 only when all hold: `traceable("0.9069275", "…Public 0.9069275…")` is True; `traceable("0.906927", "…0.9069275…")` is False; `traceable("+0.0035", "… +0.0035 [+0.0020, +0.0049] …")` is True; `numbers('<td class="n">0.9035072</td>')` returns one token; `numbers('<rect x="12.500" width="3.250">')` returns none.

- [ ] **Step 2: Write the job script**

```bash
#!/usr/bin/env bash
# Build and lint of the report page with the player-identification checks. Run through run.sbatch only.
set -uo pipefail
main() {
  local G=/home/u2507115/BK_AICUP_2026-1/reports/go_story P=/home/u2507115/BK_AICUP_2026-2/reports/pid_report rc=0
  python3 $P/check_numbers.py --selftest || { echo "selftest FAILED"; rc=1; }
  python3 $G/assemble.py || rc=1
  python3 $G/check.py || rc=1
  local files="$G/parts/c08_pid.html"
  [ -s $G/parts/c09_pidv2.html ] && files="$files $G/parts/c09_pidv2.html"
  [ -s $G/parts/a08b_pidv2.js ] && files="$files $G/parts/a08b_pidv2.js"
  # shellcheck disable=SC2086
  python3 $P/check_numbers.py --cards $files || rc=1
  [ $rc = 0 ] && echo "PID_BUILD OK" || echo "PID_BUILD FAILED"
  return $rc
}
main "$@"; exit $?
```

- [ ] **Step 3: Run it once before any content change**

Run (from `/home/u2507115/BK_AICUP_2026-1`): `sbatch --parsable -p 8gpus,16gpus --gres=gpu:1 -c 4 -t 00:15:00 -J pid_rpt scripts/nano4/run.sbatch bash /home/u2507115/.cache/report_audit/job_pid_build.sh`, wait until `squeue -j <id>` is empty, read `logs/nano4/pid_rpt-<id>.out`.
Expected: `selftest` passes; `check.py` prints `OK` with `anims 18, charts 10` and 0 errors; the checker lists the numbers of the OLD chapter `c08_pid.html` that are not in the facts (this list is the baseline; copy it into `build_log.md` under the heading `baseline`). The last line may be `PID_BUILD FAILED` at this point because of that list; that is the expected "red" state.

---

### Task 2: Numbers and charts (`a08b_pidv2.js`)

**Files:**
- Create: `/home/u2507115/BK_AICUP_2026-1/reports/go_story/parts/a08b_pidv2.js`

**Interfaces:**
- Consumes: helpers of `01_engine_head.js` (`barRows`, `f4`), the global `CHARTS`.
- Produces: `CHARTS['chart-pid2-arch']`, `CHARTS['chart-pid2-climb']`, `CHARTS['chart-pid2-fam']`, `CHARTS['chart-pid2-single']`; one object `const P2 = {…}` at the top that holds every number the charts use.

- [ ] **Step 1: Write `P2` with a source comment above each key** (Chinese comment naming the fact ids and METHOD_NOTES sections, in the style of `10_num.js`). Keys and their content:

| Key | Content (tuples) | Where the values are |
|---|---|---|
| `climb` | `[日期時間, 檔案, Public 分數, 一句話說明這個檔多了什麼]`, one tuple per Public reading of the v2 test set from the first one (0.8612697) to 0.9069275, in time order | the tables in section 2 of `facts_A.md`, `facts_B.md`, `facts_C.md` |
| `fam` | `[家族, 說明, 擬題池增益（加在當時線上組成之上，權重 ×6；沒有 ×6 就用 ×3 並在說明註明）]` for `o7F`, `o9D`, `o8D`, `o10D`, `o11U`, `o12K`, `o12F` | `facts_C.md` sections 4 and 5 (METHOD_NOTES §2.108, §2.110, §2.113) |
| `single` | `[家族, 說明, Public 池候選人的單一網路 1 局題分數, Private 池的]` for `o3S`, `o6X`, `o7F`, `o12` B 型, `o12` C 型, `o12` A 型 | `facts_C.md` (METHOD_NOTES §2.110) |

The last five tuples of `climb` must read exactly (check them against the facts):
`['11-14 00:06', 'pid_v2is_o1.csv', 0.8829460, '第一批含外部帳號的官方網路']`, `['11-14 04:50', 'pid_v2w_n8s.csv', 0.8903230, '棋盤對稱增強的網路']`, `['11-14 13:42', 'pid_v2sl_alls.csv', 0.9016519, '對稱網路 12 個種子＋4 萬步＋加大']`, `['11-14 19:00', 'pid_v2x_ns.csv', 0.9035072, '加長訓練＋有監督輸入（2 萬步快照）']`, `['11-14 21:03', 'pid_v2c2_bs.csv', 0.9069275, '有監督輸入換成 3 萬步完成版']`.
Write the scores as strings where a trailing zero matters (`'0.8829460'`), and pass `parseFloat` of them to `barRows`.

- [ ] **Step 2: Write the four charts**

- `chart-pid2-climb`: `'<h4>Public 分數（v2 測試集）：每個上傳檔</h4>' + barRows(rows, 0.85, 0.91)`, one row per `P2.climb` tuple, label = `日期時間 ＋ 說明`, displayText = the score string, the last row in `var(--accent)`, the others `#8a93a3`.
- `chart-pid2-fam`: `'<h4>把一個新家族加進組成，擬題池分數的變化</h4>' + barRows(rows, 0, 0.01)`, label = `家族 ＋ 說明`, displayText = the gain with its sign; `o12F` and `o7F` in `var(--accent)`.
- `chart-pid2-single`: `'<h4>單一網路在官方候選人上的 1 局題分數（微調前）</h4>' + barRows(rows, 0.45, 0.58)`, value = the Public-pool number, displayText = `Public 池／Private 池` both numbers verbatim, A 型 in `var(--accent)`.
- `chart-pid2-arch`: an SVG in the style of `CHARTS['chart-pid-arch']` (same viewBox width, same inline font sizes): left box 「題目：同一人 1–5 局（黑白都有）」, below it 「候選人 × 600，每人 10 局」; middle column nine small boxes with the component names of Task 3's card `card-pid2-arch`, the box 「逐手序列網路（數十個平均，題池微調 50 步）」 drawn larger and in the accent colour; then boxes 「加權融合」→「2／3／5 局題軟性配對」→「前 5 名」.

- [ ] **Step 3: Build and check**

Run the job of Task 1. Expected in the log: `check.py` reports 0 errors (its harness calls every `CHARTS` entry, so a chart that throws shows up here; the printed chart count may be 10 or 14 at this point, both are fine); the checker prints no line for `a08b_pidv2.js`.

---

### Task 3: Chapter 13 — the added pools and the v2 pipeline

**Files:**
- Create: `/home/u2507115/BK_AICUP_2026-1/reports/go_story/parts/c09_pidv2.html` (starts with the comment `<!-- ================= 第 13 章：新增題池與 v2 流程（11-14 加） ================= -->`)

**Interfaces:**
- Consumes: `chart-pid2-arch` from Task 2.
- Produces: cards `card-pid2-task` (sec 25) and `card-pid2-arch` (sec 26) with their lessons; later tasks append to the same file.

- [ ] **Step 1: Write `card-pid2-task` and `sec-25`**

Content spec:
- `div.num`: 第 13 章 · 玩家辨識：新增題池與 v2 流程 · 1／2
- `h3`: 新增題池：每人只有 10 局、題目 1–5 局、黑白都有——原本的統計式做法不夠用
- bullets (fill the bracketed parts from `facts_A.md` section 0 and the facts about METHOD_NOTES §2.29):
  1. 新增兩池各 600 位候選人 × 10 局；題目是 1／2／3／5 局，黑白兩個視角都有 〔實測〕
  2. 原始題池每人 100 局，開局與風格的統計模型就足以認人；每人 10 局時撐不起來：第一版流程在新增池的模擬分數是〔facts 的數字〕
  3. Public 是原始題目與新增題目合在一起平均；原始那一半沿用第 9 章的做法，之後的進展都來自新增那一半 〔只在 facts 有明文時寫第二句〕
- `div.cause`: 問題「每人只有 10 局，四分之一的題目只有 1 局」→ 作法「改學『這盤棋像誰下的』的序列網路」→ 限制「1 局題仍是最大的失分來源」
- figure: a table (≤ 4 rows) comparing 原始題池 and 新增題池 on: 候選人數、每人局數、題目局數、執色.
- chip: 來源 METHOD_NOTES §2.29、§2.37；facts_A 第 0 節
- lesson `sec-25`, title `13.1 測試集的變動與新增題池的結構`: (a) a table of every test-data release with date and what changed (`facts_A.md` section 0 and section 1); (b) the finding that questions of 2／3／5 games have pairwise distinct owners while 1-game questions may share an owner with another question, with the label-free statistics that support it (facts about METHOD_NOTES §2.37); (c) `p.src`.

- [ ] **Step 2: Write `card-pid2-arch` and `sec-26`**

Content spec:
- `div.num`: 第 13 章 · 玩家辨識：新增題池與 v2 流程 · 2／2; layout `report stack` like `card-pid-arch`.
- `h3`: v2 流程：九個成分各算一張分數表 → 加權融合 → 2／3／5 局題軟性配對；逐手序列網路的權重最大
- bullets:
  1. 九個成分：〔the nine component names from the facts about `aicup_pid.v2`〕；逐手序列（seq）的融合權重是〔facts 的數字〕 〔實測〕
  2. seq 是數十個序列網路的平均：每個網路把一盤棋的逐手向量讀成一個身分向量，題目與候選人用餘弦相似度比對
  3. 每個題池先用自己候選人的棋局微調 50 步（不使用題目）：〔+0.0117 [+0.0071, +0.0164]〕 〔實測〕
  4. 2／3／5 局題做軟性一對一配對（λ＝1）：〔+0.0064 [+0.0035, +0.0095]〕 〔實測〕
- `div.cause`: 問題「10 局撐不起每人一個統計模型」→ 作法「共用的序列網路，再對每個題池微調」→ 限制「網路沒看過測試玩家」
- figure: `<div class="pid-fig" id="chart-pid2-arch"></div>`.
- lesson `sec-26`, title `13.2 成分、融合權重與配對`: a table of the nine components (name, what it reads, fusion weight when the facts give it); the definition of the pool fine-tune; the table of the soft-assignment variants under the different owner structures (facts about METHOD_NOTES §2.37); `p.src`.

- [ ] **Step 3: Build and check**

Run the job of Task 1. Expected: `check.py` 0 errors; the checker prints no line for `c09_pidv2.html`; no card-limit message.

---

### Task 4: Chapter 14 — every step of the score

**Files:**
- Modify (append): `.../parts/c09_pidv2.html`

**Interfaces:**
- Consumes: `chart-pid2-climb`.
- Produces: card `card-pid2-climb` (sec 27) and lesson `sec-27`.

- [ ] **Step 1: Write the card and its lesson**

Content spec:
- `div.num`: 第 14 章 · 玩家辨識：分數的每一步 · 1／1 (first card of the chapter: add `data-chtitle`).
- `h3`: Public 0.8612697 → 0.9069275（11-13 至 11-14）：每一步都先在驗證工具上通過才上傳
- bullets (numbers verbatim from the section-2 tables):
  1. 11-13：接上逐手認人網路與題池微調 → 0.8612697；20 個網路＋軟性配對 → 0.8777615 〔實測〕
  2. 11-14 凌晨：含外部帳號的官方網路 → 0.8829460；棋盤對稱增強 → 0.8903230 〔實測〕
  3. 11-14 下午：對稱網路 12 個種子＋4 萬步＋加大 → 0.9016519（當時 1／79） 〔實測〕
  4. 11-14 晚上：加長訓練＋有監督輸入 → 0.9035072 → 0.9069275（當時 1／81） 〔實測〕
- `div.cause`: 問題「一次上傳只回一個數字」→ 作法「擬題池先比較，通過才傳」→ 限制「單次讀數的雜訊約 ±0.002」
- figure: `<div class="bars" id="chart-pid2-climb"></div>`; chip: 來源 facts 第 2 節（METHOD_NOTES §2.42、§2.68、§2.104、§2.111、§2.114）
- lesson `sec-27`, title `14.1 每個上傳檔的組成與讀數`: the complete table of Public readings of the v2 test set (日期時間、檔案、md5、Public、名次／隊數、組成或差別), every row of the section-2 tables of `facts_B.md` and `facts_C.md` and the v2 rows of `facts_A.md`; one paragraph stating which file is the final one and which are the fallbacks (from the facts about METHOD_NOTES §2.114); `p.src`.

- [ ] **Step 2: Build and check** — run the job of Task 1; expected 0 errors, no checker line.

---

### Task 5: Chapter 15 — what made the sequence networks stronger (three cards)

**Files:**
- Modify (append): `.../parts/c09_pidv2.html`

**Interfaces:**
- Consumes: `chart-pid2-fam`, `chart-pid2-single`.
- Produces: cards `card-pid2-data` (28), `card-pid2-long` (29), `card-pid2-sup` (30) and their lessons.

- [ ] **Step 1: `card-pid2-data` and `sec-28`** (first card of chapter 15: `data-chtitle`; `div.num`: 第 15 章 · 玩家辨識：讓序列網路變強的做法 · 1／3)
- `h3`: 外部帳號與棋盤對稱：讓網路學會辨認「沒看過的人」
- bullets:
  1. 測試玩家不在訓練集。加入野狐公開棋譜的 39,683 個帳號，單一網路的融合分數 〔+0.0327〕（對不加外部資料） 〔實測〕
  2. 帳號加到 11.6 萬後不再增加（18.2 萬與 11.6 萬相同）；每帳號只取 4 局明顯較差（〔−0.021～−0.030〕） 〔實測〕
  3. 棋盤對稱增強：訓練時整批套同一個隨機旋轉或鏡像，逐手成分單獨 〔+0.0592 [+0.0472, +0.0719]〕，1 局題 〔0.510 → 0.560〕 〔實測〕
- `div.cause`: 問題「網路記住的是看過的人與看過的座標」→ 作法「更多不同的人，加上隨機對稱」→ 限制「帳號數到 11.6 萬後飽和」
- figure: a table (≤ 5 rows) 外部帳號數 → 效果, using only numbers that the facts state for the same validation instrument; if the facts give them on different instruments, show them in separate rows with the instrument named.
- lesson `sec-28`, title `15.1 外部帳號與對稱增強的細節`: how accounts were selected and anonymised, the overlap check, the composition of a training batch, the definition of the augmentation, and the ablation numbers (facts_B sections 3, 4, 7; facts_C section 5 for the saturation and the 4-games result); `p.src`.

- [ ] **Step 2: `card-pid2-long` and `sec-29`** (`div.num`: … · 2／3)
- `h3`: 訓練更久：單一網路不變，合起來變好
- bullets:
  1. 3–4 萬步的 8 個網路自己就勝過整個原組成的逐手成分：1 局題 〔0.5898 對 0.5819，+0.0079 [+0.0035, +0.0125]〕 〔實測〕
  2. 加進組成：擬題池 〔+0.0038 [+0.0018, +0.0059]〕；權重從 ×9 加到 ×18 都在 +0.005 上下，已到頂 〔實測〕
  3. 遮手、藏開局、dropout、輸入子空間、資料抽樣等變化，每批只有約 +0.001；有效的是訓練長度 〔實測〕
- `div.cause`: 問題「同配方再訓練 20 個只 +0.002」→ 作法「把訓練拉長」→ 限制「同一次訓練的快照彼此太像」
- figure: `<div class="bars" id="chart-pid2-fam"></div>`; chip: 來源 METHOD_NOTES §2.108、§2.110、§2.113
- lesson `sec-29`, title `15.2 訓練長度與各批新成員`: the four-family table and the six-composition table of METHOD_NOTES §2.108, the weight sweep of §2.110, the 20k-versus-30k snapshot comparison (§2.86) and the 40k-step family (§2.90) as the facts give them; `p.src`.

- [ ] **Step 3: `card-pid2-sup` and `sec-30`** (`div.num`: … · 3／3)
- `h3`: 有監督的逐手輸入：換一種看棋的方式，單獨最強、也最互補
- bullets:
  1. 原輸入是 KataGo 主幹向量的隨機投影；改成在訓練玩家上求出的 64 個線性判別方向，取代「全盤」那一半
  2. 官方候選人上的單一網路 1 局題：一般 〔0.5053／0.5234〕 → 有監督輸入 A 型 〔0.5442／0.5574〕 〔實測〕
  3. 加長訓練的 8 個＋有監督的 16 個合起來 〔0.6023〕（一般 12 個 〔0.5734〕）；整條流程 〔+0.0082 [+0.0054, +0.0110]〕 → Public 0.9035072 〔實測〕
  4. 換成 3 萬步完成版：再 〔+0.0035 [+0.0020, +0.0049]〕 → Public 0.9069275 〔實測〕
- `div.cause`: 問題「同一種輸入的網路加再多也飽和」→ 作法「讓一部分成員讀不同的表示」→ 限制「小資料時的 +0.048，放大後只剩一部分」
- figure: `<div class="bars" id="chart-pid2-single"></div>`; chip: 來源 METHOD_NOTES §2.98、§2.109、§2.110、§2.111、§2.113
- lesson `sec-30`, title `15.3 有監督投影的三次檢定與一次改判`: (a) the small-scale test table (probe B, §2.98); (b) the medium-scale table (probe C, §2.109) and the sentence that its verdict "close" was withdrawn because the rule's threshold was finer than the instrument's resolution (§2.110; `facts_C.md` section 8); (c) the table by what the network reads (A／B／C 型, §2.110); (d) the result that networks trained without external accounts are weak alone yet still add to the older composition and add nothing on top of the newer one (§2.113); `p.src`.

- [ ] **Step 4: Build and check** — run the job of Task 1; expected 0 errors, no checker line, no card-limit message.

---

### Task 6: Chapters 16 and 17 — validation, and what did not pass

**Files:**
- Modify (append): `.../parts/c09_pidv2.html`

**Interfaces:**
- Produces: cards `card-pid2-valid` (31), `card-pid2-neg` (32) and their lessons.

- [ ] **Step 1: `card-pid2-valid` and `sec-31`** (`data-chtitle`; `div.num`: 第 16 章 · 玩家辨識：驗證工具與選檔規則 · 1／1)
- `h3`: 四個驗證工具；決定要不要上傳的是「用官方候選人切出的擬題池」
- bullets:
  1. 擬題池：每位官方候選人留 3 局當題目、其餘當參考，跑整條流程，和線上的檔做配對比較（重抽單位是候選人）
  2. 規則：擬題池對線上檔 ≥ +0.002，且舊題池模擬不低於線上檔，才上傳
  3. 五次上傳的校準：擬題池增益合計 +0.0713，Public 新增那一半合計 +0.0583（比例 0.82）；單次比例 0.45–1.95 〔實測〕
  4. 兩個教訓：單一網路的增益常在組合層級消失；門檻不能定得比工具的解析度細 〔推論〕
- `div.cause`: 問題「排行榜一天只能問 5 次」→ 作法「在官方玩家上自己造有答案的題池」→ 限制「擬題池只有 1–3 局題，且每人少 3 局」
- figure: a table with five rows — 檔案、擬題池增益、Public 新增那一半的增益、比例 — rows: `pid_v2is_o1.csv` +0.0132 / +0.0104 / 0.79; `pid_v2w_n8s.csv` +0.0196 / +0.0148 / 0.75; `pid_v2sl_alls.csv` +0.0268 / +0.0227 / 0.85; `pid_v2x_ns.csv` +0.0082 / +0.0037 / 0.45; `pid_v2c2_bs.csv` +0.0035 / +0.0068 / 1.95. (The ratio column uses the values 0.79, 0.75, 0.85, 0.45, 1.95; if the facts print a ratio with different rounding, use the facts' form.)
- lesson `sec-31`, title `16.1 四個工具的定義、事先規則與校準`: definitions of the four instruments (facts_B section 6, facts_C section 6), the pre-registered rules, the calibration table again with one sentence per reading, and the two lessons with their sources (facts_C section 8); `p.src`.

- [ ] **Step 2: `card-pid2-neg` and `sec-32`** (`data-chtitle`; `div.num`: 第 17 章 · 玩家辨識：沒有通過的方向 · 1／1)
- `h3`: 試過而沒有採用的方向：多數是「單一網路變好、整條流程無感」
- bullets (names only on the card; numbers go to the lesson):
  1. 資料面：外部帳號超過 11.6 萬、每帳號只取 4 局、同一盤棋的兩位對局者都當主體、依級別組批
  2. 訓練面：各種遮蔽互相疊加、加大網路、記憶庫與間隔式損失、每批人數與外部佔比等超參數
  3. 比對面：PLDA、樹模型重排、對手側向量、依顏色的原型、推論時的對稱視角、1 局題之間的配對、融合權重再校正
- `div.cause`: 問題「可以試的方向很多」→ 作法「每個方向先寫規則再看結果」→ 限制「只在組合層級判定，單一網路的讀數不算」
- figure: a table (≤ 6 rows) 方向｜結果｜判定 with the six negative results that have the clearest numbers in `facts_C.md` section 5.
- lesson `sec-32`, title `17.1 沒有通過的方向（完整列表）`: one table with EVERY item of section 5 of `facts_B.md` and `facts_C.md` and the v2-related items of `facts_A.md` section 5 — columns 方向｜做法一句話｜數字｜判定與原因｜出處 (fact id) — grouped under `h4` 資料面／訓練面／比對與融合面／輸入特徵; `p.src`.

- [ ] **Step 3: Build and check** — run the job of Task 1; expected 0 errors, no checker line, no card-limit message.

---

### Task 7: Chapter 18 — external resources and disclosure

**Files:**
- Modify (append): `.../parts/c09_pidv2.html`

**Interfaces:**
- Produces: card `card-pid2-ext` (33) and lesson `sec-33`.

- [ ] **Step 1: Write the card and its lesson** (`data-chtitle`; `div.num`: 第 18 章 · 玩家辨識：外部資源與揭露 · 1／1)
- `h3`: 外部資源：KataGo 人類風格網路（抽特徵）與野狐公開棋譜（額外的訓練玩家）
- bullets (every bracket from section 7 of the three facts files):
  1. KataGo 人類風格網路 〔檔名／版本〕：凍結不訓練，只用來把每一手轉成向量與落子機率
  2. 野狐公開棋譜 〔資料集名稱、授權〕：選出 116,420 個帳號、1,261,070 局當額外的訓練玩家；帳號名稱在匯入時就換成加鹽雜湊，並先檢查和競賽資料沒有重疊
  3. 貢獻：不加外部帳號對加 39,683 個帳號，單一網路 〔+0.0327〕 〔實測〕
  4. 沒有使用：fold 0、人工標註、由排行榜回饋得知的答案（只用於評估） 〔each of the three only if a fact supports it〕
- `div.cause`: 問題「規則允許外部資源，但必須揭露」→ 作法「來源、版本、授權、用法、貢獻逐項列出」→ 限制「貢獻量測用的是驗證工具，不是 Private」
- figure: a table with one row per external resource and the five columns 來源網址｜版本或檔名｜授權｜用法｜量到的貢獻; a cell the facts cannot fill is written 「（素材未載）」 and listed in the task report.
- lesson `sec-33`, title `18.1 主辦方的規定與逐項揭露`: the organisers' rule on external resources as quoted in the facts; the selection and anonymisation procedure; the overlap check and its result; the ablation numbers; one paragraph 「生成式 AI 工具」 containing exactly this placeholder sentence and nothing else: 「【待使用者確認】本專案使用生成式 AI 工具的情形與揭露文字。」; `p.src`.

- [ ] **Step 2: Build and check** — run the job of Task 1; expected 0 errors, no checker line. Then confirm by reading the built card that the resource table has no empty cell (each is a value or 「（素材未載）」).

---

### Task 8: Chapter 19 — in progress

**Files:**
- Modify (append): `.../parts/c09_pidv2.html`

**Interfaces:**
- Produces: card `card-pid2-next` (34) and lesson `sec-34`.

- [ ] **Step 1: Write the card and its lesson** (`data-chtitle`; `div.num`: 第 19 章 · 玩家辨識：進行中 · 1／1 with `<span class="tag t-i">11-14 的狀態</span>`)
- `h3`: 到 11／24 截止前還在進行的三條線
- bullets (from the facts about METHOD_NOTES §2.110, §2.115; no result numbers, these runs are unfinished):
  1. 有監督輸入＋4 萬步的 16 個網路（`o13`）
  2. 第二套特徵：把落子機率的「預期落點」放進投影，加上 4 個落子機率的純量（`o14`）；小規模檢定已通過
  3. 每一手的好壞：勝負預期與目數領先在兩手之間的變化（探針 P7）
- no `div.cause` problem strip is required; keep the `deriv-open` button.
- lesson `sec-34`, title `19.1 進行中的檢定與事先寫好的規則`: for each of the three lines the pre-registered rule as the facts state it; one sentence that two further options wait for the participant's decision (fold 0; a second feature extractor), without arguing for them; `p.src`.

- [ ] **Step 2: Verify wording** — in `c09_pidv2.html` none of these strings may occur: `完全解出`, `滿分`, `保證`, `冠軍`, `排名第一`, `一定會`. Run `grep -n -e 完全解出 -e 滿分 -e 保證 -e 冠軍 -e 排名第一 -e 一定會 parts/c09_pidv2.html`; expected: no output. (The statistic 「同第一名」 about two questions sharing their top candidate is not a rank claim and is allowed. If a fact itself uses one of the forbidden words for a measured statement, quote the fact's sentence and add the fact id.)

- [ ] **Step 3: Build and check** — run the job of Task 1; expected 0 errors.

---

### Task 9: Pinned edits to the old chapters (`c08_pid.html`)

**Files:**
- Modify: `/home/u2507115/BK_AICUP_2026-1/reports/go_story/parts/c08_pid.html` (only the elements named below)

**Interfaces:**
- Consumes: facts_A sections 1, 2 and 8 for what happened after 2026-11-06.

- [ ] **Step 1: Divider page `<section … id="pid">`**
- `p.cover-sub` becomes: 原始題池每題 5–20 局黑棋、400 位候選人；11-12 新增的題池每題 1–5 局、600 位候選人、每人只有 10 局
- insert as the FIRST `div.kpi`: value `0.9069275`, label `Public（最終提交檔 pid_v2c2_bs.csv，11-14 21:03；v2 測試集）`
- the existing KPI `0.9937580` keeps its value; its label becomes `原始測試集的 Public（11-08 重發前）：模型直接輸出的最佳檔`
- the existing warn KPI `1.0000000`: its label gets the prefix `原始測試集：`
- `p.metaline` becomes: `2026-11-14 · 程式與紀錄：AI_CUP_2026-2（METHOD_NOTES §2.1–§2.115）；第 8–12 章寫於 11-06（原始測試集），第 13 章起是新增題池`

- [ ] **Step 2: Stale sentences**
- In `card-pid-private`, the bullet that says the new candidate will be uploaded on 11-07 and its `sec-21` counterpart: replace by what the facts say happened to that file and to the test set between 11-07 and 11-12 (facts_A section 1). If the facts do not say, replace by 「（11-06 時的計畫；之後測試集重發，見第 13 章）」.
- Every `div.num` of chapters 8–12 gets the suffix `（原始測試集，11-06）` after the chapter name.
- In `a09_terms.js` do NOT change the old entry; Task 10 appends new ones.

- [ ] **Step 3: Verify** — `grep -n "11-07 上傳" parts/c08_pid.html` prints nothing; `grep -c "原始測試集" parts/c08_pid.html` prints at least 9. Run the job of Task 1; expected 0 errors; the checker's list for `c08_pid.html` is the baseline of Task 1 plus no new line (compare with `build_log.md`).

---

### Task 10: Glossary, speaker script, final build

**Files:**
- Modify (append entries): `/home/u2507115/BK_AICUP_2026-1/reports/go_story/parts/a09_terms.js`
- Modify (the player-ID part only): `/home/u2507115/BK_AICUP_2026-1/reports/go_story/講稿_關鍵.md`

- [ ] **Step 1: Append glossary entries** to the `TERMS` list in the form the file already uses (`['術語', '一行說明'],`), each at most 40 characters of explanation: 擬題池、題池微調、軟性配對、對稱增強、外部帳號、有監督投影、逐手序列網路、官方網路、EMA、fold. Explanations must agree with the definitions in facts section 3 / 6; do not add a term that is already in the list.

- [ ] **Step 2: Speaker script** — under `## 競賽一　玩家辨識`: keep the existing sub-sections, add the note 「（原始測試集，11-06）」 to their headings, fix the two lines that speak of the 11-07 upload as planned, and add one sub-section per new chapter (13–19) with 2–4 spoken sentences each, using only numbers that appear on that chapter's cards.

- [ ] **Step 3: Final build** — run the job of Task 1. Expected last line `PID_BUILD OK` once the baseline list of `c08_pid.html` is added to an allow-list inside `check_numbers.py` named `OLD_BASELINE` (numbers of the 11-06 text that predate the facts library; add them verbatim from `build_log.md`, nothing else). `check.py` must report 0 errors, 14 charts and 35 lessons (25 existing derivation sections plus the 10 new ones).

- [ ] **Step 4: Report back** — write into `build_log.md`: the job id and result of every build; the list "not found" (numbers of this plan that the facts did not contain, and what was done); the list of cells marked 「（素材未載）」; every item of section 9 of the three facts files that touched a card. Do not publish the page; the plan's owner reviews and publishes.

---

## Appendix A — what the plan's owner does after the tasks (not for the implementer)

1. Review `c09_pidv2.html` against the facts (a fresh reviewer on the strongest model), read `build_log.md`, resolve the "not found" list.
2. Ask the participant for the wording of the generative-AI disclosure (Task 7 placeholder) and for approval of the outline.
3. Tell the other session (it owns the rest of the page) that `c09_pidv2.html`, `a08b_pidv2.js` and the listed edits exist, then republish the artifact from `reports/go_story/index.html`.
4. Commit `reports/pid_report/` and this plan in `/home/u2507115/BK_AICUP_2026-2`.

## Appendix B — outline of the organisers' written report (to be written after 2026-11-24 from the facts library; report upload window 2026-11-26 to 2026-12-06)

The organisers require: the algorithm, the preprocessing and training code, the parameters including trained weights, the execution environment, and the sources of external resources with their licence, usage and contribution (ideally with ablations). A facts file `facts_D.md` (2026-11-15 to 2026-11-24, same format) will be added before writing. Sections and their sources:

| Section | Content | Facts to use |
|---|---|---|
| 1 任務與資料 | task, pools, scoring, the four test-data releases | facts_A §0, §1 |
| 2 方法總覽 | the v2 pipeline in one figure; the final file's exact composition | facts_A §3, facts_B §3, facts_C §2–§3, facts_D |
| 3 資料處理 | replay of game records, per-ply vectors, projections, symmetry handling, external accounts' selection and anonymisation | facts_A §3, facts_B §3, §7, facts_C §3 |
| 4 訓練方式 | sequence networks (architecture, loss, batch composition, steps, EMA), the families in the final composition, pool fine-tune | facts_B §3–§4, facts_C §3–§4, facts_D |
| 5 驗證與選檔 | the four instruments, pre-registered rules, calibration against Public | facts_B §6, facts_C §6, §8 |
| 6 結果 | every Public reading with file and composition; Private when published | section-2 tables of all facts files |
| 7 消融與沒有通過的方向 | contribution of each ingredient; the full negative list | facts §4 and §5 of all files |
| 8 外部資源與揭露 | five-field table; unused data; leaderboard-feedback disclosure for the original test set; generative-AI tools | facts §7 of all files; Task 7's table |
| 9 執行環境與重現步驟 | hardware, software versions, commands from data to submission file, where the weights are | to be collected into facts_D |
