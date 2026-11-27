#!/usr/bin/env python3
"""檢查本儲存庫的程式與文件規則（AGENTS.md 第 5、6、8 節與目錄規則）。

用法：
    python -I tools/check_code_rules.py --all          檢查全專案
    python -I tools/check_code_rules.py 檔案 ...        只檢查指定的檔案（中文註解與重複函式）
    python -I tools/check_code_rules.py --write-baseline 重新產生重複函式與封存區的基準檔
    python -I tools/check_code_rules.py --push-range 舊 新  檢查一次 push 到 main 的範圍（GitHub Actions 使用）

檢查項目：
1. 中文註解：src/、scripts/、tools/ 底下的 .py、.sh、.sbatch，每一行程式碼都要在同一行行尾有繁體中文註解；
   整行註解要有中文；新寫的 docstring 要有中文（tools/docstrings_baseline.txt 列出比賽時既有、維持原文的
   docstring）；不可用反斜線續行。docstring 的每一行、多行字串與 heredoc 的內部行、空行、shebang、
   編碼宣告與 #SBATCH 指令行不需要註解。
2. 重複函式：上述範圍內，本體相同（忽略名稱與 docstring）且至少 3 個敘述的函式；
   tools/duplicates_baseline.txt 列出的既有重複不算失敗。
3. 封存區：tools/doc_rules.py 的 ARCHIVE_PREFIXES 底下的檔案須與 tools/archive.sha256 完全一致（封存後不再修改）。
4. 文件規則：交給 tools/doc_rules.py（允許的文件、工作檔欄位、內部連結、數字表格）。
"""
import ast  # 解析 Python 語法樹，找 docstring 與函式本體
import hashlib  # 計算封存檔案與函式本體的雜湊
import io  # 把字串包成檔案物件給 tokenize 使用
import re  # 比對中文字元、heredoc 與續行
import subprocess  # 呼叫 git 比較 push 前後的內容
import sys  # 讀取命令列參數與設定結束碼
import tokenize  # 逐一取得 Python 的註解與程式記號
from pathlib import Path  # 以路徑物件找出本檔所在的資料夾

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I 不會把本檔所在的 tools/ 放進匯入路徑，須自行加入

import doc_rules  # noqa: E402  同目錄的文件規則；也提供儲存庫根目錄與受版控檔案清單

ROOT = doc_rules.ROOT  # 儲存庫根目錄
SCOPE = ("src/", "scripts/", "tools/")  # 套用中文註解與重複函式規則的目錄
CODE_SUFFIXES = (".py", ".sh", ".sbatch")  # 需要檢查的程式副檔名
CJK = re.compile(r"[㐀-鿿豈-﫿]")  # 判斷是否含中文字的正規表示式
HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")  # 找出 shell heredoc 的結束標記（排除 <<< here-string）
DUP_BASELINE = ROOT / "tools" / "duplicates_baseline.txt"  # 既有重複函式的基準檔
DOC_BASELINE = ROOT / "tools" / "docstrings_baseline.txt"  # 比賽時既有、維持原文的英文 docstring
ARCHIVE_MANIFEST = ROOT / "tools" / "archive.sha256"  # 封存區檔案的雜湊清單
MIN_DUP_STATEMENTS = 3  # 函式本體至少幾個敘述才列入重複檢查


def in_scope(path):  # 判斷檔案是否屬於中文註解規則的範圍
    """src/、scripts/、tools/ 底下的 .py、.sh、.sbatch 才需要檢查。"""
    return path.startswith(SCOPE) and path.endswith(CODE_SUFFIXES)  # 同時符合目錄與副檔名


def docstring_lines(tree):  # 找出所有 docstring 佔用的行號
    """回傳 docstring 涵蓋的行號集合，以及缺少中文的 docstring（起始行, 限定名稱）列表。"""
    lines, bad = set(), []  # 收集 docstring 行號與不合格的 docstring

    def visit(node, prefix):  # 依原始碼順序走訪，同時組出限定名稱
        """處理一個節點的 docstring，再遞迴處理子節點。"""
        named = isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))  # 類別或函式才有名稱
        name = f"{prefix}.{node.name}" if named and prefix else (node.name if named else prefix)  # 限定名稱，例如 Model.fit
        body = getattr(node, "body", None)  # 取得敘述列表（沒有時為 None）
        has_doc = isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body  # 只有這四種節點有 docstring
        first = body[0] if has_doc else None  # 第一個敘述
        if first is not None and isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):  # 第一個敘述是字串常數，即 docstring
            lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))  # docstring 的每一行（含結尾引號那一行）都不需要註解
            if not CJK.search(first.value.value):  # docstring 沒有中文
                bad.append((first.lineno, name or "<module>"))  # 記下起始行與限定名稱
        for child in ast.iter_child_nodes(node):  # 遞迴處理子節點
            visit(child, name)  # 子節點沿用目前的限定名稱

    visit(tree, "")  # 從模組開始
    return lines, bad  # 回傳兩個結果


def known_docstrings():  # 讀取既有英文 docstring 的基準檔
    """回傳基準檔記錄的「檔案:限定名稱」集合；只讀一次。"""
    if not hasattr(known_docstrings, "cache"):  # 第一次呼叫才讀檔
        text = DOC_BASELINE.read_text("utf-8") if DOC_BASELINE.exists() else ""  # 沒有基準檔時視為空
        known_docstrings.cache = {entry.strip() for entry in text.splitlines() if entry.strip() and not entry.startswith("#")}  # 去掉說明與空行
    return known_docstrings.cache  # 回傳快取的集合


def check_python(path, text):  # 檢查一個 Python 檔的中文註解規則
    """回傳違規訊息列表（每筆含行號）。"""
    problems = []  # 收集違規訊息
    try:  # 先確認語法正確
        tree = ast.parse(text)  # 解析語法樹
    except SyntaxError as err:  # 語法錯誤時無法繼續檢查
        return [f"{path}:{err.lineno}: 語法錯誤，無法檢查註解"]  # 直接回報
    doc_lines, bad_docs = docstring_lines(tree)  # 取得 docstring 的行號與缺中文者
    known = known_docstrings()  # 比賽時既有、維持原文的 docstring
    problems += [f"{path}:{n}: docstring 要寫繁體中文（{q}）" for n, q in bad_docs if f"{path}:{q}" not in known]  # 只有新的 docstring 算違規
    code_lines, comments, exempt = set(), {}, set()  # 程式行、註解內容、豁免行
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):  # 逐一讀取記號
        kind, (srow, _), (erow, _) = tok.type, tok.start, tok.end  # 取出記號種類與起訖行
        if kind == tokenize.COMMENT:  # 註解記號
            comments[srow] = tok.string  # 記下該行的註解文字
        elif kind in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):  # 結構記號
            continue  # 不算程式內容
        else:  # 其他記號都是程式內容
            code_lines.update(range(srow, erow + 1))  # 記號跨越的每一行都是程式行
            if erow > srow:  # 跨行的記號（多行字串）
                exempt.update(range(srow, erow))  # 只有最後一行能接註解，前面的行豁免
    lines = text.splitlines()  # 逐行內容，用來檢查續行
    for n in sorted(code_lines - doc_lines - exempt):  # 逐一檢查需要註解的程式行
        note = comments.get(n)  # 取得同一行的註解
        if note is None:  # 沒有行尾註解
            problems.append(f"{path}:{n}: 缺少行尾中文註解")  # 回報缺註解
        elif not CJK.search(note):  # 有註解但沒有中文
            problems.append(f"{path}:{n}: 行尾註解要寫繁體中文")  # 回報註解不是中文
    for n, note in comments.items():  # 檢查整行註解
        if n in code_lines:  # 行尾註解已在上面檢查
            continue  # 略過
        if n == 1 and note.startswith("#!"):  # 第一行的 shebang
            continue  # 豁免
        if n <= 2 and re.match(r"#.*coding[:=]", note):  # 編碼宣告
            continue  # 豁免
        if not CJK.search(note):  # 整行註解沒有中文
            problems.append(f"{path}:{n}: 整行註解要寫繁體中文")  # 回報
    for n in sorted(code_lines - exempt):  # 檢查反斜線續行
        if n <= len(lines) and lines[n - 1].rstrip().endswith("\\") and n not in doc_lines:  # 行末是反斜線
            problems.append(f"{path}:{n}: 不可用反斜線續行，請改用括號")  # 回報續行
    return problems  # 回傳全部違規


def shell_comment_start(line, quote=""):  # 找出 shell 單行中註解開始的位置
    """回傳（註解 # 的索引或 -1, 行尾仍未關閉的引號）。quote 是上一行結束時仍未關閉的引號；
    會略過引號內、$# 與 ${# 的 #。"""
    i = 0  # 掃描位置
    while i < len(line):  # 逐字掃描
        ch = line[i]  # 目前字元
        if ch == "\\" and quote != "'":  # 反斜線跳脫下一個字元（單引號內除外）
            i += 2  # 跳過被跳脫的字元
            continue  # 繼續掃描
        if quote:  # 在引號內
            if ch == quote:  # 遇到對應的結束引號
                quote = ""  # 離開引號
        elif ch in "'\"":  # 遇到開始引號
            quote = ch  # 進入引號
        elif ch == "#" and (i == 0 or line[i - 1] in " \t;|&(") and line[max(0, i - 1):i + 1] not in ("$#", "{#"):  # 位於字首的 #
            return i, quote  # 這是註解的開頭
        i += 1  # 前進一個字元
    return -1, quote  # 整行沒有註解


def check_shell(path, text):  # 檢查一個 shell 檔的中文註解規則
    """回傳違規訊息列表。heredoc 內容、跨行引號字串的內部行與 #SBATCH 指令行豁免。"""
    problems, heredoc_end, quote = [], None, ""  # 違規訊息、所在 heredoc 的結束標記、跨行未關閉的引號
    for n, raw in enumerate(text.splitlines(), 1):  # 逐行檢查
        line = raw.rstrip()  # 去掉行尾空白
        if heredoc_end is not None:  # 正在 heredoc 內容中
            if line.strip() == heredoc_end:  # 遇到結束標記（這一行本身不能加註解）
                heredoc_end = None  # 離開 heredoc
            continue  # heredoc 內容與結束標記行都不檢查
        start_quote = quote  # 這一行開頭是否仍在上一行的字串內
        pos, quote = shell_comment_start(line, start_quote)  # 找註解位置並更新引號狀態
        if quote:  # 這一行結束時仍在字串內（字串開頭行或內部行）
            continue  # 註解放不進字串，豁免
        stripped = line.strip()  # 去掉前後空白後的內容
        if not stripped:  # 空行
            continue  # 豁免
        if n == 1 and stripped.startswith("#!"):  # 第一行的 shebang
            continue  # 豁免
        if stripped.startswith("#SBATCH"):  # Slurm 指令行
            continue  # 豁免
        if stripped.startswith("#") and not start_quote:  # 整行註解
            if not CJK.search(stripped):  # 沒有中文
                problems.append(f"{path}:{n}: 整行註解要寫繁體中文")  # 回報
            continue  # 整行註解檢查完畢
        if pos < 0:  # 程式行沒有註解
            problems.append(f"{path}:{n}: 缺少行尾中文註解")  # 回報缺註解
        elif not CJK.search(line[pos:]):  # 註解沒有中文
            problems.append(f"{path}:{n}: 行尾註解要寫繁體中文")  # 回報
        code = line[:pos] if pos >= 0 else line  # 去掉註解後的程式部分
        if code.rstrip().endswith("\\"):  # 反斜線續行
            problems.append(f"{path}:{n}: 不可用反斜線續行，請改用陣列或括號")  # 回報
        match = HEREDOC.search(code)  # 這一行是否開始 heredoc
        if match:  # 有 heredoc
            heredoc_end = match.group(2)  # 記下結束標記
    return problems  # 回傳全部違規


def function_bodies(path, text):  # 擷取一個檔案內所有函式的正規化本體
    """回傳 (雜湊, 檔案:函式名稱:行號) 列表；本體敘述少於下限者略過。"""
    try:  # 解析失敗時略過此檔
        tree = ast.parse(text)  # 解析語法樹
    except SyntaxError:  # 語法錯誤由註解檢查回報
        return []  # 不列入重複檢查
    found = []  # 收集結果
    for node in ast.walk(tree):  # 走訪所有節點
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):  # 只看函式定義
            continue  # 其他節點略過
        body = list(node.body)  # 複製本體敘述列表
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):  # 開頭是 docstring
            body = body[1:]  # 去掉 docstring
        if len(body) < MIN_DUP_STATEMENTS:  # 本體太短
            continue  # 不列入檢查
        shape = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)  # 本體的結構字串（不含行號）
        digest = hashlib.sha256((ast.dump(node.args) + shape).encode()).hexdigest()[:16]  # 參數與本體一起取雜湊
        found.append((digest, f"{path}:{node.name}:{node.lineno}"))  # 記下雜湊與位置
    return found  # 回傳結果


def duplicate_groups(paths):  # 找出範圍內本體相同的函式群組
    """回傳 {雜湊: [位置, ...]}，只含出現兩次以上者。"""
    groups = {}  # 以雜湊分組
    for p in paths:  # 逐檔處理
        if not p.endswith(".py"):  # 只檢查 Python
            continue  # 略過 shell
        for digest, where in function_bodies(p, (ROOT / p).read_text("utf-8")):  # 取得每個函式的雜湊
            groups.setdefault(digest, []).append(where)  # 加入同組
    return {d: w for d, w in groups.items() if len(w) > 1}  # 只保留重複的組


def group_key(where_list):  # 產生不受行號影響的群組識別字串
    """以「檔案:函式」排序後組成識別字串，行號變動不影響基準比對。"""
    return " | ".join(sorted(w.rsplit(":", 1)[0] for w in where_list))  # 去掉行號後排序串接


def archive_digest(paths):  # 計算封存區每個檔案的雜湊
    """回傳 {路徑: sha256}，只含封存區（doc_rules.ARCHIVE_PREFIXES）底下的檔案。"""
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths if p.startswith(doc_rules.ARCHIVE_PREFIXES)}  # 逐檔計算


def write_baselines(paths):  # 依目前狀態重寫兩個基準檔
    """產生 tools/duplicates_baseline.txt 與 tools/archive.sha256。"""
    dups = duplicate_groups([p for p in paths if in_scope(p)])  # 目前的重複群組
    lines = sorted(group_key(w) for w in dups.values())  # 轉成識別字串
    header = "# 既有的重複函式（最終流程維持比賽時的程式，不重構）；新增的重複會讓檢查失敗\n"  # 檔案說明
    DUP_BASELINE.write_text(header + "".join(f"{k}\n" for k in lines), "utf-8")  # 寫入重複基準
    digests = archive_digest(paths)  # 封存區雜湊
    ARCHIVE_MANIFEST.write_text("".join(f"{h}  {p}\n" for p, h in sorted(digests.items())), "utf-8")  # 寫入封存清單
    docs = []  # 目前範圍內沒有中文的 docstring
    for p in paths:  # 逐檔收集
        if in_scope(p) and p.endswith(".py"):  # 只看範圍內的 Python
            docs += [f"{p}:{q}" for _, q in docstring_lines(ast.parse((ROOT / p).read_text("utf-8")))[1]]  # 檔案:限定名稱
    note = "# 比賽時既有的英文 docstring（多支程式以 __doc__ 產生 --help），維持原文；新寫的 docstring 要用繁體中文\n"  # 檔案說明
    DOC_BASELINE.write_text(note + "".join(f"{d}\n" for d in sorted(set(docs))), "utf-8")  # 寫入 docstring 基準
    print(f"已寫入 {len(lines)} 組重複、{len(digests)} 個封存檔、{len(set(docs))} 個既有 docstring 的基準")  # 回報結果


def check_archive(paths):  # 確認封存區與清單一致
    """回傳違規訊息列表：新增、缺少或內容不同的檔案。"""
    if not ARCHIVE_MANIFEST.exists():  # 沒有清單
        return [] if not any(p.startswith(doc_rules.ARCHIVE_PREFIXES) for p in paths) else ["tools/archive.sha256 不存在"]  # 有封存區就必須有清單
    want = {}  # 清單記錄的雜湊
    for line in ARCHIVE_MANIFEST.read_text("utf-8").splitlines():  # 逐行讀清單
        if line.strip():  # 略過空行
            digest, name = line.split("  ", 1)  # 拆出雜湊與路徑
            want[name] = digest  # 存入對照表
    have = archive_digest(paths)  # 目前的雜湊
    problems = [f"{p}: 封存區不可新增檔案" for p in sorted(set(have) - set(want))]  # 多出的檔案
    problems += [f"{p}: 封存區的檔案不見了" for p in sorted(set(want) - set(have))]  # 少掉的檔案
    problems += [f"{p}: 封存區的檔案被修改" for p in sorted(set(want) & set(have)) if want[p] != have[p]]  # 內容不同
    return problems  # 回傳違規


def git_out(*args):  # 執行 git 並回傳標準輸出
    """回傳 git 指令的輸出文字（去掉前後空白）。"""
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)  # 在根目錄執行
    return done.stdout.decode("utf-8").strip()  # 解碼並去掉空白


def check_push(before, after):  # 檢查一次 push 到 main 的內容（AGENTS.md 第 8 節第 3、4 點）
    """直接 push 到 main 只能改 docs/work/ 的工作檔；PR 的一般合併與引入本規則的那次 push 不受限。"""
    if set(before) == {"0"}:  # 新建分支時沒有比較基準
        return []  # 不檢查
    if not git_out("ls-tree", "--name-only", before, "tools/check_code_rules.py"):  # push 之前還沒有本規則
        return []  # 引入規則的那次 push 不受限
    parents = git_out("rev-list", "--parents", "-n", "1", after).split()  # 最新 commit 與其父節點
    subject = git_out("log", "-1", "--format=%s", after)  # 最新 commit 的標題
    if len(parents) > 2 and subject.startswith("Merge pull request"):  # PR 的一般合併
        return []  # 走 PR 的變更不受限
    changed = [p for p in git_out("diff", "--name-only", before, after).splitlines() if p]  # 這次 push 改到的檔案
    return [f"{p}: 直接 push 到 main 只能修改 docs/work/ 的工作檔，其他變更請走 pull request" for p in changed if not p.startswith("docs/work/")]  # 回報工作檔以外的變更


def main(argv):  # 命令列進入點
    """依參數執行檢查，有違規時回傳 1。"""
    if argv[:1] == ["--push-range"]:  # GitHub Actions 檢查 push 範圍的模式
        problems = check_push(argv[1], argv[2])  # 比較 push 前後
        for line in problems:  # 列出違規
            print(line)  # 一行一筆
        return 1 if problems else 0  # 有違規就回傳失敗
    paths = doc_rules.tracked_files()  # 取得全部受版控檔案
    if argv[:1] == ["--write-baseline"]:  # 重寫基準檔模式
        write_baselines(paths)  # 產生基準
        return 0  # 正常結束
    check_all = argv[:1] == ["--all"] or not argv  # 沒給檔案時視同全部
    targets = [p for p in paths if in_scope(p)] if check_all else [p for p in argv if in_scope(p)]  # 要檢查註解的檔案
    problems = []  # 收集全部違規
    for p in targets:  # 逐檔檢查中文註解
        text = (ROOT / p).read_text("utf-8")  # 讀取內容
        problems += check_python(p, text) if p.endswith(".py") else check_shell(p, text)  # 依類型檢查
    known = set()  # 基準檔記錄的既有重複
    if DUP_BASELINE.exists():  # 有基準檔時讀入
        known = {entry.strip() for entry in DUP_BASELINE.read_text("utf-8").splitlines() if entry.strip() and not entry.startswith("#")}  # 去掉說明與空行
    for where in duplicate_groups([p for p in paths if in_scope(p)]).values():  # 全範圍的重複群組
        if group_key(where) not in known:  # 不在基準內的才是新增的重複
            problems.append("重複函式（請抽到共用模組）：" + ", ".join(where))  # 回報
    if check_all:  # 全專案模式才檢查封存區與文件
        problems += check_archive(paths)  # 封存區
        problems += doc_rules.check(paths)  # 文件規則檢查（各儲存庫自己的設定）
    for line in problems:  # 輸出所有違規
        print(line)  # 一行一筆
    print(f"check_code_rules：{len(targets)} 個程式檔，{len(problems)} 項違規")  # 摘要
    return 1 if problems else 0  # 有違規就回傳失敗


if __name__ == "__main__":  # 直接執行時
    sys.exit(main(sys.argv[1:]))  # 以檢查結果作為結束碼
