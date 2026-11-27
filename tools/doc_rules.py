#!/usr/bin/env python3
"""文件規則（AGENTS.md 第 8 節）：允許的文件清單、工作檔欄位、內部連結、由單一來源產生的成績表格。

用法：
    python -I tools/doc_rules.py --update   依 docs/final_status.json 重寫文件中標記區塊內的成績表格
    python -I tools/doc_rules.py --check    只檢查並列出違規（tools/check_code_rules.py --all 也會呼叫 check()）

設定集中在本檔開頭：ALLOWED_DOCS 是允許存在的文件；LINK_CHECKED 是要檢查內部連結的現行文件；
TABLE_FILES 是含成績表格標記區塊的文件。歷史紀錄（封存的筆記與備忘）只要在允許清單內即可，不檢查連結。
"""
import fnmatch  # 以萬用字元比對允許清單
import json  # 讀取單一來源的成績檔
import re  # 解析連結、工作檔欄位與標記區塊
import subprocess  # 呼叫 git 取得受版控的檔案清單
import sys  # 讀取命令列參數與設定結束碼
import urllib.parse  # 還原連結中的百分比編碼
from pathlib import Path  # 以路徑物件處理檔案位置

ROOT = Path(__file__).resolve().parents[1]  # 儲存庫根目錄（本檔位於 tools/）
STATUS_JSON = "docs/final_status.json"  # 成績與最終檔資訊的唯一來源
DOC_SUFFIXES = (".md", ".rst", ".txt", ".docx", ".pdf")  # 視為文件的副檔名
ALLOWED_DOCS = [  # 允許存在的文件（萬用字元，* 可跨目錄）
    "README.md", "AGENTS.md", "CLAUDE.md",  # 根目錄的說明
    "docs/TASK.md", "docs/PID_FINAL_NUMBERS.md", "docs/BRANCHES.md",  # 現行文件
    "docs/work/*.md",  # 工作檔
    "tools/*_baseline.txt",  # 檢查工具的基準檔（由 check_code_rules.py --write-baseline 產生）
    "docs/records/*", "experiments/*", "results/*",  # 封存區與結果內的紀錄
]  # 結束允許清單
LINK_CHECKED = [  # 要檢查內部連結的現行文件（萬用字元）
    "README.md", "AGENTS.md", "docs/TASK.md", "docs/PID_FINAL_NUMBERS.md", "docs/BRANCHES.md",  # 根目錄與 docs/ 的現行文件
    "docs/work/*.md", "experiments/README.md",  # 工作檔與封存區入口說明
]  # 結束連結檢查清單
TABLE_FILES = ["README.md"]  # 含成績表格標記區塊的文件
ARCHIVE_PREFIXES = ("experiments/", "docs/records/", "results/archive/")  # 封存區（內容以 tools/archive.sha256 鎖定）
WORK_GLOB = "docs/work/*.md"  # 工作檔的位置
WORK_NAME = re.compile(r"^docs/work/\d{4}-\d{2}-\d{2}-[^/]+\.md$")  # 工作檔檔名格式
WORK_STATES = ("進行中", "等待中", "已完成", "已取消")  # 工作檔允許的狀態
BEGIN, END = "<!-- final-status:begin -->", "<!-- final-status:end -->"  # 成績表格標記區塊的開頭與結尾
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")  # Markdown 連結的目標


def tracked_files():  # 取得受版控（含尚未 add）的檔案清單
    """回傳 git 追蹤中且實際存在的檔案相對路徑。"""
    out = subprocess.run(  # 執行 git ls-files
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],  # 含新檔，排除被忽略的檔
        cwd=ROOT, capture_output=True, check=True,  # 在根目錄執行並擷取輸出
    )  # 結束 git 呼叫
    return sorted(n for n in out.stdout.decode("utf-8").split("\0") if n and (ROOT / n).is_file())  # 去空字串並排序


def matches(path, patterns):  # 判斷路徑是否符合任一萬用字元
    """回傳 path 是否符合 patterns 中的任一項（** 可跨目錄）。"""
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)  # fnmatch 的 * 本來就會跨越斜線


def render(status):  # 由成績檔產生 Markdown 表格
    """把 docs/final_status.json 的 rows 轉成兩欄表格。"""
    lines = ["| 項目 | 內容 |", "| --- | --- |"]  # 表頭與分隔列
    for row in status["rows"]:  # 逐列輸出
        lines.append(f"| {row['label']} | {row['value']} |")  # 一列一個事實
    return "\n".join(lines)  # 以換行串接


def block_text(text):  # 取出標記區塊內的表格
    """回傳標記區塊之間的內容；找不到標記時回傳 None。"""
    m = re.search(re.escape(BEGIN) + r"\n(.*?)\n" + re.escape(END), text, re.DOTALL)  # 非貪婪比對區塊
    return m.group(1) if m else None  # 沒有區塊就回傳 None


def update():  # 重寫所有表格區塊
    """依成績檔更新 TABLE_FILES 內的標記區塊，回傳更新的檔案數。"""
    table = render(json.loads((ROOT / STATUS_JSON).read_text("utf-8")))  # 產生最新表格
    changed = 0  # 計算更新的檔案數
    for name in TABLE_FILES:  # 逐檔處理
        path = ROOT / name  # 檔案路徑
        text = path.read_text("utf-8")  # 讀取內容
        if block_text(text) is None:  # 沒有標記區塊
            print(f"{name}: 找不到 {BEGIN} 標記，略過")  # 提示需要先放標記
            continue  # 換下一個檔
        new = re.sub(re.escape(BEGIN) + r"\n.*?\n" + re.escape(END), lambda _: f"{BEGIN}\n{table}\n{END}", text, flags=re.DOTALL)  # 替換區塊內容
        if new != text:  # 內容有變動才寫回
            path.write_text(new, "utf-8")  # 寫回檔案
            changed += 1  # 計數
    return changed  # 回傳更新數


def check_work_file(path):  # 檢查一個工作檔的檔名與三個欄位
    """回傳違規訊息列表。"""
    problems = []  # 收集違規
    if not WORK_NAME.match(path):  # 檔名不是 YYYY-MM-DD-主題.md
        problems.append(f"{path}: 工作檔檔名要是 docs/work/YYYY-MM-DD-主題.md")  # 回報檔名
    head = "\n".join((ROOT / path).read_text("utf-8").splitlines()[:12])  # 只看開頭幾行
    owner = re.search(r"^- \*\*負責人：\*\* *(\S.*)$", head, re.M)  # 負責人欄位
    state = re.search(r"^- \*\*狀態：\*\* *(\S+)", head, re.M)  # 狀態欄位
    stamp = re.search(r"^- \*\*最後更新：\*\* *(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s*$", head, re.M)  # 最後更新欄位
    if not owner:  # 缺負責人
        problems.append(f"{path}: 缺少「負責人」欄位")  # 回報
    if not state or state.group(1) not in WORK_STATES:  # 缺狀態或狀態不在允許值內
        problems.append(f"{path}: 「狀態」要是 {'／'.join(WORK_STATES)} 之一")  # 回報
    if not stamp:  # 缺最後更新或格式不對
        problems.append(f"{path}: 「最後更新」要是 YYYY-MM-DD HH:MM")  # 回報
    return problems  # 回傳違規


def check_links(path, tracked, dirs):  # 檢查一個現行文件的內部連結
    """回傳指向不存在檔案或目錄的相對連結。"""
    problems = []  # 收集違規
    base = (ROOT / path).parent  # 相對連結以文件所在目錄為基準
    for n, line in enumerate((ROOT / path).read_text("utf-8").splitlines(), 1):  # 逐行掃描
        for target in LINK.findall(line):  # 每一個連結目標
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):  # 外部網址或頁內錨點
                continue  # 不檢查
            rel = urllib.parse.unquote(target.split("#", 1)[0]).rstrip("/")  # 去掉錨點並還原編碼
            if not rel:  # 只有錨點
                continue  # 不檢查
            full = (base / rel).resolve()  # 換算成絕對路徑
            try:  # 連結可能指到儲存庫外
                key = full.relative_to(ROOT).as_posix()  # 轉回儲存庫相對路徑
            except ValueError:  # 指到儲存庫外（例如並排的另一個儲存庫）
                continue  # 不檢查
            if key not in tracked and key not in dirs:  # 既不是檔案也不是目錄
                problems.append(f"{path}:{n}: 連結的目標不存在：{target}")  # 回報
    return problems  # 回傳違規


def check(paths=None):  # 執行全部文件規則
    """回傳違規訊息列表；paths 為受版控檔案清單（省略時自動取得）。"""
    paths = tracked_files() if paths is None else paths  # 取得檔案清單
    tracked = set(paths)  # 方便查詢
    dirs = {p.rsplit("/", i)[0] for p in paths for i in range(1, p.count("/") + 1)}  # 所有上層目錄
    problems = []  # 收集違規
    for p in paths:  # 檢查允許清單
        if p.endswith(DOC_SUFFIXES) and not matches(p, ALLOWED_DOCS):  # 文件不在允許清單內
            problems.append(f"{p}: 不在允許的文件清單內（tools/doc_rules.py 的 ALLOWED_DOCS）")  # 回報
    for p in paths:  # 檢查工作檔
        if fnmatch.fnmatchcase(p, WORK_GLOB):  # 位於 docs/work/
            problems += check_work_file(p)  # 檢查欄位
    for p in paths:  # 檢查現行文件的連結
        if p.endswith(".md") and matches(p, LINK_CHECKED):  # 只檢查現行文件
            problems += check_links(p, tracked, dirs)  # 檢查內部連結
    if (ROOT / STATUS_JSON).exists():  # 有成績檔時檢查表格是否同步
        table = render(json.loads((ROOT / STATUS_JSON).read_text("utf-8")))  # 應有的表格
        for name in TABLE_FILES:  # 逐檔比對
            current = block_text((ROOT / name).read_text("utf-8")) if (ROOT / name).exists() else None  # 目前的表格
            if current != table:  # 不一致或缺少區塊
                problems.append(f"{name}: 成績表格與 {STATUS_JSON} 不一致，請執行 python -I tools/doc_rules.py --update")  # 回報
    else:  # 缺少成績檔
        problems.append(f"{STATUS_JSON} 不存在")  # 回報
    return problems  # 回傳違規


def main(argv):  # 命令列進入點
    """--update 更新表格；--check 檢查並以結束碼回報。"""
    if argv[:1] == ["--update"]:  # 更新模式
        print(f"已更新 {update()} 個檔案的成績表格")  # 回報更新數
        return 0  # 正常結束
    problems = check()  # 檢查模式
    for line in problems:  # 列出違規
        print(line)  # 一行一筆
    print(f"doc_rules：{len(problems)} 項違規")  # 摘要
    return 1 if problems else 0  # 有違規就回傳失敗


if __name__ == "__main__":  # 直接執行時
    sys.exit(main(sys.argv[1:]))  # 以檢查結果作為結束碼
