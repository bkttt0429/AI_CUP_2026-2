# Deletion of the checkpoints of the round-10 families o64-o71 (64 files + their 64 family links).
# The user's word 2026-11-20 00:0x: "同意刪 o64–o71 的檢查點". Dry run unless the argument is "go".
import os, re, sys, json, glob, collections, time
go = len(sys.argv) > 1 and sys.argv[1] == 'go'
R2 = '/home/u2507115/BK_AICUP_2026-2'
SN = '/work/u2507115/pid_v2/seqnet'
ck = re.compile(r'ckpt_-1_o(6[4-9]|7[01])(As[0-5]|Cs[01]|Ss[0-3]|SMs[0-3]|Fs[0-7])(k\d+)?e\.pt$')
todo = [os.path.join(SN, f) for f in sorted(os.listdir(SN)) if ck.match(f)]
S = set(todo)
reg = [p for p in todo if not os.path.islink(p)]
lnk = [p for p in todo if os.path.islink(p)]
fam = collections.Counter(re.search(r'ckpt_-1_(o\d+)', p).group(1) for p in reg)
size = sum(os.lstat(p).st_size for p in reg)
print('regular', len(reg), dict(fam), 'links', len(lnk), 'MB', round(size / 2**20))
bad_h = [p for p in reg if os.lstat(p).st_nlink != 1]
bad_l = [(p, os.path.realpath(p)) for p in lnk if os.path.exists(os.path.realpath(p)) and os.path.realpath(p) not in S]
inb = [p for p in (os.path.join(SN, f) for f in os.listdir(SN)) if os.path.islink(p) and p not in S and os.path.realpath(p) in S]
print('hard links', len(bad_h), 'links leaving the set', len(bad_l), 'inbound links from outside', len(inb))
# documents that reference these networks
tag = re.compile(r'_net-1_(o(6[4-9]|7[01])F)s\d+e')
docs = collections.defaultdict(set)
for d in glob.glob(R2 + '/results/pid_fusion_*.json'):
    try: j = json.load(open(d))
    except Exception: continue
    for x in j.get('seq_official', []):
        m = tag.search(os.path.basename(x[0]))
        if m: docs[os.path.basename(d)[11:-5]].add(m.group(1))
print('documents that reference them:', {k: len(v) for k, v in sorted(docs.items())})
keep = ('v2fin9_w', 'v2fin11_c', 'w9_0144', 'w11c', 'w11p', 'w11q', 'zz_w9chk', 'v2fin7_a', 'zz_r7a', 'v2fin4_f', 'zz_r4f', 'v2fin_x')
hit = [k for k in keep if k in docs]
miss = [k for k in keep if not os.path.exists(R2 + '/results/pid_fusion_%s.json' % k)]
print('final / fallback / candidate documents that reference them:', hit, '| documents missing:', miss)
ok = len(reg) == 64 and len(lnk) == 64 and not bad_h and not bad_l and not inb and not hit and not miss
print('checks', 'PASSED' if ok else 'FAILED')
if go:
    assert ok
    rec = R2 + '/results/final/deleted_o64_71_ckpt_20261120.txt'
    assert not os.path.exists(rec)
    with open(rec, 'w') as fh:
        fh.write('# deleted %s on the user\'s word "同意刪 o64–o71 的檢查點" (round-10 families, in no composition; METHOD_NOTES 2.232)\n' % time.strftime('%F %T'))
        for p in todo: fh.write('%s\t%s\t%d\n' % ('link' if os.path.islink(p) else 'file', p, os.lstat(p).st_size))
    for p in lnk + reg: os.unlink(p)
    print('deleted', len(todo), 'record', rec)
