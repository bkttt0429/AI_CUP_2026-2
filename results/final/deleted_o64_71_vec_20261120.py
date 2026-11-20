# Deletion of the vectors of the round-10 families o64F-o71F (their checkpoints were deleted on 11-20 00:08).
# The user's word 2026-11-20 23:3x: "好可以刪掉" (answer to: delete the leftover o64-o71 vectors?). Dry run unless "go".
import os, re, sys, json, glob, collections, time
go = len(sys.argv) > 1 and sys.argv[1] == 'go'
R2 = '/home/u2507115/BK_AICUP_2026-2'
pat = re.compile(r'(official|offsim|oldsim)_.*_net-1_o(6[4-9]|7[01])Fs\d+e_t\d+\.npz$')
roots = ['/work/u2507115/pid_v2', R2 + '/data']
todo = []
for root in roots:
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if '_net-1_o' in f and re.search(r'_net-1_o(6[4-9]|7[01])F', f):
                p = os.path.join(dp, f)
                assert pat.match(f), ('unexpected name', p)
                todo.append(p)
S = set(todo)
reg = [p for p in todo if not os.path.islink(p)]
lnk = [p for p in todo if os.path.islink(p)]
by = collections.Counter(os.path.dirname(p) for p in todo)
size = collections.Counter()
for p in reg: size[os.path.dirname(p)] += os.lstat(p).st_size
for d, n in sorted(by.items()): print('  %-58s %5d names %6d MB' % (d, n, size[d] / 2**20))
bad_h = [p for p in reg if os.lstat(p).st_nlink != 1]
bad_l = [(p, os.path.realpath(p)) for p in lnk if os.path.exists(os.path.realpath(p)) and os.path.realpath(p) not in S]
inb = []
for d in list(by) + [R2 + '/data/pid_v2_f0tmp/ft', R2 + '/data/pid_v2_f0tmp/old', R2 + '/data/pid_v2_seq_ft_offsim', R2 + '/data/pid_v2_seq_ft_oldsim']:
    if not os.path.isdir(d): continue
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.islink(p) and p not in S and os.path.realpath(p) in S: inb.append(p)
print('regular', len(reg), 'links', len(lnk), 'total MB', round(sum(size.values()) / 2**20), '| hard links', len(bad_h), 'links leaving the set', len(bad_l), 'inbound links', len(inb))
tag = re.compile(r'_net-1_(o(6[4-9]|7[01])F)s\d+e')
docs = set()
for d in glob.glob(R2 + '/results/pid_fusion_*.json'):
    try: j = json.load(open(d))
    except Exception: continue
    if any(tag.search(os.path.basename(x[0])) for x in j.get('seq_official', [])): docs.add(os.path.basename(d)[11:-5])
keep = ('v2fin11_c', 'v2fin9_w', 'w9_0144', 'w11c', 'w11p', 'w11q', 'zz_w9chk', 'zz_fin11_re', 'v2fin7_a', 'zz_r7a', 'v2fin4_f', 'zz_r4f', 'v2fin_x')
print('documents that reference them:', sorted(docs), '| finals among them:', [k for k in keep if k in docs])
ok = len(todo) > 0 and not bad_h and not bad_l and not inb and not [k for k in keep if k in docs]
print('checks', 'PASSED' if ok else 'FAILED')
if go:
    assert ok
    rec = R2 + '/results/final/deleted_o64_71_vec_20261120.txt'
    assert not os.path.exists(rec)
    with open(rec, 'w') as fh:
        fh.write('# deleted %s on the user\'s word "好可以刪掉" (the leftover vectors of the round-10 families o64F-o71F; METHOD_NOTES 2.236 / 2.245)\n' % time.strftime('%F %T'))
        for p in todo: fh.write('%s\t%s\t%d\n' % ('link' if os.path.islink(p) else 'file', p, os.lstat(p).st_size))
    for p in lnk + reg: os.unlink(p)
    print('deleted', len(todo), 'record', rec)
