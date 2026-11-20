# Deletion of class C (o40-o43 checkpoints + vectors), o59 checkpoints, o53 k50 / o56 k45 snapshot checkpoints.
# The user's word 2026-11-19 12:2x: "同意刪 C 類、o59 和快照". Dry run unless the argument is "go".
import os, re, sys, collections, time
go = len(sys.argv) > 1 and sys.argv[1] == 'go'
R2 = '/home/u2507115/BK_AICUP_2026-2'
SN = '/work/u2507115/pid_v2/seqnet'
ck = re.compile(r'ckpt_-1_(o4[0-3][A-Za-z]+s\d+e|o59[A-Za-z]+s\d+e|o53[A-Za-z]+s\d+k50e|o56[A-Za-z]+s\d+k45e|o53KFs\d+e|o56KFs\d+e)\.pt$')
vec = re.compile(r'(official|offsim|oldsim)_.*_net-1_o4[0-3]Fs\d+e_t\d+\.npz$')
stores = [R2 + '/data/seq_ft', R2 + '/data/pid_v2_seq_ft_offsim', R2 + '/data/pid_v2_seq_ft_oldsim', R2 + '/data/pid_v2_f0tmp/ft', R2 + '/data/pid_v2_f0tmp/old',
          '/work/u2507115/pid_v2/vec_overflow/offsim', '/work/u2507115/pid_v2/vec_overflow/oldsim', '/work/u2507115/pid_v2/vtmp8/ft', '/work/u2507115/pid_v2/vtmp8/old']
todo = []  # (path, kind)
for f in sorted(os.listdir(SN)):
    if ck.match(f): todo.append(os.path.join(SN, f))
n_ck = len(todo)
other_c = collections.Counter()
for d in stores:
    assert os.path.isdir(d) and not os.path.islink(d), d
    for f in sorted(os.listdir(d)):
        if '_net-1_o4' in f and re.search(r'_net-1_o4[0-3]F', f):
            if vec.match(f): todo.append(os.path.join(d, f))
            else: other_c[d] += 1
print('checkpoint names', n_ck, 'vector names', len(todo) - n_ck, 'class-C-looking names NOT matched by the strict pattern', dict(other_c))
S = set(todo)
reg = [p for p in todo if not os.path.islink(p)]
lnk = [p for p in todo if os.path.islink(p)]
# every link must resolve into the set (or dangle); every regular file must have no hard link outside the set
ino = collections.defaultdict(list)
for p in reg:
    st = os.lstat(p); assert os.path.isfile(p)
    ino[(st.st_dev, st.st_ino)].append(p)
bad_h = [(k, v) for k, v in ino.items() if os.lstat(v[0]).st_nlink != len(v)]
bad_l = []
for p in lnk:
    t = os.path.realpath(p)
    if os.path.exists(t) and t not in S: bad_l.append((p, t))
print('regular', len(reg), 'links', len(lnk), 'hard links leaving the set', len(bad_h), 'links leaving the set', len(bad_l))
for x in (bad_h[:3] + bad_l[:3]): print('  ', x)
size = collections.Counter()
for (dev, i), v in ino.items():
    size[os.path.dirname(v[0])] += os.lstat(v[0]).st_size
for d, b in size.items(): print('  %-62s %6d MB' % (d, b / 2**20))
print('total MB', round(sum(size.values()) / 2**20))
# files of vec_overflow that are not in the set
rest = [os.path.join(dp, f) for dp, dn, fn in os.walk('/work/u2507115/pid_v2/vec_overflow') for f in fn if os.path.join(dp, f) not in S]
print('vec_overflow files outside the set', len(rest), rest[:3])
# inbound links from anywhere in my vector stores / seqnet that point into the set but are not in it
inb = []
for d in stores + [SN, R2 + '/data/pid_v2_f0tmp', '/work/u2507115/pid_v2/vtmp8']:
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.islink(p) and p not in S and os.path.realpath(p) in S: inb.append(p)
print('inbound links from outside the set', len(inb), inb[:3])
ok = not bad_h and not bad_l and not inb and not other_c
print('checks', 'PASSED' if ok else 'FAILED')
if go:
    assert ok
    rec = R2 + '/results/final/deleted_C_o59_snap_20261119.txt'
    assert not os.path.exists(rec)
    with open(rec, 'w') as fh:
        fh.write('# deleted %s on the user\'s word "同意刪 C 類、o59 和快照" (class C = o40-o43 checkpoints + vectors; o59 checkpoints; o53 k50 / o56 k45 snapshot checkpoints)\n' % time.strftime('%F %T'))
        for p in todo:
            fh.write('%s\t%s\t%d\n' % ('link' if os.path.islink(p) else 'file', p, os.lstat(p).st_size))
    n = 0
    for p in lnk + reg:
        os.unlink(p); n += 1
    print('deleted', n, 'record', rec)
