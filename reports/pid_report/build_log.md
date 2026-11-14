# build log

time | job id | result
2026-11-15 00:44 | 778187 | selftest OK; check.py OK (0 errors, node missing: anims 0, charts 0); checker red (baseline below) -> PID_BUILD FAILED

## baseline

job 778187, old chapter c08_pid.html, numbers not in facts:

```
c08_pid.html:42: 0.913 not in facts: class="n">0.913</td><td class=
c08_pid.html:42: 0.924 not in facts: class="n">0.924</td></tr>
c08_pid.html:75: 0.343 not in facts: <td>0.328／0.343 ＝ 0.96</td><td
c08_pid.html:75: 0.343 not in facts: <td>0.328／0.343 ＝ 0.96</td><td
c08_pid.html:148: +0.0119 not in facts: +0.0046, +0.0119]</td><td>通過</
c08_pid.html:195: +0.089 not in facts: 0311，n＝5 +0.089；同一批候選人但題目會重複同一
c08_pid.html:201: +0.0260 not in facts: lass="n">+0.0260</td></tr>
c08_pid.html:204: +0.0178 not in facts: lass="n">+0.0178</td></tr>
c08_pid.html:222: 0.928 not in facts: 候選只剩 50 局 0.928；按檔案順序切（時間代理）增益
c08_pid.html:274: 2.4968 not in facts: 7580 ⇒ 損失 2.4968 分 ＝ 0.632 ＋ 0
c08_pid.html:274: 0.865 not in facts: ＝ 0.632 ＋ 0.865 ＋ 1</div>
```

2026-11-15 00:45 | 778195 | node found; check.py: anims 18, charts 10, cards with derivation buttons 25, lessons 25 / OK; baseline unchanged (11 lines) -> PID_BUILD FAILED
2026-11-15 00:49 | 778221 | selftest OK (incl. js '<' case); check.py: anims 18, charts 14, cards with derivation buttons 25, lessons 25 / OK; a08b_pidv2.js: no checker line; baseline unchanged (11 lines) -> PID_BUILD FAILED
2026-11-15 00:55 | 778274 | selftest OK; check.py: charts 14, cards with derivation buttons 28, lessons 28 / OK; c09_pidv2.html and a08b_pidv2.js: no checker line, no card-limit message (bullet limit now 64); baseline unchanged (11 lines) -> PID_BUILD FAILED (chapters 13 and 14 together)
2026-11-15 00:56 | 778277 | final state of chapters 13 and 14: selftest OK; check.py OK, cards with derivation buttons 28, lessons 28, charts 14; no checker line for c09_pidv2.html / a08b_pidv2.js; baseline unchanged (11 lines) -> PID_BUILD FAILED
