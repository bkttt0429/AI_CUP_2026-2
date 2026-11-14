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
2026-11-15 00:58 | 778292 | fix round 1: selftest OK (new card/lesson scope assertion); check.py OK, cards with derivation buttons 28, lessons 28, charts 14; no checker line for c09_pidv2.html / a08b_pidv2.js, no card-limit message; baseline unchanged (11 lines) -> PID_BUILD FAILED
2026-11-15 01:59 | 778777 | task 9 (old chapters marked 原始測試集, stale 11-07 sentences replaced by the fallback sentence because facts A-Q10 does not say whether the upload happened): selftest OK; check.py OK, charts 14, cards with derivation buttons 35, lessons 35; checker lines for c08_pid.html = the same 11 baseline lines (line numbers +1 from the new KPI), no new line -> PID_BUILD FAILED (expected until OLD_BASELINE exists)
2026-11-15 02:01 | 778789 | task 10 final build (glossary 11 entries, speaker script chapters 13-19, OLD_BASELINE in check_numbers.py, a08b_pidv2.js o9D label, c09_pidv2.html card-pid2-ext bullet 3): selftest OK (incl. the OLD_BASELINE assertions); check.py: anims 18, charts 14, cards with derivation buttons 35, lessons 35 / OK; no checker line at all -> PID_BUILD OK

## OLD_BASELINE

check_numbers.py holds `OLD_BASELINE` = the numbers of the 11 baseline lines above (10 distinct numbers; 0.343 occurs twice on one line): 0.913, 0.924, 0.343, +0.0119, +0.089, +0.0260, +0.0178, 0.928, 2.4968, 0.865. They are skipped only when the file name is `c08_pid.html`; every other file, and every other number in c08_pid.html, still has to trace to the facts. The numbers added to c08_pid.html in Task 9 (0.9069275 and the counts) are in the facts and are not on the list.

## not found (numbers or sentences of the plan that the facts did not contain, and what was done)

State at the end of the update (earlier tasks' lists, checked against the files now):
- Fine-tune gain "+0.0117 [+0.0071, +0.0164]": missing from the facts in Task 3, later added as D-R01; card-pid2-arch now uses it with its instrument (官方擬題池).
- "0.5898 對 0.5819" (Task 5): 0.5819 is in no facts file; the clause is left out (card shows only what C-R18 gives).
- "沒有人工標註測試資料" and "排行榜回饋只用於評估" (Task 6): first not found, later supported by D-K01 / D-K02; the card says 「排行榜得知的答案不用於訓練」, the lesson states the two uses (offline evaluation; the 11-06 correction file for 3 Public rows).
- KataGo humanv0 source URL and licence (A-Q14, B-Q07): the source is D-E01 (release v1.15.0); the licence is not stated on the release page and is written 「發布頁未載明，待確認」.
- "數十個網路" (Task 3): the facts give 20, 28 and more, no single figure; the card writes 「多個網路的平均」. The diagram label in a08b_pidv2.js still says 數十個 (labels only; not changed).
- Whether the 11-07 upload of results/submission_f3oreft_assign/ happened (A-Q10): not in the facts; the two sentences in chapter 12 and sec-21 now read 「（11-06 時的計畫；之後測試集重發，見第 13 章）」.
- Rank of the 0.9035072 file: no rank in the facts; none written. Ranks elsewhere only in the forms the facts give.
- o9D label in a08b_pidv2.js (P2.fam): was 「多種設定的網路」, fact C-C09 says o9D = 各種遮蔽; changed to 「各種遮蔽」.
- check.py "WARN node not found" on the compute node (Task 1): the job script looks up the node binary from the VS Code server directory (job_pid_build.sh), since 778195.

## cells marked 「（素材未載）」

All in lesson sec-33 (資源表), none on a card: 官方 Tutorial 與資料集 repo 的授權; 第三方函式與演算法的來源網址 and 授權 (2 cells); 2023 AI CUP 公開專案的授權. 4 cells in 3 rows. (Resolved since Task 6: the M88/M90/M101 rows left the table, ct9 is described in a paragraph; the KataGo row has URL and version, licence 「待確認」.)

## section 9 (待確認) items of the facts files that touched a card

- A-Q10 (was the 11-07 upload made?): chapter 12 / sec-21 of the old chapters, fallback sentence (Task 9).
- A-Q14 and B-Q07 (KataGo humanv0 source and licence; B-Q07 is cited in the sources of sec-33, A-Q14 only by its content): card-pid2-ext and sec-33 resource table; licence cell 「發布頁未載明，待確認」.
- B-Q10 (fold 0 in the official networks; cited in sec-32 and sec-33): card-pid2-ext bullet 4 and sec-33: own training never reads fold 0; the borrowed M88/M90/M101 networks do (D-E02).
- C-Q01 (notes changed during the work, new section §2.116): sec-34 (chapter 19) states the 11-14 status and name the later section without giving results.
- C-Q10 (member codes py, px2, pb, px, m101x undefined): sec-33 (sources and text) says the mapping of m101x to M101 is by name only.
- C-Q13 (the +0.0583 total versus the sum of its five numbers, +0.0584): sec-31 (and the card-pid2-valid numbers) follow the notes (+0.0583) and say so.
- C-Q15 (is the external-account saturation really too short training?): sec-32 (chapter 17) table row says the notes do not answer.
- C-Q22 (tie rule between two notes): cited in the sources of sec-31 only.
- D-Q01 (opponent-side signals re-tested with stronger networks, 11-15): sec-32 (chapter 17) table row, not adopted.
The other items of section 9 (A-Q01 to A-Q09, A-Q11 to A-Q13, A-Q15 to A-Q20, B-Q01 to B-Q06, B-Q08, B-Q09, B-Q11, B-Q12, C-Q02 to C-Q09, C-Q11, C-Q12, C-Q14, C-Q16 to C-Q21, C-Q23) are not cited by any card or lesson.
2026-11-15 02:05 | 778801 | fix round 1 (speaker script fold 0 exception, rule both parts; glossary EMA and 官方網路): selftest OK; charts 14, cards with derivation buttons 35, lessons 35 / OK; no checker line -> PID_BUILD OK (778800 ran before the chapter-16 line was saved; superseded)
2026-11-15 02:36 | 778963 | final review fix wave (external-account count 182,167 in the disclosure, like-for-like x2 vs b, upload-rule dates, t-p tag on chapter 14, glossary split, family chart base, node check in the job script, card-limit self-tests): selftest OK; charts 14, cards with derivation buttons 35, lessons 35 / OK; no checker line, no card-limit message -> PID_BUILD OK

## final review fix wave (2026-11-15)

Applied the final review's Critical 1-3 and Important 1-6, 8-14 (Important 7 and the checker's matching rules deferred by the plan's owner), the node check of Important 15, and the Minor text items. Where a fact differs from the review's proposal: C-E02 gives 182,167 only for o4S and 「18.2 萬」 for o5, o6, o7, o8D, o9D, so sec-33 writes 18.2 萬 for those families and 182,167 for o4S (the final file contains o4S, C-M65). Not changed: the four 「（素材未載）」 cells of the sec-33 resource table (not in the review).
2026-11-15 02:50 | 779010 | residual edits after the final re-review: selftest OK; charts 14, cards with derivation buttons 35, lessons 35 / OK; no checker line -> PID_BUILD OK
