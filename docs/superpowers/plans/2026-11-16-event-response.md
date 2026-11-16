# 事件反應與瓶頸診斷 Implementation Plan

> **For agentic workers:** use superpowers:executing-plans, native single implementer, independent scientific/code review. User has delegated method decisions and explicitly asked to continue; do not request routine approval again.

**Goal:** Determine whether opponent-surprise-aligned changes in subsequent choices contain repeatable player identity beyond marginal style, then build/evaluate a bounded augmentation only if this prerequisite is supported.

**Architecture:** Reuse frozen humanv0/proj3 and the verified13,000-game manifest. Extract both sides' pre-move policy surprisal, value, entropy and33-dimensional played-minus-expected action features. Fit nuisance regressions on700calibration reference games only, construct event-conditioned response kernels at1/2/4/8subject moves, and test reference-to-held-game repeatability against matched within-game event permutations.

**Tech Stack:** existing NumPy/PyTorch/SciPy, R1venv, productionSlurm; existing isolated worktree.

**Spec:** mathematical/design section below is the short design. Choices are agent decisions under user delegation, not user-prescribed numbers.

## Global Constraints

- Preserve all frozen choice/exact code and reports. New code/modules and events/ subtree only; reuse initial manifest andsourcehash verification.
- Continue SAME total budget:2simultaneousGPUs,6GPUh including previous1.18GPUh,1GiB artifact total including existing753.5MB; /workuserquota>=6GiB,/home>=3GiB. Samejobs.txt records all new jobs. No large training or officialsubmission.
- Computation/tests on productionSlurm. No fold0. No officialquestionCSV/answers. No gain-based grid sweep after outcome.
- Existing Transformer has temporal position embeddings and both-side input; do NOT claim missing order. Novel probe is explicit opponent policy surprise × subsequent choice change, not local displacement (alreadyused).

## Design and alternatives

The preceding low-dimensional value-conditioned model failed the+.002practical gate despite exact scoring. This limits that model; it does not identify an information-theoretic ceiling. Alternatives: deeper same-feature fitting (not prioritized without stable new signal), more global covariance (previous probes alreadyweak), explicit event-aligned response (chosen because information differs from marginal choice/value and can be falsified cheaply).

All moves are before-move inputs, first256plies, withcheckedturnorder/legal mask. At eachply retain residualr=psi_played−Epolicypsi, surprisals=−logpolicyplayed, valuev side-to-move, entropyh, color,t. Game records retain subjectcolor. For eventcontext v is opponent-to-move perspective; keep that explicit, do not call it target winrate. Responsecontext uses subjectperspective. No future move enters event definition.

Nuisance basisX=[1,isWhite,phase60to149,phase150plus,v,v²,h,h²]. On ONLY700calibration reference games, standardize nonintercept columns, fit ridge1e-3 average squared loss to predict s and eachr dimension; scale target residuals by reference residualRMS withfloor1e-6. No heldgame in nuisance fitting. Clip standardized s residual to[-3,3]. Then construct opponent-only eligible events BEFORE centering. Allfourlags use common support witha priorownmove and8observedfutureownmoves. Center shocks within eachgame×matchedcell onthiscommoneligibleeventset; singletoncellsbecomezero. Thuscensoredtaileventsandothersidepliescannotchange retainedshockmarginals.

For each eligibleopponent event, Δr_L=r_adjusted(response atL)−r_adjusted(subjectmove beforeevent),L=[1,2,4,8], nozero-padding. Allfourlags share thelag8eligibleanchors describedabove. E's pergame sufficient statistics: numeratorΣs_event*Δr_L, denominatorΣs_event², countN. Candidate kernel is numerator/(denominator+N), pooled acrossreferencegames beforedivision. D usesΣΔr_L/N, M usesmean adjusted subjectr. These are descriptive conditional associations, NOT psychological/causal responses. Rawsurpriseisbefore-move; centeringusescompletedgameevents, so the finalweight/kernel isretrospective, notanonlineeventdetector.

PermutationP: permute centeredshock within samegame×phase×eventvaluebin([-1,-1/3,1/3,1])×entropybin(reference median); fixed keyhashsalt andONE sharedpermutation acrossallfourlags oncommoneligibleanchors. Sameformula/counts/features andexactdenominators asE. Count fraction in cells with>=2entries and fraction actually moved. No shuffle seed selection. ConstantΔr (lineardrift), notjustconstant r, mustgivezeroE; a synthetic identity withidenticalmarginalmean butoppositeeventresponse mustbeseparable.

Metric: M/D/E/P each normalized in a calibration-reference-only diagonal metric. E/P share the RMS of both reference feature families. For eachcandidate pool reference7, queryhold3, all100calibrationplayers are scored by cosine of pooledkernels. Report1/2/3game equally weighted rankscore, Top1, true-minus-meanwrong cosine, and paired E−P ownerbootstrap5000CI. This is pilot evidence among training-projection players, NOT unseen-player effectiveness or baseline augmentation.

Predeclared diagnostic learningcurves: use first1/3/7ofthesamehash-orderedreference7, andcomparequerysizes1/2/3, withmetricheldfixedfromreference7. ReportpairedownerCI forreference7−1 andquery3−1. These characterize sample-sensitivity ofthisrepresentation/scorer; they are notinformation-theoreticceilings, do notselectnewhyperparameters, andcannotreplaceE−Ppilotgate. They do notconstitute strictend-to-endone-referenceperformance becauseglobalmetricusesreference7.

Pilot prerequisite: >=95%calibrationgames have>=20events available atlag8; >=60%events are in permutablecells; E−P paired rankscore95%CI lower>0 AND E's true-minus-wrong cosine95%CI lower>0. Passing onlysupportsrepeatablealignmentrelative toP,conditional onfixedreferencebank/nuisance/projection; itdoesNOTestablishincremental information beyondM/DorTransformer. Failuremeansthisdefinition lacksadequateevidence, notalltemporalidentityabsent. No switch toD/M todeclarepass. Ifpass, separatelyfreeze officialaugmentation using all600candidates, primaryseed4/5(newmethodoutcomesnotread),samecrossfit andalpha[0,.05,.1,.2,.4]; onlythenimplementdownstream. Keepofficialevidenceclaimseparatefromcalibration.

## Review Focus

1. Black/white and firstmove/handicap/pass: eventmustbeopponent, predecessor/responsemustbesubject, nofutureeventdefinition.
2. Near-end events: explicitcensor perlag, neverzero-pad missingresponses.
3. Nuisance/cosinemetric fit only reference; changing helddata cannot change fittedobjects.
4. Permutation retains matchingcellmultisets; noactualswaps can occur acrosscells/games.
5. Correlated events nottreatedasindependentsamples; bootstrapowner. Storageinterruptionfailsclosed andsourcehashmustmatch.

## Task1: Mathematical signal and falsification tests

Create src/aicup_pid/event_response.py andtests/test_event_response.py. Interfaces nuisance_fit(records), nuisance_apply(record,fit), event_moments(record,fit,key), pool_kernel(rows,arm), compare_repeatability(records,features). RED tests: delayedresponse/censor,colors,constantresidualzero,phase-cellpermutation,heldoutisolation,syntheticequalmarginaloppositeresponse. GREEN onSlurm thencommit.

## Task2: Real calibration extraction and study

Create scripts/pid_event_probe.py. Read legacyverifiedmanifest; extract both colors withcheckedreplay, mergebyoriginalply; frozenforward, no newnetworktraining. Store1000calibrationgames in64gamecompressedshards withchecksum, code/sourceprotocol. Pilotfirst64thenfinish1000ifcostwithinremainingbudget. Rawcalibration<50MB, summaries<10MB. Fitnuisance700ref, buildallkernels, evaluate100players only, savefixedgate andcoverage/CI inevents/calibration_report.json. Testsourcechange/orphan/coverage guards. Independentcode/resultsreview. Ifstablegatepasses, nexttaskofficialevaluation is planned fromfrozenfeatures; otherwise recordwhere information testfailed anddo notpretend robustsignalorabsenceofallrawinformation.

## Task3: Deliver bottleneck decision

Retain exact prior failure, distinguish architecturalcapacity from learned/usefulinformation, report data/noise/representation limitations andnewmodelstatus. UpdateMETHOD_NOTES andsmallresultsJSON; finaltoolverification/commit/cleanstatus. Do notstopatplanonly; execute calibration study inthisturn.
