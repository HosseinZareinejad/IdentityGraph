# Results

Every number here is produced by `python scripts/evaluate.py`, which trains
nothing and only measures the models that ship in `data/`. The machine-readable
version is `data/metrics.json`; the real-data section additionally comes from
`python scripts/validate_real_data.py` (`data/real_validation/metrics.json`).

Reproduce end to end:

```bash
python data/generate_synthetic_world.py && python etl/run_all.py && python scripts/train_fusion.py && python scripts/train_mapper.py && python scripts/evaluate.py
```

---

## 1. The headline

| Question | Answer |
|---|---|
| Given a cluster of accounts, do we name the right real person? | **82.2%** top-1, 92.5% top-5 |
| …when the system is willing to answer at all | **91.4%** precision, on 85.2% of clusters |
| …on people who have a same-name/city/birth-year twin | **66.0%** top-1, 78.8% precision when answering |
| Do the linked accounts actually belong together? | **B-cubed F1 0.8894** (P 0.911, R 0.869) |
| Can the pipeline even see the right pair? | **97.0%** blocking recall |

Measured on 1,752 clusters belonging to held-out test entities that no model
saw during fitting or calibration. The pipeline is deterministic: two runs
produce byte-identical `data/metrics.json` (see §7, item 5).

**The twin row is the real answer to the feasibility question.** Phase 1 plants
200 pairs of distinct people sharing full name, city and birth year. Nothing but
job, education, phone or email can separate them, and those fields live almost
only on LinkedIn and Telegram — so a cluster reaching neither is *undecidable*,
not merely hard. That is a property of the available data, not of the method,
and no model will fix it.

---

## 2. Corpus

5,200 people, 14,250 accounts, 4 platforms, plus a 5,200-record registry
standing in for an official database.

| accounts per person | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| share of people | 10.2% | 30.6% | 34.3% | 25.0% |

Field coverage — the missing-data structure the whole system exists to work
around:

| platform | city | birth year | job | education | phone | email |
|---|---|---|---|---|---|---|
| twitter | 19% | 5% | 0% | 0% | 0% | 9% |
| instagram | 30% | 5% | 0% | 0% | 6% | 9% |
| telegram | 0% | 0% | 0% | 0% | 25% | 6% |
| linkedin | 100% | 24% | 100% | 100% | 0% | 19% |

---

## 3. Validation on real data

The synthetic world was written by the same person who wrote the matcher, so
its numbers alone prove only that the matcher beats its author's idea of noise.
The name machinery is therefore also tested against **7,812 real (Persian,
English) name pairs of Iranian people from Wikidata** — romanisations written by
many unrelated human editors.

| measurement | result |
|---|---|
| deterministic transliteration vs. human romanisation (mean Jaro-Winkler) | 0.785 |
| the same, comparing consonant skeletons | **0.965** (93.9% of pairs above 0.85) |
| rank the right English name out of all 7,812, from the Persian name alone | **90.0% top-1**, 94.6% top-10, MRR 0.916 |
| phonetic blocking key: distinct keys over 7,793 distinct real names | 7,792 (one collision) |
| homophone variants that survive the key (ذ→ز, ص→س, ط→ت, غ→ق, ض→ز, ح→ه) | **100%** |

Two production changes came directly out of this and are described in §7.

There is no comparable public corpus of labelled cross-platform Persian
accounts, so the linkage and mapping stages remain validated on synthetic data
only. That limit is stated rather than papered over.

---

## 4. Candidate generation

790,760 candidate pairs from 101.5M possible ones — a reduction to 7.79e-03 —
in under 10 s, at **97.0% recall**. Recall here is a hard ceiling: a true pair that
never enters the candidate set can never be scored.

| path | candidates | recall | recall lost if removed | candidates saved |
|---|---|---|---|---|
| cross_field | 188,928 | 81.0% | **−3.4%** | 78,600 |
| name_fuzzy | 213,540 | 80.4% | **−2.1%** | 125,000 |
| username | 213,540 | 59.5% | −1.1% | 113,000 |
| phonetic | 12,275 | 51.6% | −0.0% | 0 |
| username_skel | 213,540 | 50.5% | −0.1% | 77,500 |
| vector_bio | 106,770 | 6.1% | −0.2% | 103,500 |
| vector_posts | 106,770 | 4.1% | −0.2% | 103,900 |
| contact | 200 | 1.4% | −0.03% | 5 |

Per-path recall overstates each path's worth; the "lost if removed" column is
what a path is really responsible for. Five paths could be dropped for under
half a point of recall each. They are kept because candidate volume is not the
bottleneck (scoring 787k pairs takes 50 s) and each is the only path that fires
on a failure mode the others share.

---

## 5. Linking accounts to each other

Three configurations, same scores, same threshold (0.30):

| | pairwise F1 | B-cubed F1 | largest cluster |
|---|---|---|---|
| A) threshold + connected components | 0.096 | 0.675 | **455** |
| B) + constrained 1-1 matching | 0.638 | 0.847 | 35 |
| C) + cluster hygiene | **0.818** | **0.889** | 4 |

Configuration A is what naive transitive closure does: one false edge welds two
identities together and the damage chains, producing a 455-account cluster.
Pairwise F1 partly hides this, which is why B-cubed is the headline metric.

**Threshold protocol.** Swept on validation entities, reported on test entities.
The objective is flat from 0.30 to 0.75 (validation B-cubed F1 0.8882 → 0.8869,
against a standard error of 0.0046), so the argmax inside it is noise. The tie
is broken one stage down, on end-to-end mapping accuracy, where 0.30 wins
(83.4% vs 82.6% top-1, 68.5% vs 67.1% on twins). A stricter threshold buys
cluster purity by splitting clusters, and a split cluster loses exactly the
pooled evidence the mapper needs.

**A caveat on pairwise probabilities.** Entity-disjoint splitting discards every
pair that straddles the split — the majority of candidates, all of them
negatives, including many hard ones. On pairs inside one split the model's 0.9+
bucket is right 81% of the time; over every candidate the blocker actually
emits, 71%. Quote the cluster-level numbers in §6, not the pairwise ones.

---

## 6. Mapping a cluster to a real person

| group | n | top-1 | top-5 | answered | precision when answered |
|---|---|---|---|---|---|
| all | 1752 | 82.2% | 92.5% | 85.2% | 91.4% |
| non-twin | 1590 | 83.9% | 92.8% | 86.4% | 92.4% |
| twin | 162 | **66.0%** | 90.1% | 72.8% | 78.8% |

### Where the misses come from

Each failed cluster is charged to the earliest stage that could have prevented
it, so the counts partition the 311 misses rather than overlapping.

| stage | misses | of all clusters |
|---|---|---|
| blocking never surfaced the person | 114 | 6.5% |
| the cluster contained someone else | 101 | 5.8% |
| the cluster was missing the person's other accounts | 76 | 4.3% |
| the ranker had the person and chose wrong | **20** | **1.1%** |

The ranker itself is responsible for 7% of the errors. Everything else is
upstream. Improving the scorer would move the headline by at most a point;
improving blocking and clustering is where the remaining accuracy is.

### Degradation

How much does the system need to know about someone?

| identifying fields present | n | top-1 |
|---|---|---|
| 0 | 357 | **55.5%** |
| 1 | 225 | 87.6% |
| 2 | 60 | 95.0% |
| 3 | 456 | 86.2% |
| 4 | 471 | 91.3% |
| 5 | 162 | 91.4% |

| platforms in the cluster | n | top-1 |
|---|---|---|
| 1 | 449 | **60.1%** |
| 2 | 424 | 90.1% |
| 3 | 489 | 91.2% |
| 4 | 390 | 87.9% |

The jump from 0→1 field (55.5% → 87.6%) and from 1→2 platforms (60.1% → 90.1%)
is the whole story. A name alone, on one platform, is not enough; a name plus
any one corroborating field usually is. Note the non-monotonicity at 3 fields
and 4 platforms — a cluster with more accounts is also a cluster with more
chances to have absorbed the wrong one.

| field present | n | top-1 with | top-1 without |
|---|---|---|---|
| phone | 311 | 94.5% | 79.6% |
| email | 412 | 92.7% | 79.0% |
| job / education | 1104 | 89.0% | 70.7% |
| city | 1250 | 88.3% | 67.1% |
| birth year | 352 | 88.6% | 80.6% |

Association, not causation: LinkedIn supplies job, education and city together,
so those rows largely describe the same population — clusters that reached
LinkedIn at all.

### Does the displayed confidence mean anything?

| predicted | observed | n |
|---|---|---|
| 0.03 | 0.05 | 123 |
| 0.33 | 0.52 | 119 |
| 0.78 | 0.75 | 103 |
| 0.89 | 0.90 | 531 |
| 0.97 | 0.96 | 827 |

Brier 0.0872. The three bins holding 83% of the mass are accurate to within two
points. The sparse middle bins are *under*-confident — clusters shown at 33%
are right about half the time — which is the safe direction to be wrong in, but
it is a miscalibration and it is not hidden here.

### Contested attributions

14.1% of clusters have a runner-up within 10% of the leader. Top-1 accuracy
there is **44.9%**, versus 88.4% elsewhere. Twins are 23.5% of contested cases
against 9.2% of all clusters — the statistic detects exactly what it was built
to detect, and the UI shows those cases a warning rather than a name.

---

## 7. What Phase 6 changed, and why

Four defects were found by measuring against real data, and fixed:

1. **The phonetic blocking key was missing ض, ظ and ح.** All are homophones in
   Persian. Substituting a homophone into a real Wikidata name changed its key
   77% of the time — the path was missing most of the confusions it exists to
   absorb. Adding them takes variant survival to 100% while collisions across
   7,793 real names stay at exactly one, so the recall is free.

2. **Synthetic usernames were vowel-less.** `transliterate()` emits only the
   letters Persian writes, so handles came out as `mhmdrza.ahmdy` — which
   nobody writes — and every platform got the *same* string, making
   handle-to-handle matching far easier than reality. Handles are now built by
   a syllable-aware romanizer (validated at 0.846 mean Jaro-Winkler against
   human romanisations, up from 0.785) with per-site spelling drift.

3. **Matching compared spellings that cannot agree.** Two romanisations of one
   Persian name differ almost only in the two things the script never recorded:
   short vowels, and whether a consonant is doubled (nobody types the shadda,
   so "mohammadreza" and "mohamadreza" are equally ordinary spellings). The
   comparison now happens on consonant skeletons with repeated letters
   collapsed: on the real corpus that takes top-1 retrieval from 66.1% to 86.3%
   to **90.0%**. Blocking recall rose to 97.0% despite the harder usernames,
   `cross_field` became the strongest path, and the same fix repaired operator
   search, where typing "mohammadreza" had stopped returning the accounts
   spelled "mhmdrza".

4. **Confidence ignored ambiguity.** It was a function of the top candidate's
   score alone, so a twin pair — two people with identical, entirely convincing
   evidence — was reported at 93%. Confidence now also reads the share of
   candidate mass the leader holds (~0.5 in a two-way tie), combined by a
   logistic and calibrated by isotonic regression on disjoint halves of
   held-out predicted clusters. The system now abstains instead of guessing:
   holding the threshold fixed so the change is attributable, precision when
   answering rose from 88.3% to 91.5% overall and from 70.0% to 76.0% on twins,
   while top-1 accuracy was unchanged (calibration does not move an argmax).
   At the finally deployed settings those figures are 91.8% and 80.2%.

5. **The pipeline was not reproducible.** Two runs on identical inputs gave
   different cluster counts and end-to-end accuracy that moved by ~0.4 points.
   The cause was `networkx`: `Graph.subgraph(nodes)` returns a filtered view
   whose `FilterAtlas.__iter__` walks a plain Python `set`, so iterating a
   subgraph is ordered by string hashes and differs in every process no matter
   how carefully the caller sorts its input. That decided which edge cluster
   hygiene called "weakest", and therefore how clusters split. The graph
   traversals are now written out explicitly (`_induced_edges`, `_components`
   in `src/clustering.py`), and the majority-entity vote breaks ties on the
   entity id. Two consecutive runs of `scripts/evaluate.py` now produce
   identical output.

One thing was measured and deliberately **not** changed. A leave-one-modality-out
ablation suggested dropping stylometry (+0.009 AP) or graph (+0.010 AP). Fitting
all 32 modality subsets on the validation split showed the whole space spans
0.9212–0.9277 F1 against a standard error of 0.0057 — the differences are noise,
and selecting on them would fit the validation split. Everything is kept.

---

## 8. What the ablation says about the proposal

The proposal specified five AI models. Measured on the validation split:

| feature set | F1 | AP | recall on pairs with exact contact evidence |
|---|---|---|---|
| name only | 0.9212 | 0.9262 | 91.4% |
| all six modalities | 0.9274 | 0.9426 | **100%** |

**Name features alone reach 99.3% of the full model.** Text embeddings,
stylometry, temporal rhythm and graph structure together are worth +0.006 F1.
The one place they are not optional is metadata: exact phone and email are rare,
but when present they settle the case, and they take recall on those pairs from
91.4% to 100%.

Standalone discriminative power, test AP: name 0.876, text 0.098, metadata
0.091, temporal 0.087, stylometry 0.073, graph 0.058 (0.05 would be chance).

This is worth stating plainly to anyone budgeting from the proposal: the
image model was dropped by agreement, and of the four that remain, one carries
the result. The others are cheap enough to keep and occasionally decisive, but
they are not what makes this work.

---

## 9. Cost

| stage | time |
|---|---|
| generate the world | ~10 s |
| ETL + embed + load 14,250 accounts into Qdrant | 136 s |
| blocking (787k candidates) | 10 s |
| scoring 787k pairs | 50 s (~16k pairs/s) |
| clustering | ~12 s |
| mapping 5,831 clusters | ~4 s |
| full evaluation | 100 s |

Single machine, no GPU.
