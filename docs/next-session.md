# What is left — start here in a new session

Companion to `docs/refactor-handoff.md` (what the code **is**, and why) and
`docs/ipi-refactor-plan.md` (what was **decided**, going in). This file is the only one that
is a **to-do list**. Everything below is either not done, not verified, or not decided.

Ordered by what blocks what. Items 1 and 2 gate everything else.

---

## 0. First, confirm the tree is green

```bash
python3 scripts/smoke_check.py                    # 24 checks, network-free, no model loads
python3 scripts/check_seed_fidelity.py            # 224 of 257 `original` templates vs upstream
python3 scripts/check_metrics_fidelity.py         # success checks pinned to a golden table
python3 scripts/check_defensivetoken_fidelity.py  # 4 chat templates byte-identical
python3 scripts/check_pisanitizer_fidelity.py     # port vs vendored upstream
python3 -m compileall -q ipi                      # torch-gated modules only parse-check
```

All six pass as of commit `7fa523e`. Run them before and after anything. If one fails before
you have touched anything, stop and find out why — do not "fix" it forward.

**Read order for a cold start:** this file → `docs/refactor-handoff.md` §0 and §7 →
`CLAUDE.md` "Known gotchas" → the module docstring of whatever you are about to touch.

---

## 1. Push. Nothing is on GitHub. *(blocking)*

```bash
git log --oneline origin/main..main    # 10 commits, unpushed
git push
```

`main` tracks `origin/main`, so a bare `git push` is correct. There are two remotes pointing at
the same GitHub repo — `origin` and `adaptive`; `adaptive/main` has never been fetched, so
`git push adaptive main` would also work but is not the tracked path.

> **Credential hygiene, before anyone copies this directory.** `origin`'s URL in `.git/config`
> embeds a GitHub personal access token in plaintext. It is **not** in any tracked file and not
> in git history — checked — so nothing leaked through the repo itself. But it is readable by
> anything with access to the working directory: a backup, a shared machine, an uploaded folder.
> Rotate the token and switch `origin` to SSH or a credential helper:
> `git remote set-url origin git@github.com:alirezaAalaie/IPI-Aaptive.git`.

Kaggle notebooks install with
`pip install git+https://github.com/alirezaAalaie/IPI-Aaptive.git`, so **no notebook can see any
of the last two sessions' work until this happens.** Every item below that needs a GPU needs
this first.

Before pushing, sanity-check the wheel actually contains the two JSON data files — `pip wheel`
has never been run in any sandbox this was built in (trap 17), and packaging correctness is so
far argued structurally rather than verified:

```bash
pip wheel --no-deps -w /tmp/wheeltest . && unzip -l /tmp/wheeltest/*.whl | grep -E 'seed_templates|dual_verifiable'
```

Both must be present. If they are not, `[tool.setuptools.package-data]` in `pyproject.toml` is
wrong and every seed-based attack breaks at import on Kaggle.

---

## 2. The GPU validation pass *(blocking for any white-box number)*

No torch and no GPU existed in the sandbox where the last two sessions' changes were written.
The following code has been **read and compile-checked, never executed**. It is small and
local, but it is the first thing to run on a real machine.

| Where | What changed | What to look for |
|---|---|---|
| `attacks/beast.py` `run_scenario` | now composes `prompt_prefix + instance.query` | the injection actually contains the goal; check one `ScenarioResult.injection` by eye |
| `attacks/beast.py` end of `run_beast` | `n_queries = 1`, `n_forward_passes = (steps_run + 1) * k1 * k2` | `steps_run` reflects a `budget_seconds` cut-off, not the full loop |
| `attacks/gcg.py` | `eval_target_str` threaded through; `n_forward` split from `n_queries` | success fires on the full target, not the prefix; counters are plausible |
| `mutation/gradient.py` | pure move out of `attacks/gcg.py` | it *imports* — the maths was AST-compared against the prior commit, so only the wiring is new |
| `selector/reference_loss.py` `_score_batch` | batched forward + masked-label CE | the block itself; `_build_batch` around it is already mutation-tested without torch |

**A cheap equivalence test for the selector**, which is the one place a silent numerical error
could hide: score the same small dataset with `batch_size=1` and with `batch_size=None` and
assert the per-instance `_loss` values agree to ~1e-4. They are the same computation by
construction (right padding + causal attention), so a mismatch means the masking is wrong.

RS and Beam-RS are the exception to all of this — their `eval_target_str` split *is* exercised
against a mock victim in `smoke_check.py`, so they need no special attention.

---

## 3. Re-baseline the numbers *(blocked by 1 and 2)*

`docs/refactor-handoff.md` §7 lists **nine** behaviour changes, grouped by how much they move.
The short version for a results table:

- **BEAST: every prior number is void.** It was not attacking the scenario it reported on.
  Re-run from scratch; do not carry anything forward.
- **ICA: ASR moves by construction** — the default demonstrations changed (§9). `variant="original"`
  still reproduces the paper row.
- **GPTFuzzer and TAP: ASR moves** — reward, selection policy and the on-topic filter all changed.
- **RS, Beam-RS, AutoDAN, GCG: query counts move, ASR does not.** They were searching against
  criteria that could never fire.
- **BEAST and GCG `avg_queries` drop by orders of magnitude** and become comparable with the
  rest of the table for the first time. Compute now lives in `extra["n_forward_passes"]` —
  report it separately or the white-box rows look free.

**Do not put a pre-refactor and a post-refactor number in the same table** without saying which
is which.

---

## 4. Open decisions — nobody has chosen yet

### 4a. Convert `gcg` / `beast` onto `AttackDataset` + `ReferenceLossSelector`?

The original plan said yes; the audit said it was ~100x slower because the selector ran one
unbatched forward per instance. **That objection is now weaker** — the selector batches (commit
`8ef4bd7`) — but it is not gone: GCG scores at the **token** level and shares a prompt prefix
across all ~512 candidates in a step, which a text-level selector re-encoding each candidate
cannot exploit.

**Measure before deciding.** Time one GCG step both ways on the same model. If the converted
path is within ~2x, take it for the composability; if it is 10x, do not.

### 4b. Promote AutoDAN's operators into `mutation/rule.py`?

The audit established that `autodan._crossover` and `_hga_word_replace` are **better**
implementations than `mutation.rule.SentenceCrossOver` and `ReplaceWordsWithSynonyms` — they
preserve paragraphs, protect model-name tokens, cap replacements, and use a real roulette wheel.
`SentenceCrossOver` additionally returns its parents unchanged whenever `num_points >= max_swaps`,
which silently disables crossover for most of the 41-template pool.

Moving AutoDAN's versions in would be a pure refactor with no algorithm change, and would fix a
latent trap for anyone who reaches for `SentenceCrossOver`. It is not urgent — nothing currently
uses the EasyJailbreak versions. **At minimum, put a warning in `SentenceCrossOver`'s docstring
about the `num_points >= max_swaps` no-op** even if the promotion never happens.

### 4c. `judge=` on `beast` / `autodan` / `gcg`

Confirmed dead in all three: `self.judge` is read zero times, and none of `run_beast`,
`run_autodan_ga`, `run_autodan_hga`, `run_gcg` takes a `judge` at all. Either give it a job (gate
early stopping, as RS/Beam-RS now do) or delete the parameter. Deleting it is a breaking change
for notebook kwargs; wiring it changes query counts. Both are defensible; neither is done.

---

## 5. Known defects, recorded and not fixed

1. **BEAST's `ngram > 1` branch is wrong, and dead.**
   ```python
   if step % ngram != 0:
       curr_tokens = [[score_tokens[ii] for ii in range(0, len(score_tokens), k1)]]
       continue
   ```
   `score_tokens` has `k1 * k2` entries, so striding by `k1` keeps `k2` of them — equal to the
   beam width **only because `k1 == k2 == 15` by default**. With `k1=20, k2=10` the beam silently
   halves; with `k1=10, k2=20` it doubles. It is unreachable at the default `ngram=1`
   (`step % 1 == 0` always), which is why it has never bitten. Fix it or delete the branch —
   leaving a wrong-but-unreachable path is how it gets reached later.

2. **BEAST has no early stopping.** Success is evaluated once, after the whole beam search. Not
   a bug — it is what the algorithm does — but it means `eval_mode` steers nothing in BEAST, and
   its `n_queries` is 1 by construction. Do not read either as broken.

3. **TAP's defaults are not paper-TAP.** `TAPAttacker` ships `on_topic_prune=False`, `width=5`,
   `branching_factor=2`, `depth=10`, `keep_last_n=3`. For a row **labelled "TAP"** set
   `on_topic_prune=True`, `width=10`, `branching_factor=4`. See `docs/easyjailbreak-audit.md`.
   This is a reporting hazard, not a code bug, and it has been open since before the refactor.

4. **PAIR's query counts are not comparable to published PAIR.** On a JSON parse failure it
   `continue`s without counting a query, where the reference falls back to attacking with the raw
   goal. Also no JSON prefill for local attacker models. Both in `CLAUDE.md`.

---

## 6. Notebooks: updated, never run

All five in `experiments/` were edited for the new API and **none has ever been executed** — zero
cells carry outputs or execution counts. Their imports are statically verified (every
`from ipi… import X` resolves, except the three torch-gated modules), and every code cell parses.
That is all.

`ipi_attack_benchmark.ipynb` got the largest edit: its dataset cell still offered the BIPIA /
Hijack / AgentDojo / manual loaders that were deleted back in Phase A, and its judge cell still
imported `ipi.judges`. Header, dataset cell and judge cell were rewritten. **Expect it to need
another pass on first real run** — it is 38 cells and only the changed ones were reasoned about.

The four defense notebooks needed only import renames.

---

## 7. Beyond the refactor — the actual research

The refactor was infrastructure. The project's goal is a **prevention-style attention-based
defense against IPI**, stress-tested by this harness. Left to do there:

- **Run the baselines.** `StruQ`, `SecAlign`, `DefensiveToken` and `PISanitizer` are all
  implemented, fidelity-checked against vendored upstream, and have notebooks — **none has a
  recorded result**. `results/` holds 5 JSON files, all from 2026-08-01 and all pre-refactor:
  static attacks (naive/escape/combined) against `MockVictim`, `TargetLLM` and the in-context
  `Spotlight+Instructional` composite. Every one of them predates the nine behaviour changes in
  §3, so treat them as stale, not as a baseline.
- **PISanitizer is the closest published peer** to the defense being built (attention-based,
  prevention rather than detection, and the only baseline that defends an *API* victim). It is
  the comparison that will be asked about first.
- **The defense itself is not in this repo.** This is the attack + baseline-defense harness.

---

## 8. If you only do three things

1. **Push** (§1) — everything else is blocked on it.
2. **Run the six checks plus the GPU validation pass** (§0, §2) — the white-box edits are the
   only unverified code in the tree.
3. **Re-run BEAST from scratch** (§3) — its old numbers are not merely stale, they measured the
   wrong thing.
