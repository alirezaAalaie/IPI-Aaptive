# Migrating a recipe onto the `single_attack` seam

Working doc for the port of `ipi/attacks/` onto the component spine. Reference
conversions already landed: **`tap.py`**, **`pair.py`**, **`gptfuzzer.py`**. Read one of
those before starting — `tap.py` is the fullest (all four component families).

## Why

Every recipe carried a `*Result` dataclass, a `run_*` free function taking a
`target_fn`, and a `run_scenario` adapter mapping one to the other. Measured across the
thirteen modules that is **1091 lines** — 191 dataclass + 489 adapter + 296 `__init__`
kwarg copying + 115 `__repr__` — all plumbing, no algorithm. EasyJailbreak's recipes are
half the size because they have one carrier (`Instance`) and no result type at all.

## The spine (already in the tree, do not re-implement)

**`ipi/attacker.py`**

| Member | What it does |
|---|---|
| `single_attack(target, instance, verbose) -> AttackDataset` | the method you write |
| `run_scenario(...)` | concrete on `BaseAttacker`: `single_attack` → `best_of` → `build_result` → `update`. **Delete your override.** |
| `report(best, query, *, success=None, score=None, **extra)` | last line of `single_attack`; stamps bookkeeping and wraps in an `AttackDataset` |
| `keep_best(best, dataset)` | replaces the `best_score`/`best_injection`/`best_response` triple |
| `score_of(instance)` | stamped score, else `eval_results[-1]` through `Evaluator.as_score` |
| `normalise_scores(dataset)` | collapse `eval_results` to one 1-10 int (bool → 10/1) |
| `as_attacker_llm(model)` | model string → `APILLM` at the house attacker settings |
| `_ATTACK_NAME` | the label in the results table |

`BaseAttacker.__init__` raises `TypeError` if a subclass defines neither
`single_attack` nor `run_scenario`, which is the guard the old `@abstractmethod` gave.

**`ipi/harness.py` — `VictimQuery`**, the fifth component family:
`AttackDataset -> AttackDataset`, same shape as mutation / constraint / evaluator.

```python
query = VictimQuery(instance, target, budget=self.max_queries)   # budget optional
dataset = query(dataset)          # writes target_responses + attack_attrs["injection"]
query.n_queries                   # victim calls — the definition the results table uses
query.exhausted                   # budget spent
```

It owns the try/except (a raising victim yields `""`, logged), the query count and the
budget. When the budget runs out mid-batch it returns only the candidates it reached, so
the evaluator never sees an instance with no response on it. `render=` handles a
candidate whose field is still a `{query}` template (GPTFuzzer) — the rendered string is
written to `attack_attrs["injection"]` and is what gets reported as `attack_str`.

Pass `data_separator=` for the OPI one-shots, where the separator *is* the attack.

**`ipi/mutation/generation.py` — `IntrospectBranching`**: the conversational attacker
turn shared by TAP (`branching_factor > 1`) and PAIR (`branching_factor = 1`).

## The shape

```python
class FooAttacker(AdaptiveAttacker):
    _ATTACK_NAME = "foo"

    def __init__(self, ..., judge=None):
        super().__init__(judge)
        self.mutator   = ...      # build every scenario-independent component here
        self.selector  = ...
        self.evaluator = judge

    def single_attack(self, target, instance, verbose=False) -> AttackDataset:
        goal = instance.query or ""
        target_str, eval_mode = resolve_attack_target(instance, self.eval_mode)
        query = VictimQuery(instance, target)
        dataset = self._seed(goal, ...)
        best = None
        for step in range(self.n_steps):
            dataset = self.mutator(dataset)
            dataset = query(dataset)
            self.evaluator(dataset)
            self.normalise_scores(dataset)          # only if a graded selector reads it
            best = self.keep_best(best, dataset)
            dataset = self.selector.select(dataset)
            if best is not None and <stop condition>:
                break
        return self.report(best, query, steps=step + 1)
```

## Rules

1. **Delete** the `*Result` dataclass and the `run_*` free function; fold the algorithm
   into `single_attack`. **Exception:** `gcg.py`, `beast.py`, `autodan.py` keep their
   bare `run_*` functions and result types — they are a documented standalone
   (non-scenario) entry point, and `docs/next-session.md` §4a defers the structural
   swap that would remove them. There, only the `run_scenario` adapter goes.
2. **Delete** the `run_scenario` override. Anything it put in `extra={}` becomes
   `**extra` on `report()`.
3. **Never hand-roll** the victim call, the query count, or best-tracking. `VictimQuery`
   and `keep_best` own those.
4. `eval_mode` stays owned by the data: default `None`, resolved with
   `metrics.resolve_attack_target(instance, self.eval_mode)`. Never hard-code a default.
5. Keep every **"IPI adaptations vs original"** docstring section current — it is the
   audit trail reviewers ask about. Note behaviour changes explicitly.
6. Seed strings stay in `ipi/seed/`. Do not inline a prompt.
7. **Do not touch** `scripts/smoke_check.py`, `ipi/__init__.py`,
   `ipi/attacks/__init__.py`, `CLAUDE.md` or the other docs — those are owned centrally
   and updated in one pass. Report what you changed instead.
8. `code/` is vendored reference material. Read it, never edit it.

## Verifying without a GPU

`python3 -m compileall -q ipi` plus an ad-hoc script with a fake victim:

```python
class V:
    backend = "api"; system_prompt = "s"; model_name = "m"; max_bs = 1
    system_prompt_template = ""
    def generate(self, messages, max_tokens=None, temperature=None):
        return "HACKED"
```

Run it under `PYTHONPATH=.` from the repo root. Check `n_queries` is what you expect —
that is the number most easily broken by this migration.

---

## State

| Module | Converted | Notes |
|---|---|---|
| `tap.py` | yes | 587 → 371. All four component families visible in the loop. |
| `pair.py` | yes | 383 → 277. Fixes the transcript-truncation defect (see below). |
| `gptfuzzer.py` | yes | 320 → 253. `judge=` is now one call per scenario. |
| `static_injection.py` | yes | 587 → 538. Five attackers; per-attack `data_separator` preserved. |
| `deepinception.py` | yes | 223 → 191. |
| `ica.py` | yes | 244 → 204. |
| `multilingual.py` | yes | 367 → 337. |
| `renellm.py` | yes | 281 → 235. |
| `beast.py` | yes | conservative: `run_beast` + `BEASTResult` kept. |
| `gcg.py` | yes | conservative: `run_gcg` + `GCGResult` kept. |
| `autodan.py` | yes | conservative: `run_autodan_*` + `AutoDANResult` kept. |
| `adaptive.py` | yes | conservative: `run_adaptive_rs` / `run_adaptive_beam` + `AdaptiveResult` kept. |

**All twelve are on the seam** — no recipe overrides `run_scenario` any more.

The four "conservative" modules (`adaptive`, `beast`, `gcg`, `autodan`) keep their bare
`run_*` functions and result types for the same reason: their algorithms are gradient,
logprob or beam searches with no per-candidate `victim.generate` for `harness.VictimQuery`
to own, so they do not decompose into the mutation / selector / evaluator families. Only
the `run_scenario` adapter went. `docs/next-session.md` 4a defers the structural swap that
would remove them, and it needs a GPU measurement first.

`ipi/runner.py` is deleted, along with `run_attack`, `run_experiment` and
`ExperimentResult`. Nothing used them: the notebooks were checked and use only `*Attacker`
classes plus `AttackEvaluator`.

## Behaviour changes to re-baseline

Three, all in recipes that were converted. None affects the *undefended* static rows.

1. **PAIR moves.** Its old inline truncation was `conv[:1] + conv[-12:]` — the system prompt
   plus the tail. In the IPI framing the goal, the user task and the tool schema live in the
   **opening user message**, so past thirteen messages (roughly six iterations) the attacker
   silently lost the objective it was refining toward. `IntrospectBranching.truncate` keeps
   both head messages. This is a fix, and it changes PAIR's numbers.
2. **GPTFuzzer's reported `score` may move.** The judge now grades only the reported
   candidate rather than every candidate. The *search* reward was already binary success and
   is unchanged, so ASR should not move — but confirm rather than assume.
3. **Row labels are pinned, not derived.** `BaseAttacker.name` would have turned
   `BeamRSAttacker` into `"beamrs"` and `AutoDANAttacker` into `"autodan"`; the results
   table has always said `"beam"` and `"autodan_ga"` / `"autodan_hga"`. Both are pinned
   with `_ATTACK_NAME` so a class rename cannot silently move a row label.
4. **`trace` is gone** from TAP, PAIR, GPTFuzzer and ReNeLLM. It never reached the results
   JSON (it stopped at the `ScenarioResult` boundary), so nothing published depended on it.
   The lineage lives on the `Instance`s instead, and `extra["max_level"]` reports how far
   each search actually descended.

## Verified

`scripts/smoke_check.py` (42 checks), `check_seed_fidelity`, `check_metrics_fidelity`,
`check_defensivetoken_fidelity`, `check_pisanitizer_fidelity`, `compileall` — all pass.

Separately, all eleven converted recipes were run end-to-end through `AttackEvaluator`
against a compliant and a refusing fake victim over real dataset instances, asserting that
(a) `n_queries` equals what the algorithm should spend, (b) it equals what the victim
actually received, and (c) the reported `injection` is a string the victim was really sent.
Query counts observed: static one-shots 1; ReNeLLM 1 → `evo_max`; GPTFuzzer stops exactly on
budget; TAP `width x branching` per level; PAIR `n_streams` per iteration.

RS and Beam-RS were run separately against a fake logprob victim: row labels come out as
`rs` / `beam`, the injection is reported, and `n_queries` still counts the logprob probes
the way it did before (the runners were not touched — the diff is the two adapters and two
helpers).

`beast` and `gcg` were **not executed** — they need torch, which this environment does not
have. They are parse-checked and read only. `autodan` imports torch-free by design and its
adapter was checked that far, no further.

## Notebooks

`experiments/ipi_attack_benchmark.ipynb` gained a **defense x attack sweep** (a markdown
header plus two code cells at the end): a matrix over `DEFENSES` x `ATTACKS` on `subset`,
reporting ASR and utility per cell, saving each `EvalResult` to `results/`, and drawing a
heatmap with a worst-case-ASR / mean-utility panel beside it. Search attacks and the
model-loading defenses are behind `INCLUDE_SEARCH_ATTACKS` / `INCLUDE_STRUCTURED_DEFENSES`
flags, both off by default. A defense that cannot be constructed is skipped with a printed
reason rather than taking the sweep down.

The sweep cell was executed here against a fake target (6 defenses x 7 attacks x 8
scenarios, 42 cells, none skipped). The table/figure cell was **not** executed — no pandas
in this environment — so it is reviewed, not run. Two bugs were found by that review and
fixed: the defense was being constructed once per attack (a model reload per column for
PISanitizer / DefensiveToken / StruQ / SecAlign), and an extra `invert_yaxis()` flipped the
heatmap against the bar panel so the bars lined up with the wrong defenses.

The other four notebooks were audited statically — every `from ipi... import` resolves —
and needed no changes. None of the five has ever been executed.
