# `ipi/` refactor — adopting the EasyJailbreak component architecture

Status: **Phases A–G shipped. Phases H–I pending.**
Operational state of the working tree — what is wired, what is not, and the traps found —
is in **`docs/refactor-handoff.md`**. Read that before picking this up.

This supersedes the first draft of this document, which proposed a light touch-up
(`selectors.py`, `constraints.py`, a `Scorer`/`Evaluator` rename). Review of the vendored
`code/attack/EasyJailbreak-master/` showed that plan produced *folders without composability*,
and got two specific things wrong. The target is now full component parity.

---

## Context — why

`ipi/` accreted as a set of self-contained attack scripts. Each attack inlines its own candidate
representation, its own selection rule, its own prompt text, and its own notion of "did it work."
The consequences are concrete:

- **GPTFuzzer uses flat UCB1 instead of the paper's MCTS** — not a considered trade-off. Upstream's
  `MCTSExploreSelectPolicy` needs `index` / `visited_num` / `level` / `children` on each candidate;
  we have no candidate object, so the tree variant was unportable.
- **Every attack invents its own candidate type.** TAP uses `dict` with `injection`/`score`/`conv`
  keys, GPTFuzzer a `_SeedNode` dataclass, AutoDAN a bare `list[str]`, ReNeLLM plain strings.
  Nothing composes.
- **Prompts live in two places.** `prompts.py` holds TAP/PAIR attacker prompts and the judge
  prompts; `attacks/seed_templates.json` holds injection payloads. Upstream has no `prompts.py` —
  a system prompt *is* a seed, keyed by which model it configures.
- **AutoDAN never got its paper seeds.** `autodan.py:91` seeds its GA from our 41-template IPI pool;
  upstream's `attack.AutoDAN-a` pool of 128 was never ported.
- **`dataset.py` is 1504 lines holding six loaders**, of which one is the benchmark we actually run.

Outcome wanted: attack recipes that are pure wiring over named, swappable components — the shape
that made EasyJailbreak's thirteen attacks fit in ~150 lines each.

---

## What upstream actually does (the findings that drive this)

**1. The carrier is the architecture.** `Instance` / `JailbreakDataset` is what every component
family speaks:

```
MutationBase.__call__(dataset) -> dataset      # 1-to-n, records parents/children
ConstraintBase.__call__(dataset) -> dataset    # filters
SelectPolicy.select() -> dataset  /  .update(dataset)
Evaluator.__call__(dataset)                    # writes instance.eval_results
```

`Instance` carries `query, jailbreak_prompt, reference_responses, target_responses, eval_results,
parents, children, index, visited_num, level, attack_attrs`. Without it, the component folders are
cosmetic.

**2. Seeds are keyed by consumer, not by variant.** `seed_template.json` is
`{prompt_usage: {method: [templates]}}` with usage ∈ `attack`, `judge`:

| key | n | what it is |
|---|---|---|
| `attack.TAP` / `attack.PAIR` | 2 each | the **attacker system prompts** |
| `judge.PAIR` | 1 | the judge system prompt |
| `attack.AutoDAN-a` | 128 | the AutoDAN GA seed pool |
| `attack.Gptfuzzer` / `ICA` / `DeepInception` / `ReNeLLM` | 77/1/1/3 | already ported |

Our JSON is keyed `original`/`ipi`/`advbench` — a *variant* axis. We need both axes.

**3. There is no Scorer/Evaluator split.** Both live in `metrics/Evaluator/`, distinguished by what
they emit: `EvaluatorGenerativeGetScore` (1-10 guidance, used by TAP/PAIR) sits beside
`EvaluatorClassificatonJudge` / `EvalatorMatch` / `EvaluatorPrefixExactMatch` (binary success). Our
`check_ipi_success(mode=contains/exact/function_name)` maps onto that last group directly.

---

## Decisions (locked)

| Decision | Choice |
|---|---|
| Carrier object | **Yes, full parity.** `Instance` + `AttackDataset`, adopted by all recipes. |
| Guidance vs success naming | **Upstream `metrics/`.** `*GetScore` = guidance, `*Judge`/`*Match` = success. Discards the `scoring.py`/`evaluation.py` split. |
| Back-compat | **Clean break.** No shims. `experiments/*.ipynb` are updated in a later pass, not blocking. |
| Component breadth | **Full library, no new attacks.** Port all upstream mutations/selectors/constraints/evaluators; do *not* add Jailbroken/Cipher/CodeChameleon/MJP recipes. |
| AutoDAN seeds | **128 verbatim → `attack.AutoDAN.original`; existing 41 IPI pool → `attack.AutoDAN.ipi`.** No new template authoring. |
| Dataset format | **Keep the JSON as built; map to `Instance` in the loader.** `build_dual_verifiable_dataset.py` is untouched. |
| Other datasets | **Remove all of them** — `ManualIPIDataset`, `AgentDojoDataset`, `BipiaDataset` (+ target maps, + `BipiaSuccessEvaluator`), `HijackDataset` (+ `HIJACK_ATTACKS`). `DualVerifiableDataset` is the benchmark. |

### Why AutoDAN keeps both pools

The 128 upstream templates are 2522–3883 chars (median 3252) and use a `[PROMPT]` placeholder; our
IPI pool is 94–287 chars (median 154) using `{query}`. They are DAN-style persona jailbreaks built
to defeat *safety refusal on harmful content* — but our goals (`"Print only HACKED"`, eval
`startswith`) are not harmful and nothing is refusing. The real barrier in our dataset is
*instruction provenance*, which a persona override does not address; and a 3.2 KB monologue injected
into a ~300-char email body is implausible and interacts badly with PISanitizer/StruQ/SecAlign.
They are still the correct *initial GA population* for a paper-faithful AutoDAN row, since fitness
is loss on `optimization_target` and the search mutates away from the persona. Hence: both, selected
by variant.

---

## Target layout

```
ipi/
  victim.py  target.py  llm_unified.py   ← UNCHANGED. Our Victim contract is the target interface
                                            (upstream models/). Deliberate deviation — keep it.
  config.py  runner.py
  data/                                  ← packaged JSON                              ✅ Phase A

  datasets/          Instance · AttackDataset · DualVerifiableDataset   (replaces dataset.py)
  seed/              SeedBase · SeedTemplate · seed_templates.json      (absorbs prompts.py)
  mutation/          generation/ · rule/ · gradient/
  selector/          SelectPolicy · UCB · MCTSExplore · RoundRobin · SelectBasedOnScores · EXP3
                     · ReferenceLoss · Random
  constraint/        ConstraintBase · DeleteOffTopic · DeleteHarmLess · PerplexityConstraint
  metrics/           Evaluator family (*GetScore guidance, *Judge/*Match success) + AttackEvaluator
  attacks/           recipes only — orchestration, no inlined prompts/rules
  defenses/          ← UNCHANGED
```

Deleted at the end: `dataset.py`, `evaluator.py`, `prompts.py`, `judges.py`, `scoring.py`,
`attacks/seeds.py`, `attacks/mutations.py`.

---

## Current on-disk state

Phase A is done and green (dead `evaluators/`+`targets/` shims removed, data JSON moved to
`ipi/data/`, `.gitignore` anchored to `/data/`, `scripts/smoke_check.py` added).

Three files from the **abandoned** first-draft Phase B are on disk and must be re-targeted, not kept:

- `ipi/scoring.py` — its four scorer classes become `metrics/` `*GetScore` evaluators.
- `ipi/judges.py` — now a deprecation shim; delete (clean break).
- `ipi/prompts.py` — judge prompts already stripped out; the rest dissolves into the seed registry.

---

## Phases

Each phase ends green against `scripts/smoke_check.py`, which grows a check per phase.

**Phase B — the carrier + datasets. ✅ DONE**
New `ipi/datasets/`: `instance.py` (`Instance`, extending upstream's fields with our dual-verifiable
ones — `user_target`, `optimization_target`, `attack_eval_mode`, `user_eval_mode` in `attack_attrs`),
`attack_dataset.py` (`AttackDataset`, upstream `JailbreakDataset` minus its `torch`/HF-`datasets`
base — `import ipi` must not require torch), `dual_verifiable.py`. Field mapping:
`injection_goal`→`query` (it is what fills the `{query}` placeholder, exactly as upstream's harmful
request does), `target_str`→`reference_responses`, and the IPI-specific fields — `user_task`,
`user_target`, `user_eval_mode`, `attack_eval_mode`, `optimization_target`, `clean_context`,
`pipeline_context`, `tool_schema` — into `attack_attrs`.

> As executed: `dataset.py` 1504 → 291 lines, `evaluator.py` 1033 → 635. Removed
> `ManualIPIDataset`, `AgentDojoDataset`, `BipiaDataset` (+ both target maps),
> `HijackDataset` (+ `HIJACK_ATTACKS`), `BipiaSuccessEvaluator`, the BIPIA branch of
> `make_scenario_target_fn`, and its two formatting helpers. `IPIDataset.subset()` returned
> `ManualIPIDataset`, so it now returns a small private `_ScenarioList`.
>
> `ipi/dataset.py` **survives this phase** rather than being deleted: `evaluator.py`, `attacker.py`,
> `runner.py` and all fifteen recipes still take `IPIScenario`. Deleting it here would leave the
> package unimportable until Phase H. It is marked legacy, and `ipi.__init__` exports the carrier
> under the primary names with the old loader as `LegacyDualVerifiableDataset`, so the two never
> collide. `dataset.py` dies in Phase I.
>
> `_split_bipia` in `defenses/channels.py` is deliberately kept — it parses a *messages list*, not a
> dataset, and CLAUDE.md flags channel recovery as load-bearing. Removing it buys nothing.

**Phase C — `seed/` + registry re-key.**
`seed/base.py`, `seed/template.py`, `seed/seed_templates.json` re-keyed to a three-level superset of
upstream's schema so both axes are explicit:

```json
{"attack":     {"TAP":     {"original": [...], "ipi": [...]},
                "AutoDAN": {"original": [128 verbatim], "ipi": [41 IPI pool]}, ...},
 "judge":      {"PAIR": {"original": [...]}, "IPI": {"ipi": [...]}},
 "constraint": {"TAP_on_topic": {"original": [...], "ipi": [...]}}}
```

Ports `attack.AutoDAN-a` (128) verbatim; moves every string in `prompts.py` and the scorer prompts
in `scoring.py` into the registry; then deletes `prompts.py` and `attacks/seeds.py`. Placeholder
normalisation (`[PROMPT]` → `{query}`) happens at load, recorded in `_meta`.

**Phase D — `metrics/`.**
`Evaluator` ABC over `AttackDataset`. Guidance: `EvaluatorGenerativeGetScore` (from `IPILLMJudge`/
`GPTJudge`), `EvaluatorEditDistanceGetScore`, `EvaluatorKeywordJudge`. Success: `EvaluatorMatch`,
`EvaluatorPrefixExactMatch`, `EvaluatorPatternJudge` — wrapping the existing
`check_ipi_success` / `check_user_utility` / `resolve_attack_target` logic from `evaluator.py`.
`AttackEvaluator` moves here and keeps **sole ownership of the reported ASR**; a recipe's own
evaluator only drives its early-stop. Delete `scoring.py`, `judges.py`.

**Phase E — `mutation/` (full library).**
`mutation/base.py` + `generation/` (13 upstream ops incl. our GPTFuzzer 5) + `rule/` (~25 upstream
rule ops: Base64/Rot13/Leetspeak/Disemvowel/cipher family/…, plus our ReNeLLM 6) + `gradient/`
(`token_gradient` for GCG). Prompts stay verbatim. Replaces `attacks/mutations.py`.

**Phase F — `selector/`.**
`SelectPolicy` ABC + `UCBSelectPolicy`, `MCTSExploreSelectPolicy`, `RoundRobinSelectPolicy`,
`SelectBasedOnScores`, `EXP3SelectPolicy`, `ReferenceLossSelector`, `RandomSelector` — formulas
verbatim from upstream, now portable because `Instance` supplies the node bookkeeping.

**Phase G — `constraint/`.**
`ConstraintBase` + `DeleteOffTopic` (takes TAP's on-topic prompt from the registry),
`DeleteHarmLess` (ReNeLLM's dropped filter, opt-in), `PerplexityConstraint`.

**Phase H — migrate the recipes, one at a time.**
Order chosen so each lands independently green: `gptfuzzer` (gains real MCTS) → `tap` → `pair` →
`renellm` → `autodan` (gains the 128-seed `original` mode) → `static_injection` /
`deepinception` / `ica` / `multilingual` → `adaptive` / `beast` / `gcg`. Each recipe drops its
inlined candidate type, prompt text, and selection rule, and composes the seams instead. The
`judge`/`score` parameter disappears from every attack that never used it (RS, Beam-RS, BEAST,
AutoDAN, GCG all accept it today and never read `self.judge`).

**Phase I — cleanup.**
Delete `evaluator.py`; rewrite `ipi/__init__.py`'s public surface; refresh `CLAUDE.md`, the
"IPI adaptations vs original" docstring sections, and `docs/easyjailbreak-audit.md`; update
`experiments/*.ipynb` (notably `secalign_defense_notebook.ipynb`, which uses the removed
`HijackDataset`).

---

## Verification

`python3 scripts/smoke_check.py` — network-free, mock victim, no model loads. Extend per phase:

- **now**: import surface · notebook module paths · `importlib.resources` on both packaged JSONs ·
  seed-registry counts · static attack + `AttackEvaluator` end-to-end.
- **B**: `Instance` round-trips through `copy()`; `AttackDataset` slicing/`group_by`; the 360
  dual-verifiable records map to Instances with both targets intact.
- **C**: registry counts per `usage × variant × method` (incl. AutoDAN 128/41); every template
  either contains `{query}` or is registered as a system prompt; byte-equality of ported upstream
  templates against `code/attack/EasyJailbreak-master/.../seed_template.json`.
- **D**: each success evaluator reproduces `check_ipi_success` on the same inputs (differential
  test against the pre-refactor function, as `check_pisanitizer_fidelity.py` does).
- **F**: MCTS selection/back-prop matches upstream on a seeded synthetic tree.
- **H**: per recipe, compliant victim → success, refusing victim → budget exhausted, against a real
  `DualVerifiableDataset` scenario.
