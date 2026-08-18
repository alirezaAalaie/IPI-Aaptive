# IPI-Adaptive — Repo Guide

Research codebase for a **prevention-style defense against Indirect Prompt Injection (IPI)**.
The defense is attention-based (in the spirit of Attention Tracker) but *prevents* rather than
merely *detects*. This repo is the **attack + baseline-defense evaluation harness** used to
stress-test it.

## Layout

```
ipi/                    ← the package (this is the product; pip-installed on Kaggle)
  llm_unified.py        UnifiedLLM ABC · APILLM (litellm) · LocalLLM (HF)
  victim.py             Victim ABC — the interface EVERY defense implements
  attacker.py           BaseAttacker ABC → StaticAttacker / AdaptiveAttacker / JudgeGuidedAttacker
  metrics/              Evaluator family — *GetScore guidance (1-10) beside *Judge/*Match
                        success (bool) · checks.py primitives · AttackEvaluator (owns ASR)
  seed/                 SeedTemplate registry + seed_templates.json — payloads, attacker /
                        judge / constraint system prompts, ICA demos. Replaces prompts.py
  datasets/             Instance (the carrier) · AttackDataset · DualVerifiableDataset
  harness.py            Instance→Victim plumbing: make_target_fn (the one place the IPI
                        prompt shape is defined) · attack_context · resolve_optimization_target
  runner.py             run_attack / run_experiment — target_fn-level API, NOT the
                        benchmark path (that is AttackEvaluator over an AttackDataset)
  target.py             TargetLLM (Victim wrapping UnifiedLLM) + make_target
  config.py             shared hyperparameter defaults
  mutation/             MutationBase ops — generation.py (LLM-driven: GPTFuzzer 5,
                        ReNeLLM 6, Jailbroken 2, HistoricalInsight) · rule.py
                        (deterministic: encodings, CodeChameleon, crossover, synonyms)
  selector/             SelectPolicy — Random · RoundRobin · UCB · EXP3 · MCTSExplore
                        · SelectBasedOnScores · ReferenceLossSelector (white-box)
  constraint/           ConstraintBase filters — DeleteOffTopic (TAP pruning) · DeleteHarmLess
                        (ReNeLLM's, opt-in) · PerplexityConstraint
  attacks/              tap · pair · adaptive(RS/Beam-RS) · beast · autodan · gcg · static_injection
                        · deepinception · ica · multilingual · renellm · gptfuzzer
  defenses/             base(DefendedVictim) · channels · in_context · secalign/ · struq/
                        · defensive_token/ · pisanitizer/
  data/                 packaged runtime data (dual_verifiable_dataset.json)

code/                   ← vendored reference implementations from other papers (READ-ONLY)
  attack/               AutoDAN · BEAST · EasyJailbreak · JailbreakingLLMs(PAIR)
                        · llm-adaptive-attacks · Open-Prompt-Injection · TAP
  defense/              SecAlign · StruQ · spotlighting-datamarking
  benchmark/            BIPIA
papers/                 PDFs for the above
experiments/            Kaggle notebooks (the actual run surface)
scripts/                dataset builders (setup_struq_data, build_dual_verifiable_dataset)
                        · smoke_check.py (network-free import/packaging/e2e check)
                        · check_seed_fidelity.py (registry vs vendored upstream)
                        · check_metrics_fidelity.py (success checks pinned to a golden table)
data/ results/          alpaca corpora · saved eval JSON
docs/                   analysis notes (see easyjailbreak-audit.md)
```

## How this actually runs

Experiments run **on Kaggle**, not locally. The notebooks in `experiments/` do:

```python
!pip install git+https://github.com/alirezaAalaie/IPI-Aaptive.git
```

so **`ipi/` must stay pip-installable and import-clean**. Consequences:

- Heavy deps (`torch`, `transformers`, `nltk`) are **optional extras**, imported lazily.
  `attacks/__init__.py` wraps white-box attack imports in `try/except ImportError` and sets the
  names to `None`. Keep that pattern — never make `import ipi` require torch.
- API keys come from Kaggle Secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
  `DEEPSEEK_API_KEY`, `METIS_API_KEY`, `OPENROUTER_API_KEY`).
- Changes must be pushed to GitHub before a notebook can see them.

## Core contracts

**A defense is a `Victim`.** To add one, subclass `ipi.victim.Victim`:

```python
class MyDefense(Victim):
    backend = "local"                     # or "api"
    def generate(self, messages, max_tokens=200, temperature=0.0) -> str: ...
    def get_first_token_logprobs(self, messages, n_top=20) -> dict: ...
```

`get_first_token_logprobs` is **required by the logprob attacks** (RS / Beam-RS). White-box
attacks (GCG, AutoDAN, BEAST) additionally need `hf_model` + `tokenizer` exposed and
`backend == "local"`; they gate on `BaseAttacker.requires_local_target()`.

**An attack is a `BaseAttacker`.** The single method that matters:

```python
def run_scenario(self, target: Victim, instance: Instance, verbose=False) -> ScenarioResult
```

Every attack module also exports a bare `run_*()` function (the algorithm, no class ceremony)
plus a `*Result` dataclass. `run_scenario` is a thin adapter over it via
`harness.make_target_fn(instance, target)`. Follow this shape for new attacks — the notebooks
use both entry points. `instance.query` is the attacker's goal; everything IPI-specific
(`user_task`, `tool_schema`, `pipeline_context`, `target_str`, `optimization_target`) is in
`instance.attack_attrs` and is read through `harness` / `metrics.resolve_attack_target`, never
by hand — `attack_attrs.get` returns `None` on a typo where an attribute would have raised.

**Attack taxonomy** (drives which attacks can run against which target):

| Class | Queries | Needs judge | Needs local model |
|---|---|---|---|
| `StaticAttacker` (Naive/Escape/Ignore/FakeCompletion/Combined, DeepInception, ICA) | 1 | no | no |
| `StaticAttacker` (Multilingual) | 1 per language (best-of) | no | no (needs a translator) |
| `AdaptiveAttacker` (ReNeLLM, GPTFuzzer) | ≤ budget | optional (GPTFuzzer reward) | no (needs attacker LLM) |
| `JudgeGuidedAttacker` (TAP, PAIR) | many | **yes** | no |
| logprob search (RS, Beam-RS) | many | optional | logprobs only |
| white-box (GCG, AutoDAN, BEAST) | many | no | **yes** |

## Conventions

- `prompt_mode` on TAP/PAIR: `"ipi_single"` (default) · `"ipi_universal"` · `"original"`.
  `"original"` exists for paper-reproduction; the IPI modes swap the JSON key from
  `"prompt"` to `"injection_string"` and reframe the system prompt around tool-calling.
- Every attack module docstring carries an **"IPI adaptations vs original"** section. Keep this
  up to date — it is the audit trail for what deviates from each paper, and reviewers will ask.
- Success checking is centralised in `metrics.check_ipi_success(response, target_str, mode)`
  with modes `contains` / `exact_call` / `startswith` / `function_name`. Don't hand-roll
  per-attack checks — and don't edit it without running `scripts/check_metrics_fidelity.py`,
  which pins every branch to a golden table because every ASR the repo has published comes
  out of that function.
- `code/` is vendored reference material. **Do not edit it.** Port *from* it into `ipi/`.

## Known gotchas

- **TAP report-config still needs setting** (`attacks/tap.py`). The unbounded-conversation bug is
  fixed (`keep_last_n`, default 3), but the default `on_topic_prune=False`, `width=5`,
  `branching=2` is *not* paper-TAP. For a row labelled "TAP" set `on_topic_prune=True`,
  `width=10`, `branching=4`. See `docs/easyjailbreak-audit.md`.
- **Four recipes are carrier-native** (`gptfuzzer`, `tap`, `pair`, `renellm`): their candidates
  are `Instance`s and they compose `mutation/ selector/ constraint/ metrics/` instead of
  inlining a candidate type or a selection rule. The four static one-shots need nothing (no
  candidate type, no selector); `autodan`/`beast`/`gcg` are still to do.
  `docs/refactor-handoff.md` §8 tracks which is which. Two behaviour notes: GPTFuzzer's search
  reward is now **binary success**, not the judge's score (its `judge=` only annotates the
  trace), and RS/Beam-RS finally *use* the `judge=` their classes always accepted — to gate
  early stopping only, never the returned `success`.
- **EasyJailbreak-ported attacks auto-resolve their eval mode** (`deepinception`, `ica`,
  `multilingual`, `renellm`, `gptfuzzer`). Their `eval_mode` defaults to `None`, meaning
  "read `scenario.metadata['attack_eval_mode']`" via `metrics.resolve_attack_target`. Don't
  hard-code `eval_mode='function_name'` for these — that's the AgentDojo convention and never
  matches `DualVerifiableDataset`'s literal-string goals (`startswith`/`contains`). The final ASR
  is recomputed by `AttackEvaluator` from the same metadata regardless; the attacker's mode only
  steers its own early-stop / best-candidate choice.
- **Every fixed string an attack uses lives in `ipi/seed/`.** `SeedTemplate` reads
  `seed/seed_templates.json`, keyed `usage → method → variant → [templates]`, where `usage` ∈
  `attack | judge | constraint | demo` and `variant` is `original` (verbatim from the paper's
  code) or `ipi` (our reframing). There is no `prompts.py` and no `attacks/seeds.py` — attack
  payloads, attacker/judge/constraint system prompts and ICA's in-context demos all come from
  the one registry — including `usage="mutation"`, the wrapper prompt each rule mutation
  attaches. `scripts/check_seed_fidelity.py` proves all 222 `original` templates are
  byte-identical to the vendored upstream (the `mutation` ones are pulled out of
  `mutation/rule/*.py` by AST, since upstream inlines them). The JSON must ship in the wheel — it's
  declared in `[tool.setuptools.package-data]` as `"ipi.seed"`; don't drop that or Kaggle
  installs break at import of any seed-based attack.
- **`variant` selects the paper row vs the IPI row.** `AutoDANAttacker(seed_variant=...)` picks
  the 128 upstream seeds or the 41-template IPI pool (`seed_variant`, not `variant` — that one
  is already `"ga"|"hga"`). `ICAAttacker(variant=...)` picks the 10 authored IPI demonstrations
  (default) or the paper's 30 AdvBench pairs. TAP/PAIR's `prompt_mode` maps onto the variants
  `original` / `ipi` / `ipi_universal`.
- **Guidance and success are both `metrics.Evaluator`, and must not be confused.** A
  `*GetScore` evaluator writes an int 1-10 and only steers an attack's search; a `*Judge` /
  `*Match` writes a bool. `AttackEvaluator` **overwrites** whatever the attack concluded with
  `EvaluatorIPISuccess` against the scenario's ground truth — that is what makes ASR comparable
  across attacks with different internal signals, so never report an attack's own `.success`.
  `Instance.num_jailbreak` is `sum(eval_results)`, so pair a reward-driven selector (UCB, MCTS)
  with a binary evaluator and a 1-10 one with `SelectBasedOnScores`. `EvaluatorUserUtility`
  writes to `instance.utility_results`, deliberately *not* `eval_results`, for the same reason.
  The old `Judge` name is gone: it meant guidance here and success everywhere else.
- **A mutation operator is `MutationBase`, in `ipi/mutation/`.** `generation.py` needs a model
  (bound at construction — `Expand(llm)`, then `op.mutate(text)`); `rule.py` is deterministic.
  Prompts are verbatim upstream and the wrappers live in the seed registry. Two guards in the
  base class are load-bearing: an operator rewriting a *template* falls back to its input if the
  model drops `{query}` (a template without it is un-attackable and the failure is silent), and
  empty output never propagates. `new_child()` records the parent/child edge and bumps `level` —
  that lineage is what a future MCTS selector descends, so don't hand-roll instance copying.
  Deliberately not ported: the four cipher experts, MJPChoices, Artificial, Inception — they
  serve recipes we decided not to add. `gradient/` lands with the GCG migration, extracted from
  our own GCG (upstream's needs their `WhiteBoxModelBase` stack, which our `Victim` replaces).
- **A selector's reward tells you which evaluator it needs.** The bandits (`UCB`, `EXP3`,
  `MCTSExplore`) reward with `Instance.num_jailbreak` = `sum(eval_results)`, so they require a
  **binary** evaluator; `SelectBasedOnScores` reads `eval_results[-1]` as a magnitude and wants a
  1-10 `*GetScore`. Crossing them doesn't error, it silently rescales the search signal.
  `MCTSExploreSelectPolicy` descends `Instance.children` and discounts by `.level`, so anything
  that adds a candidate must go through `MutationBase.new_child()` (sets both) and
  `SelectPolicy.register()` (assigns `index`, the policy's reward slot) — upstream leaves index
  assignment to the recipe and a miss reads another node's reward.
- **`ReferenceLossSelector` batches, and its `_build_batch` is torch-free on purpose.**
  Scoring is one forward pass per `batch_size` candidates, with `-100`-masked labels
  outside the reference span. The padding/label index arithmetic is split into a pure
  Python method so `smoke_check.py` can verify it without a GPU — keep that split. Its
  `batch_size=None` means one batch for the *whole* dataset (upstream's default); set it
  explicitly for gradient-scale candidate sets or it OOMs.
- **A constraint filters, a mutation rewrites, an evaluator scores.** `ipi/constraint/` only
  drops candidates, and every one it drops is a victim query saved — that is the whole point.
  `PerplexityConstraint` is the odd one: perplexity filtering is a published *defense*, used
  here offensively so the search only keeps candidates that would survive it. Don't confuse it
  with `ipi/defenses/`, which is what the victim runs. `DeleteHarmLess` is ported for the
  paper-faithful ReNeLLM row and is deliberately **not** used by the IPI row — our payloads are
  instructions, not harmful content, so it would reject every candidate and starve the search.
- **PAIR has no JSON prefill.** The reference implementation prefills the attacker's reply with
  `{"improvement": "", "prompt": "` for local HF attacker models. Without it, local attacker
  models fail JSON parsing often. Fine for API attackers.
- On parse failure PAIR does `continue` (skips the stream, **does not count a query**), whereas
  the reference falls back to attacking with the raw goal. Query counts are therefore not
  directly comparable to published PAIR numbers.
- StruQ/SecAlign training lives under `ipi/defenses/{struq,secalign}/trainer.py` and has had
  repeated multi-GPU / TRL-compat breakage (see recent commits). Their notebooks are separate:
  `experiments/{struq,secalign}_defense_notebook.ipynb`.
- **StruQ/SecAlign delimiters are load-bearing strings.** `SpclSpclSpcl` is
  `'[MARK] [INST][COLN]'` — no space before `[COLN]`. Both configs build them by the same
  concatenations as upstream so they can't drift; don't hand-edit them into literals. A single
  stray space silently un-matches every released checkpoint.
- **StruQ training order is load-bearing.** The tokenizer must be resized *before* the corpus is
  tokenized, and `modules_to_save=["embed_tokens","lm_head"]` must stay in the `LoraConfig` —
  otherwise the five delimiter tokens either tokenize differently at train vs inference time, or
  never receive gradient at all. Both failures are silent.
- **The defense is two halves.** The structured prompt only holds because `recursive_filter`
  strips `FILTERED_TOKENS` from the data channel. Don't ship an ASR number with
  `apply_defensive_filter=False` unless that ablation is the point.
- `ipi/defenses/channels.py` recovers the instruction/data split from a harness messages list.
  Role-based guessing does not work — in the AgentDojo shape the legitimate task is *inside* the
  user turn, and in the BIPIA shape the untrusted context is inside the *system* turn. When a
  caller knows the split, `defense.set_channels(instruction, data)` beats parsing.
- Structured defenses emit a single `{"role": "raw"}` turn, which `LocalLLM._build_local_prompt_ids`
  encodes verbatim. These models are fine-tuned on bare Alpaca-format strings; wrapping them in a
  chat template feeds them a format they were never aligned on.
- **DefensiveToken fails silently by default.** An unpatched tokenizer renders `add_defensive_tokens`
  as an undefined Jinja variable — falsy — so the prompt comes out with no tokens in it and the run
  reports the *undefended* ASR under a defended label. `DefensiveTokenDefense.__init__` probes the
  template both ways and raises; `apply_defensive_tokens` reads each embedding row back after
  writing it (a `meta`/offloaded/quantised table swallows the write). Don't remove either check.
  Its chat templates are also **not** the models' stock ones — they're the Meta-SecAlign-style
  formats the embeddings were optimised against — and `system` carries the *trusted* instruction
  while `user` carries the *untrusted* data, which is inverted from Meta-SecAlign's own convention.
  `scripts/check_defensivetoken_fidelity.py` asserts byte-equality against the vendored upstream.
  Notebook: `experiments/defensivetoken_defense_notebook.ipynb`. Tokens are per-model and do not
  transfer; only the four models in `SUPPORTED_MODELS` have released weights.
- **`eval_mode` is owned by the data.** Every `*Attacker` defaults to `eval_mode=None` and
  resolves it with `metrics.resolve_attack_target(instance, self.eval_mode)`; the benchmark
  is 180 `startswith` + 180 `contains` and never `function_name` (that is the AgentDojo
  convention). Pass a string only to force a deliberate deviation, and never add a
  hard-coded default — a smoke check fails if you do.
- **The optimisation target and the eval target are two different strings.** GCG, RS and
  Beam-RS optimise toward `harness.resolve_optimization_target(instance)` — which must be a
  real token sequence — and judge success against `eval_target_str`, the instance's own
  `target_str`. They differ on 120 of 360 scenarios, and collapsing them makes the search
  stop on a partial match. Any new search attack needs both.
- **`n_queries` means calls to the victim, everywhere.** Model forward/backward passes go in
  `n_forward_passes` (on `BEASTResult`/`GCGResult`, surfaced in `ScenarioResult.extra`).
  BEAST used to report `new_gen_length * ngram * k1 * k2` as its query count and GCG ~513
  per step; both actually query once per step. Keep the two counters distinct or the
  avg_queries column stops meaning anything.
- **BEAST's `prompt_prefix` is framing, not the whole prompt.** `run_scenario` appends
  `instance.query` to it. Passing it alone — which the code did until this was found —
  means BEAST optimises tokens for an instruction the victim never sees.
- **PISanitizer is the closest published peer to our own defense** — attention-based, and
  prevention rather than detection. It is the only baseline that also defends an *API* victim,
  since only the sanitizer needs local weights. `PISanitizerDefense` subclasses `DefendedVictim`
  directly, not `StructuredChannelDefense`: it edits the untrusted span in place and leaves the
  message structure alone. If the recovered span isn't found verbatim in a single message the
  defense is inert — it warns; use `set_channels`. Three things in it are method, not tuning:
  the anchor prompt *invites* instruction-following (reversing it inverts the signal), the
  `" X" * 500` padding holds the context off the attention sinks at the prompt boundaries, and
  `group_peaks` removes **only the single highest span per round** (≤5 rounds, so ≤5 spans ever).
  `attention.py` reconstructs attention from hidden states because SDPA/flash never materialise
  it — that couples it to transformers internals (upstream pins 4.56.2). Needs scipy:
  `pip install ipi-adaptive[pisanitizer]`. `scripts/check_pisanitizer_fidelity.py` differentially
  tests the port against vendored upstream; notebook:
  `experiments/pisanitizer_defense_notebook.ipynb`.

## Current work

The architecture refactor onto EasyJailbreak's component families (carrier object +
`seed/ mutation/ selector/ constraint/ metrics/`) is **committed**. Phases A–G and I are
shipped; **Phase H has 3 recipes left** — `autodan`, `beast` and `gcg`, all torch-gated, all
needing a GPU machine to migrate safely. Design doc: `docs/ipi-refactor-plan.md`. **What is
done, what is left and the traps: `docs/refactor-handoff.md` — read it before touching `ipi/`,
and start at its §0.** Green check:
`python3 scripts/smoke_check.py`, `python3 scripts/check_seed_fidelity.py` and
`python3 scripts/check_metrics_fidelity.py`.

Gathering and validating **defense baselines** and the **adversaries** to evaluate them against.
`code/attack/EasyJailbreak-master/` was added recently as a source of additional attacks —
see **`docs/easyjailbreak-audit.md`** for the fidelity comparison of the four attacks we share
with it (PAIR, TAP, GCG, AutoDAN) and the triage of its other nine attacks.
