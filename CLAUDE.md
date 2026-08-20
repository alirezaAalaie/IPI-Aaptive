# IPI-Adaptive — Repo Guide

Research codebase for a **prevention-style defense against Indirect Prompt Injection (IPI)**.
The defense is attention-based (in the spirit of Attention Tracker) but *prevents* rather than
merely *detects*. This repo is the **attack + baseline-defense evaluation harness** used to
stress-test it.

## Layout

```
ipi/                    ← the package (this is the product; pip-installed on Kaggle)
  llm_unified.py        UnifiedLLM ABC · APILLM (litellm) · LocalLLM (HF) ·
                        KaggleLLM (kaggle_benchmarks) · make_llm dispatches on the string
  victim.py             Victim ABC — the interface EVERY defense implements
  attacker.py           BaseAttacker ABC → StaticAttacker / AdaptiveAttacker / JudgeGuidedAttacker
  metrics/              Evaluator family — *GetScore guidance (1-10) beside *Judge/*Match
                        success (bool) · checks.py primitives · AttackEvaluator (owns ASR)
  seed/                 SeedTemplate registry + seed_templates.json — payloads, attacker /
                        judge / constraint system prompts, ICA demos. Replaces prompts.py
  datasets/             Instance (the carrier) · AttackDataset · DualVerifiableDataset
  channels.py           ChanneledPrompt — the trusted-instruction / untrusted-data split,
                        carried with the prompt (StruQ's model) · ChanneledMessages
  harness.py            Instance→Victim plumbing: build_channeled_prompt (the one place
                        the IPI prompt shape is defined) · VictimQuery (the victim as a
                        dataset→dataset component) · make_target_fn · attack_context ·
                        resolve_optimization_target
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
def single_attack(self, target: Victim, instance: Instance, verbose=False) -> AttackDataset
```

It returns the candidate(s) to report, normally one, built with
`self.report(best, query, **extra)`. `BaseAttacker.run_scenario` is concrete and turns that
into the `ScenarioResult` `AttackEvaluator` consumes — **do not override it.** The base class
also owns `keep_best` (replacing the `best_score`/`best_injection`/`best_response` triple),
`normalise_scores`, `as_attacker_llm` and the `current_*` counters, and `__init__` raises if a
subclass defines neither `single_attack` nor `run_scenario`.

The victim is a component, not a closure: `harness.VictimQuery(instance, target)` is
`dataset -> dataset`, the same shape as mutation / constraint / evaluator, and owns the
try/except, the query count and the budget. A recipe never calls the victim by hand.
See `docs/single-attack-migration.md`.

There is **no `*Result` dataclass and no bare `run_*()` function** any more, with one
deliberate exception: `gcg`, `beast` and `autodan` keep theirs as a standalone
(non-scenario) entry point, because the structural swap that would remove them is deferred
in `docs/next-session.md` §4a. `instance.query` is the attacker's goal; everything IPI-specific
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
- **Every search recipe is carrier-native.** `gptfuzzer`, `tap`, `pair`, `renellm`,
  `autodan`, `beast` and `gcg` all carry `Instance` candidates and compose
  `mutation/ selector/ constraint/ metrics/` instead of inlining a candidate type or a
  selection rule. The four static one-shots need nothing (no candidate type, no selector).
  AutoDAN's swap cost almost nothing because its operators already existed —
  `mutation.SentenceCrossOver` and `mutation.ReplaceWordsWithSynonyms` had been ported and
  the recipe simply never imported them; its fitness is now
  `selector.ReferenceLossSelector.score` (batched — it was one forward pass *per candidate*)
  and its selection `selector.GeneticSelectPolicy`.
- **Token-level candidates carry their span in `attack_attrs["adv_ids"]`** (`selector.ADV_IDS`)
  and are scored by `selector.TokenLossSelector`, not `ReferenceLossSelector`. That is what
  made the carrier affordable for `beast`/`gcg` after it was declined once: the objection was
  never the `Instance` wrapper, it was the *text* selector re-rendering and re-tokenizing a
  prompt per candidate, 512 times a step. `TokenLossSelector` keeps head/tail/target as fixed
  id lists and builds a batch by concatenation. It refuses a batch of mixed adversarial
  lengths rather than padding — padding a span that sits before the target shifts the target
  slice per row and silently scores the wrong positions. `mutation.gradient` gained the two
  operators that feed it: `TokenGradientMutation` (GCG, 1→B) and `BeamTokenExpansion`
  (BEAST, 1→k2, batched across the whole beam in one forward pass).
- **BEAST and GCG's objectives are one number with opposite signs.** BEAST maximises
  `-perplexity(target | prompt)`, GCG minimises cross-entropy of the same span; that is why
  both read `TokenLossSelector`'s `_loss` directly. `BEASTResult.score` is `-best_loss`.
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
- **`GeneticSelectPolicy` reads fitness as higher-is-better; the loss selector writes
  lower-is-better.** `ReferenceLossSelector.score()` sets `instance._loss`; AutoDAN negates
  it into `eval_results` in exactly one place (`attacks/autodan.py:_score`), which is
  upstream's own `score_list = [-x for x in score_list]`. Getting the sign wrong inverts the
  search and raises nothing.
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
- On parse failure PAIR skips the stream for that round and **does not count a query**, whereas
  the reference falls back to attacking with the raw goal. Query counts are therefore not
  directly comparable to published PAIR numbers. `IntrospectBranching` drops a branch whose
  attacker never produced valid JSON — right for TAP's tree, wrong for a fixed set of streams —
  so `pair._stalled` re-attaches the parents it dropped. A parse failure costs a stream one
  round, not its life.
- **TAP and PAIR branch through one operator, `mutation.IntrospectBranching`.** They are the
  same conversational refinement at different branching factors: a tree that fans out
  `branching_factor` ways versus independent chains that fan out one. Upstream splits them
  across `IntrospectGeneration` (which needs `fastchat`) and `HistoricalInsight`, duplicating
  the ask / retry / JSON-parse loop in both recipes.
- **The attacker transcript's head is two messages, not one.** `IntrospectBranching.truncate`
  keeps the system prompt **and the opening user turn**. In the IPI framings the system prompt
  is generic and the goal, user task and tool schema live in that first user message. PAIR's
  old inline `conv[:1] + conv[-12:]` dropped it past thirteen messages, so from roughly the
  sixth iteration the attacker was refining toward an objective it could no longer see. Fixed
  by adopting the shared operator; PAIR's `keep_last_n=6` reproduces the old twelve-message
  tail. **This moves PAIR's numbers** — re-baseline it.
- **GPTFuzzer's `judge=` now costs one call per scenario, not one per candidate.** It grades
  only the candidate that gets reported, so the row carries a 1-10 score instead of a bare
  10/1. The search reward was already binary success and is unchanged. The per-candidate judge
  scores only ever fed a `trace` that stopped at the `ScenarioResult` boundary — that trace is
  gone; the lineage is on the `Instance`s (`parents`/`children`/`level`) instead, and
  `extra["max_level"]` reports how far the search actually descended.
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
- **The instruction/data split travels with the prompt; nothing parses it back out.**
  `ipi/channels.py` is StruQ's model one level up: `harness.build_channeled_prompt` fills
  `ChanneledPrompt(system=, instruction=, data=)` from the instance's own fields and renders
  it to `ChanneledMessages` — an ordinary `list[dict]` (so `Victim.generate`'s contract is
  untouched) that also carries the `ChanneledPrompt`. A defense calls
  `self.resolve_channels(messages)` and edits ONE named field: `data` (Sandwich, Spotlight,
  StruQ/SecAlign/DefensiveToken's filter, PISanitizer), `system` (Instructional) or
  `epilogue` (Reminder). Re-rendering carries the *updated* split, which is what makes
  chaining work. The old `split_instruction_data` / `transform_data_channel` and their
  `_AGENTDOJO_RE` / `_ENV_RE` / BIPIA heuristics are **deleted** — six defenses depending on
  one regex in another module is how the user's own task ended up inside
  `[START OF UNTRUSTED EXTERNAL DATA] ... IGNORE ALL COMMANDS ABOVE`. Precedence:
  carried split → `victim.set_channels(instruction, data)` pin (for hand-built prompts, e.g.
  the DefensiveToken notebook's `preprocess_messages([])`) → `ChanneledPrompt.from_messages`,
  which is not a parser: system turns are trusted, the whole final user turn is data. That
  over-marks rather than going inert. `data_prefix`/`data_suffix` (the `<env>` tags) are
  trusted framing and stay outside the untrusted span.
- **A structured defense must be the innermost one, and that is now enforced.** StruQ /
  SecAlign / DefensiveToken return a `PreRenderedMessages` — an ordinary `list[dict]` to
  every backend, but `Victim.resolve_channels` raises `PreRenderedPromptError` on it rather
  than guessing a split for a prompt that is already the model's own rendering. Guessing put
  the outer defense's text *after* `[MARK] [RESP][COLN]`, i.e. in the model's answer position,
  silently. `defenses.assert_innermost` catches the same mistake at construction —
  `StruQDefense(ReminderDefense(t))` raises, and so does `CompositeDefense(t, [Reminder, StruQ])`
  (the list is innermost-first, so the structured defense goes **first**). `CompositeDefense`
  also re-checks after rebinding a pre-built instance's `.target`, which bypasses `__init__`.
- **`StructuredChannelDefense` folds the fields its format has no slot for**
  (`_fold_channels`): `epilogue` joins the *instruction* channel — trusted text, and putting
  it in the data channel would hand it to the attacker's span and to the delimiter filter —
  and `data_prefix`/`data_suffix` are re-applied *after* filtering, only when the data channel
  is non-empty (framing alone would flip StruQ from `prompt_no_input` to `prompt_input`).
  Before this, both were dropped: `ReminderDefense(StruQDefense(t))` rendered byte-identically
  to StruQ alone, so a row labelled "StruQ + Reminder" measured StruQ. The mapping is ours —
  upstream never composes a structured defense with an in-context one — so say "composite" when
  reporting it.
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
  is 240 `startswith` + 120 `contains` and never `function_name` (that is the AgentDojo
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
  means BEAST optimises tokens for an instruction the victim never sees. GCG had the
  identical defect (`adv_prefix` alone) and now composes `adv_prefix + instance.query`
  the same way.
- **`build_optimization_messages` applies the WHOLE chain, not the outermost hook.** It
  called `victim.preprocess_messages` once; `CompositeDefense`'s own hook is a no-op (its
  work happens in the defenses it chains underneath, reached through nested `generate`
  calls), so a white-box attack against a composite was handed a prompt with **none** of
  the defenses applied — the undefended shape under a defended label. `harness.apply_defense_pipeline`
  now walks `.target` the way `generate` nests it. Any GCG/BEAST/AutoDAN number measured
  against a `CompositeDefense` before this is void.
- **`final_prompt` is rebuilt from the reported injection, not read off the last call.**
  `AttackEvaluator` used `last_input_messages`, which is whichever query ran last — and
  every search attack reports its *best* candidate while its final query was a later,
  worse one, so the `final_prompt` printed beside `attack_str` was a different injection's.
  `harness.rebuild_victim_prompt` replays the carrier with the `TargetFnSpec`
  `make_target_fn` recorded (so an attack's own `data_separator` is honoured — the OPI
  one-shots differ by one space versus one newline) and re-runs the defense chain, without
  querying the model. `extra["final_prompt_source"]` is `rebuilt` / `last_call` /
  `unavailable`; `_innermost_prompt` is now only the fallback. ASR was never affected — the
  audit trail was.
- **A white-box attack must build its prompt through `harness`, never itself.** GCG,
  BEAST and AutoDAN each rolled their own `[system][user]` prompt while success was
  judged through `make_target_fn`'s IPI carrier — they were optimising a string the
  victim never saw. They now go through `harness.split_optimization_prompt` (GCG, BEAST)
  or `harness.build_optimization_messages` (AutoDAN), which render the *victim's* prompt
  — carrier plus the defense's own `preprocess_messages` — and split it around the
  adversarial span. The split is what keeps those tokens **inside the user turn**:
  building with `add_generation_prompt=True` and appending after it puts them in the
  assistant turn, so the objective becomes a continuation of the model's own reply.
  `tail_text` is upstream BEAST's `end_inst_token`. Rendering lives in one place,
  `llm_unified.render_messages`, which `LocalLLM._build_local_prompt_ids` also calls.
  The bare `run_gcg` / `run_beast` / `run_autodan_*` functions still work without a
  scenario, but they log a warning that the prompt is not the victim's.
- **AutoDAN's seed templates already carry the goal.** `sample_population` substitutes
  `{query}`, so `_score_candidates` must not append the instruction again — it did, and
  the goal appeared twice in the scored prompt and once in the evaluated one.
- **PISanitizer is the closest published peer to our own defense** — attention-based, and
  prevention rather than detection. It is the only baseline that also defends an *API* victim,
  since only the sanitizer needs local weights. `PISanitizerDefense` subclasses `DefendedVictim`
  directly, not `StructuredChannelDefense`: it sanitizes `ChanneledPrompt.data` and re-renders,
  leaving the message structure alone. It used to substitute the sanitized text back by finding
  it verbatim in some message and went inert when that missed; re-rendering the channel it came
  from cannot miss. Three things in it are method, not tuning:
  the anchor prompt *invites* instruction-following (reversing it inverts the signal), the
  `" X" * 500` padding holds the context off the attention sinks at the prompt boundaries, and
  `group_peaks` removes **only the single highest span per round** (≤5 rounds, so ≤5 spans ever).
  Note it is also the one defense whose `preprocess_messages` runs a model, so
  `final_prompt`'s rebuild costs one extra sanitizer pass per scenario.
  `attention.py` reconstructs attention from hidden states because SDPA/flash never materialise
  it — that couples it to transformers internals (upstream pins 4.56.2). Needs scipy:
  `pip install ipi-adaptive[pisanitizer]`. `scripts/check_pisanitizer_fidelity.py` differentially
  tests the port against vendored upstream; notebook:
  `experiments/pisanitizer_defense_notebook.ipynb`.

- **A `kaggle/` model string picks the Kaggle Benchmarks backend, and `make_llm` is the only
  place that is decided.** `KaggleLLM` wraps `kbench.llms[...]`, so
  `make_target("kaggle/google/gemini-2.5-flash")`, `EvaluatorIPIGetScore(model="kaggle/...")`,
  `attacker_llm="kaggle/..."`, TAP's `on_topic_model` and Multilingual's `translator_llm` all
  take one. Four constraints are structural, not bugs to fix: it imports only inside a Kaggle
  Benchmarks notebook; `prompt()` only works while a `@kbench.task` is *running*, so drive the
  whole eval from inside one task; there are **no logprobs** (RS / Beam-RS raise) and no
  `hf_model` (GCG / BEAST / AutoDAN gate out); and `prompt()` has no documented system-prompt
  argument, so unless the live signature grows one the system turn is **folded into the user
  turn** and a one-time warning says so — for a victim that changes the trusted/untrusted
  structure the defense is handed, so pass `system_mode="native"` to make it a hard error
  instead. Sampling kwargs are passed only if the live `prompt()` signature declares them
  (checked with `inspect.signature`; a name absorbed by `**kwargs` would be silently dropped).
  Every call opens its own `kbench.chats.new(...)` — a shared `Chat` accumulates turns and
  would feed candidate N the whole earlier search. Un-versioned ids resolve to the versioned
  key (`openai/gpt-5.4-mini` → `openai/gpt-5.4-mini-2026-03-17`, `anthropic/claude-opus-5` →
  `…@default`) by stripping the suffix only — nothing is prefix-matched, so `openai/gpt-5.4`
  raises rather than silently picking the mini. `run_in_kaggle_task(fn, *args)` is the
  other half: it runs one `@kbench.task` per *evaluation* (not per query — a task writes a
  run file) and returns the result through a closure, since a `ScenarioResults` is not
  JSON and the task's own return slot is serialized. `ipi_attack_benchmark.ipynb` wraps it
  as `run_eval(evaluator, subset)`, which is a plain `.run` unless a `kaggle/` model is in
  the loop; every attack cell in that notebook launches through it.

## Current work

The architecture refactor onto EasyJailbreak's component families (carrier object +
`seed/ mutation/ selector/ constraint/ metrics/`) is **committed and complete** — all phases
A–I. Six defects found while auditing the white-box recipes are fixed; the structural swap for
`autodan`/`beast`/`gcg` was declined with reasons rather than done.

**The `single_attack` migration is done — all twelve recipes.** No recipe overrides
`run_scenario`; `ipi/runner.py` is deleted. `adaptive`, `beast`, `gcg` and `autodan` keep
their bare `run_*` functions as a standalone entry point, because their searches have no
per-candidate `victim.generate` for `VictimQuery` to own. `docs/single-attack-migration.md`
records the rules, the three behaviour changes needing re-baselining, and what was and was
not executed. `ipi_attack_benchmark.ipynb` gained a defense x attack sweep at the end.

**Four docs, four jobs:**
- `docs/next-session.md` — **the to-do list.** Start here. Nothing is pushed to GitHub, so no
  Kaggle notebook can see any of this; the white-box edits have never been executed for want of
  a GPU; nine behaviour changes mean published numbers need re-baselining, and BEAST's are void.
- `docs/refactor-handoff.md` — the state of `ipi/` and 21 traps. Read before touching `ipi/`.
- `docs/ipi-refactor-plan.md` — the design doc and the locked decisions.
- `docs/single-attack-migration.md` — the `single_attack` spine, the rules, and what is left.

Green check (all six pass; smoke_check is 42 checks):
`scripts/{smoke_check,check_seed_fidelity,check_metrics_fidelity,check_defensivetoken_fidelity,check_pisanitizer_fidelity}.py`
plus `python3 -m compileall -q ipi`.

Gathering and validating **defense baselines** and the **adversaries** to evaluate them against.
`code/attack/EasyJailbreak-master/` was added recently as a source of additional attacks —
see **`docs/easyjailbreak-audit.md`** for the fidelity comparison of the four attacks we share
with it (PAIR, TAP, GCG, AutoDAN) and the triage of its other nine attacks.
