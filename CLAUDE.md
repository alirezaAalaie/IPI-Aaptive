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
  judges.py             EditDistanceJudge · IPILLMJudge · GPTJudge · KeywordJudge
  prompts.py            attacker + judge system prompts (IPI-reframed)
  dataset.py            IPIScenario · ManualIPIDataset · BipiaDataset · HijackDataset
                        · DualVerifiableDataset · AgentDojoDataset
  evaluator.py          check_ipi_success · make_scenario_target_fn · AttackEvaluator · ScenarioResult
  runner.py             run_attack / run_experiment (scenario-level convenience API)
  target.py             TargetLLM (Victim wrapping UnifiedLLM) + make_target
  config.py             shared hyperparameter defaults
  attacks/              tap · pair · adaptive(RS/Beam-RS) · beast · autodan · gcg · static_injection
                        · deepinception · ica · multilingual · renellm · gptfuzzer
                        · ipi_seeds (shared seed pool for AutoDAN + GPTFuzzer)
  defenses/             base(DefendedVictim) · in_context · secalign/ · struq/

code/                   ← vendored reference implementations from other papers (READ-ONLY)
  attack/               AutoDAN · BEAST · EasyJailbreak · JailbreakingLLMs(PAIR)
                        · llm-adaptive-attacks · Open-Prompt-Injection · TAP
  defense/              SecAlign · StruQ · spotlighting-datamarking
  benchmark/            BIPIA
papers/                 PDFs for the above
experiments/            Kaggle notebooks (the actual run surface)
scripts/                dataset builders (setup_struq_data, build_dual_verifiable_dataset)
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
def run_scenario(self, target: Victim, scenario: IPIScenario, verbose=False) -> ScenarioResult
```

Every attack module also exports a bare `run_*()` function (the algorithm, no class ceremony)
plus a `*Result` dataclass. `run_scenario` is a thin adapter over it via
`make_scenario_target_fn(scenario, target)`. Follow this shape for new attacks — the notebooks
use both entry points.

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
- Success checking is centralised in `evaluator.check_ipi_success(response, target_str, mode)`
  with modes `contains` / `exact` / `function_name`. Don't hand-roll per-attack checks.
- `code/` is vendored reference material. **Do not edit it.** Port *from* it into `ipi/`.

## Known gotchas

- **TAP report-config still needs setting** (`attacks/tap.py`). The unbounded-conversation bug is
  fixed (`keep_last_n`, default 3), but the default `on_topic_prune=False`, `width=5`,
  `branching=2` is *not* paper-TAP. For a row labelled "TAP" set `on_topic_prune=True`,
  `width=10`, `branching=4`. See `docs/easyjailbreak-audit.md`.
- **PAIR has no JSON prefill.** The reference implementation prefills the attacker's reply with
  `{"improvement": "", "prompt": "` for local HF attacker models. Without it, local attacker
  models fail JSON parsing often. Fine for API attackers.
- On parse failure PAIR does `continue` (skips the stream, **does not count a query**), whereas
  the reference falls back to attacking with the raw goal. Query counts are therefore not
  directly comparable to published PAIR numbers.
- StruQ/SecAlign training lives under `ipi/defenses/{struq,secalign}/trainer.py` and has had
  repeated multi-GPU / TRL-compat breakage (see recent commits). Their notebooks are separate:
  `experiments/{struq,secalign}_defense_notebook.ipynb`.

## Current work

Gathering and validating **defense baselines** and the **adversaries** to evaluate them against.
`code/attack/EasyJailbreak-master/` was added recently as a source of additional attacks —
see **`docs/easyjailbreak-audit.md`** for the fidelity comparison of the four attacks we share
with it (PAIR, TAP, GCG, AutoDAN) and the triage of its other nine attacks.
