"""Mock Python REPL environment for Phase 2 multi-turn tool-use training.

Provides a sandboxed Python REPL as a tool. Problems are designed so
the model needs 2-3 tool calls to solve them. Binary reward: 1.0 if
the correct numeric answer appears in the model's final response.
"""

from __future__ import annotations

import io
import random
import re
import signal
from contextlib import redirect_stdout
from contextlib import contextmanager
from typing import Any

from .reward_text import strip_think_blocks
from .types import Message, ToolCall, ToolResponse


_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute Python code and return the result. Use print() to show output.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                }
            },
            "required": ["code"],
        },
    },
}

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


class SandboxTimeoutError(TimeoutError):
    pass


@contextmanager
def _time_limit(seconds: float):
    def _raise_timeout(signum, frame):
        raise SandboxTimeoutError(f"execution exceeded {seconds:.1f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _exec_sandbox(code: str, namespace: dict[str, Any], timeout: float = 1.0) -> str:
    """Execute code in a restricted sandbox. Returns stdout or repr of last expression."""
    buf = io.StringIO()

    # Restrict builtins
    sandbox_globals = {"__builtins__": dict(_SAFE_BUILTINS)}
    sandbox_globals.update(namespace)

    # Try to evaluate as expression first (for things like "x * 3 - 7")
    try:
        with _time_limit(timeout), redirect_stdout(buf):
            result = eval(compile(code.strip(), "<repl>", "eval"), sandbox_globals)
        # Capture any side-effect variables back
        namespace.update({k: v for k, v in sandbox_globals.items()
                         if k != "__builtins__"})
        printed = buf.getvalue()
        if printed:
            return printed.rstrip("\n")
        if result is not None:
            return repr(result)
        return ""
    except SyntaxError:
        pass

    # Fall back to exec for statements
    buf = io.StringIO()
    with _time_limit(timeout), redirect_stdout(buf):
        exec(compile(code, "<repl>", "exec"), sandbox_globals)  # noqa: S102

    # Capture any side-effect variables back
    namespace.update({k: v for k, v in sandbox_globals.items()
                     if k != "__builtins__"})

    output = buf.getvalue().rstrip("\n")
    return output


# ---------------------------------------------------------------------------
# Problem generators — each returns (question, answer)
# ---------------------------------------------------------------------------

def _prob_multi_step_arithmetic(rng: random.Random) -> tuple[str, int]:
    """Two multiplications added together. Needs 2-3 steps."""
    a, b = rng.randint(11, 49), rng.randint(11, 49)
    c, d = rng.randint(11, 49), rng.randint(11, 49)
    answer = a * b + c * d
    question = (
        f"Use the Python tool to compute ({a} * {b}) + ({c} * {d}). "
        f"Show your work step by step, then give the final numeric answer."
    )
    return question, answer


def _prob_variable_tracking(rng: random.Random) -> tuple[str, int]:
    """Assign variables and compute. Needs 2 steps."""
    x = rng.randint(10, 99)
    m = rng.randint(2, 9)
    s = rng.randint(1, 50)
    answer = x * m - s
    question = (
        f"Use the Python tool to compute: if x = {x} and y = x * {m} - {s}, what is y? "
        f"Show your work, then give the final numeric answer."
    )
    return question, answer


def _prob_sum_list(rng: random.Random) -> tuple[str, int]:
    """Sum a list of numbers. Straightforward but needs tool use."""
    nums = [rng.randint(1, 99) for _ in range(rng.randint(5, 8))]
    answer = sum(nums)
    question = (
        f"Use the Python tool to compute the sum of {nums}. "
        f"Give the final numeric answer."
    )
    return question, answer


def _prob_fibonacci(rng: random.Random) -> tuple[str, int]:
    """Compute nth Fibonacci number. Needs a loop."""
    n = rng.randint(8, 15)
    # Compute answer: fib(0)=0, fib(1)=1, ...
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    answer = b
    question = (
        f"Use the Python tool to compute the {n}th Fibonacci number "
        f"(where fib(1) = 1, fib(2) = 1, fib(3) = 2, ...). "
        f"Give the final numeric answer."
    )
    return question, answer


def _prob_nested_arithmetic(rng: random.Random) -> tuple[str, int]:
    """Three-step nested computation."""
    a = rng.randint(10, 50)
    b = rng.randint(10, 50)
    c = rng.randint(2, 9)
    answer = (a + b) ** 2 % (c * 10 + 1)
    mod = c * 10 + 1
    question = (
        f"Use the Python tool to compute ({a} + {b}) ** 2 % {mod}. "
        f"Show your work step by step, then give the final numeric answer."
    )
    return question, answer


_PROBLEMS = [
    _prob_multi_step_arithmetic,
    _prob_variable_tracking,
    _prob_sum_list,
    _prob_fibonacci,
    _prob_nested_arithmetic,
]


def _extract_number(text: str) -> int | None:
    """Extract the last integer from model output (stripping think blocks)."""
    text = strip_think_blocks(text)
    numbers = re.findall(r"-?\d+", text)
    if not numbers:
        return None
    return int(numbers[-1])


class MockREPLEnv:
    """Mock Python REPL environment for multi-turn tool-use training."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._answer: int = 0
        self._namespace: dict[str, Any] = {}

    @property
    def tools(self) -> list[dict]:
        return [_TOOL_SPEC]

    def reset(self, seed: int) -> list[Message]:
        rng = random.Random(seed)
        prob_fn = rng.choice(_PROBLEMS)
        question, self._answer = prob_fn(rng)
        self._namespace = {}
        return [Message(role="user", content=question)]

    def step(self, call: ToolCall) -> tuple[ToolResponse, bool]:
        if call.name != "python":
            return ToolResponse(
                content=f"Unknown tool: {call.name!r}. Available tools: python",
                is_error=True,
            ), False

        code = call.arguments.get("code", "")
        if not isinstance(code, str) or not code.strip():
            return ToolResponse(
                content="Error: empty code",
                is_error=True,
            ), False

        try:
            output = _exec_sandbox(code, self._namespace)
        except Exception as e:
            return ToolResponse(
                content=f"{type(e).__name__}: {e}",
                is_error=True,
            ), False

        return ToolResponse(content=output), False

    def reward(self, trajectory) -> float:
        text = trajectory.decode_last_gen_turn(self.tokenizer)
        extracted = _extract_number(text)
        if extracted is None:
            return 0.0
        return float(extracted == self._answer)
