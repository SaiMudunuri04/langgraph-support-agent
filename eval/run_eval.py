"""Golden-set eval: runs the graph in mock mode over eval/golden_set.json.

Reports pass/total. Exits non-zero on any failure (CI gate).
All data is synthetic and bundled; the mock LLM is deterministic.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["MOCK_MODE"] = "true"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from support_agent.graph import build_graph, run_chat  # noqa: E402
from support_agent.llm import MockChatLLM  # noqa: E402


def main() -> int:
    cases = json.loads(Path("eval/golden_set.json").read_text(encoding="utf-8"))
    app = build_graph(MockChatLLM(), "data")
    passed = 0
    for i, case in enumerate(cases, 1):
        result = run_chat(app, case["message"])
        answer = result["answer"].lower()
        ok_intent = result["intent"] == case["expected_intent"]
        ok_terms = all(t.lower() in answer for t in case["must_contain"])
        ok = ok_intent and ok_terms and bool(answer)
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] case {i}: intent={result['intent']} escalated={result['escalated']}")
        if not ok:
            print(f"   message={case['message']!r}")
            print(f"   answer={result['answer'][:200]!r}")
    print(f"eval: {passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
