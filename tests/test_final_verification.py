import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.text_llm import TextLLM
from agents.coordinator.main import CoordinatorSession
from agents.coordinator.nodes import NodeContext
from backend.services.safety_kernel import SafetyKernel
from utils.models import StrategyRecommendations

# Load model and data
llm = TextLLM("backend-engines/models/Qwen3-1.7B-Q4_K_M.gguf")
strat_files = sorted(Path("data/full_pipeline_results").glob("*_strategies.json"))
risk_files = sorted(Path("data/full_pipeline_results").glob("*_risk_results.json"))

if not strat_files or not risk_files:
    print("SKIP: No test data files found")
    sys.exit(0)

# Create coordinator with TextLLM
ctx = NodeContext()
ctx.fallback_only = True
ctx.safety_kernel = SafetyKernel()
ctx.output_dir = "data/full_pipeline_results"
ctx.text_llm = llm

coordinator = CoordinatorSession(
    session_id="final-test", ctx=ctx,
    image_path="test.png", fallback_only=True,
)
ps = coordinator.get_pipeline_state()
with open(strat_files[0]) as f:
    ps["strategy_result"] = StrategyRecommendations(**json.load(f))
with open(risk_files[0]) as f:
    ps["risk_result"] = json.load(f)

results = []

def test(query, expected_action, max_mods=None, min_mods=None, description=""):
    """Send a message and check the result."""
    t0 = time.time()
    result = coordinator.handle_message(query)
    elapsed = (time.time() - t0) * 1000

    action = result["intent"]["action"]
    pipeline = result.get("pipeline_action_taken", "") or "none"
    response = result.get("response_text", "")
    mod_count = response.count("modify_strategy=")

    ok = True
    fails = []

    if action != expected_action:
        ok = False
        fails.append(f"action={action} expected={expected_action}")

    if max_mods is not None and mod_count > max_mods:
        ok = False
        fails.append(f"mods={mod_count} > max={max_mods}")

    if min_mods is not None and mod_count < min_mods:
        ok = False
        fails.append(f"mods={mod_count} < min={min_mods}")

    if "No elements found" in response:
        ok = False
        fails.append("No elements found")

    status = "PASS" if ok else "FAIL"
    detail = f" ({', '.join(fails)})" if fails else ""
    print(f"{status}{detail} [{elapsed:.0f}ms] {description}: {query}")
    print(f"  -> action={action}, pipeline={pipeline}, mods={mod_count}")
    results.append(ok)

# === MULTI-TURN TEST SEQUENCE ===
# Uses a SINGLE coordinator instance to test conversation memory

print("=" * 60)
print("MULTI-TURN HITL VERIFICATION")
print("=" * 60)
print()

# Turn 1: Change face to blur
test("Change the face to blur",
     expected_action="modify_strategy", max_mods=1, min_mods=1,
     description="T1: Face -> blur")

# Turn 2: Change black boxes (solid_overlay values) to blur
test("Can you change the black box to blur?",
     expected_action="modify_strategy", max_mods=5, min_mods=2,
     description="T2: Black boxes -> blur (values only)")

# Turn 3: Change it BACK to black box (multi-turn reference)
test("Change it back to black box",
     expected_action="modify_strategy", max_mods=5, min_mods=1,
     description="T3: Change back (should target same elements as T2)")

# Turn 4: Query
test("What risks were detected?",
     expected_action="query",
     description="T4: Query")

# Turn 5: Approve
test("approve",
     expected_action="approve",
     description="T5: Approve")

print()
print("=" * 60)
passed = sum(results)
total = len(results)
print(f"FINAL SCORE: {passed}/{total}")
if passed == total:
    print("ALL TESTS PASSED")
else:
    print(f"FAILED: {total - passed} test(s)")
print("=" * 60)
