"""Quick multi-turn intent classification test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.text_llm import TextLLM
from agents.coordinator.intent_classifier import IntentClassification, build_intent_llm_prompt

llm = TextLLM("backend-engines/models/Qwen3-1.7B-Q4_K_M.gguf")
schema = IntentClassification.model_json_schema()
sys_p, _ = build_intent_llm_prompt("test")

user_msg = (
    "Pipeline context:\n  stage: execution | elements: 11 | critical: True\n\n"
    "Recent conversation:\n"
    "User: Can you change the protected black box for sensitive values to blur?\n"
    "Assistant: Applied 4 modifications: text values changed from solid_overlay to blur.\n\n"
    "Last action: Changed 4 element(s) from solid_overlay to blur\n\n"
    'User message: "Can you change it back to black box?"'
)

raw = llm.call(sys_p, user_msg + " /no_think", json_schema=schema)
ic = IntentClassification.model_validate_json(raw)

print("action=" + str(ic.action))
print("types=" + str(ic.target_element_types))
print("method=" + str(ic.method_specified))
print("constraints=" + str(ic.extracted_constraints))
print("confidence=" + str(ic.confidence))

v = "PASS" if ic.action == "modify_strategy" else "FAIL"
print("VERDICT: " + v)
