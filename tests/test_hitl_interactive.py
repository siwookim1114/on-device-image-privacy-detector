"""
Interactive HITL test for the Coordinator Agent.

Usage:
    conda run -n lab_env python tests/test_hitl_interactive.py data/test_images/sample3.png
    conda run -n lab_env python tests/test_hitl_interactive.py data/test_images/sample3.png --demo

Commands:
    why was this face blurred?
    use avatar for all faces
    make text protection stronger
    ignore the laptop screen
    don't protect face_001
    show detections
    undo
    quit
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEMO_SCRIPT = [
    "Show me what was detected",
    "Why was this face blurred?",
    "Use avatar replacement for all faces",
    "Make the text protection stronger",
    "Ignore the laptop screen",
    "Undo",
    "Show current strategies",
]

USE_COLOR = sys.stdout.isatty()


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def _green(t): return _c(t, "32")
def _yellow(t): return _c(t, "33")
def _red(t): return _c(t, "31")
def _blue(t): return _c(t, "34")
def _bold(t): return _c(t, "1")


class InteractionLog:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = []

    def add(self, turn, user_msg, result, elapsed_ms):
        entry = {
            "turn": turn,
            "user_message": user_msg,
            "intent": result.get("intent"),
            "response": result.get("response_text", ""),
            "pipeline_action": result.get("pipeline_action_taken"),
            "elapsed_ms": round(elapsed_ms, 1),
            "timestamp": time.time(),
        }
        self.entries.append(entry)
        self._save()

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump({"interactions": self.entries}, f, indent=2, default=str)


def print_pipeline_summary(session):
    state = session._state
    ps = state.get("pipeline_state", {})

    detections = ps.get("detections")
    risk = ps.get("risk_result")
    strategy = ps.get("strategy_result")

    print(f"\n{_bold('Pipeline State:')}")

    if risk and hasattr(risk, "risk_assessments"):
        assessments = risk.risk_assessments
        print(f"  Elements: {len(assessments)}")
        for a in assessments[:15]:
            sev = a.severity.value if hasattr(a.severity, "value") else str(a.severity)
            desc = a.element_description[:40] if hasattr(a, "element_description") else "?"
            color_fn = _red if sev == "critical" else (_yellow if sev == "high" else _green)
            print(f"    {a.detection_id[:12]}  {a.element_type:6s}  [{color_fn(sev):>8s}]  {desc}")
        if len(assessments) > 15:
            print(f"    ... and {len(assessments) - 15} more")
    else:
        print("  No risk assessments loaded")

    if strategy and hasattr(strategy, "strategies"):
        protected = sum(1 for s in strategy.strategies
                       if s.recommended_method and s.recommended_method.value != "none")
        print(f"  Strategies: {len(strategy.strategies)} total, {_bold(str(protected))} protected")


def print_result(result, elapsed_ms):
    intent = result.get("intent", {})
    action = intent.get("action", "?") if isinstance(intent, dict) else "?"
    conf = intent.get("confidence", 0) if isinstance(intent, dict) else 0

    print(f"\n  {_blue('Intent:')} {action} (confidence={conf:.2f})")

    pipeline_action = result.get("pipeline_action_taken")
    if pipeline_action:
        print(f"  {_yellow('Pipeline:')} {pipeline_action}")

    response = result.get("response_text", "")
    if response:
        for line in response.split("\n"):
            print(f"  {_green('>')} {line}")

    disagreements = result.get("disagreements", [])
    if disagreements:
        print(f"  {_red('Disagreements:')} {len(disagreements)}")
        for d in disagreements[:3]:
            print(f"    {d.get('detection_id', '?')}: Phase1={d.get('phase1_value')} → Phase2={d.get('phase2_value')}")

    hitl = result.get("hitl_presentation")
    if hitl:
        print(f"  {_yellow('HITL Checkpoint:')} {hitl.get('checkpoint', '?')} (confidence={hitl.get('confidence', 0):.2f})")

    print(f"  {_bold('Time:')} {elapsed_ms:.0f}ms")


def run_interactive(image_path, fallback_only=True, output_dir="data/hitl_demo"):
    from agents.pipeline import PipelineOrchestrator, PipelineConfig
    from agents.coordinator.main import CoordinatorSession
    from agents.coordinator.adaptive_learning import PreferenceManager

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{_bold('='*60)}")
    print(f"{_bold('HITL Interactive Test — Coordinator Agent')}")
    print(f"{_bold('='*60)}")
    print(f"Image: {image_path}")
    print(f"Mode: {'Phase 1 only' if fallback_only else 'Full pipeline (VLM)'}")

    # Initialize pipeline
    print(f"\nLoading pipeline...")
    config = PipelineConfig(fallback_only=fallback_only)
    orc = PipelineOrchestrator(config=config)

    # Run initial pipeline
    print(f"Running initial pipeline on {Path(image_path).name}...")
    t0 = time.perf_counter()
    output = orc.run(image_path)
    init_time = (time.perf_counter() - t0) * 1000
    print(f"Pipeline complete: {init_time:.0f}ms, success={output.success}")

    if not output.success:
        print(_red(f"Pipeline failed: {output.error_message}"))
        orc.close()
        return

    # Create coordinator session
    from agents.coordinator.state import InnerPipelineState

    # Build a mock session record for the coordinator
    class MockSession:
        def __init__(self):
            self.session_id = "hitl_test"
            self.image_path = image_path
            self.config = {"mode": "hybrid"}
            self.detections = output.risk_analysis
            self.risk_result = output.risk_analysis
            self.strategy_result = None
            self.execution_report = output.execution_report
            self.protected_image_path = output.protected_image_path
            self.status = "completed"
            self.coordinator_history = []
            self.last_intent = None

    mock_session = MockSession()

    # Try to get strategy from the pipeline output
    try:
        strategy_path = Path(output.protected_image_path).parent / f"{Path(image_path).stem}_strategies.json"
        if strategy_path.exists():
            print(f"  Strategies loaded from {strategy_path.name}")
    except Exception:
        pass

    session = CoordinatorSession(
        session=mock_session,
        orc=orc,
        safety_kernel=None,  # Will use inline validation
        preference_manager=PreferenceManager(),
    )

    # Pre-populate with pipeline results
    session._state["pipeline_state"]["detections"] = output.risk_analysis
    session._state["pipeline_state"]["risk_result"] = output.risk_analysis
    session._state["pipeline_state"]["execution_report"] = output.execution_report
    session._state["pipeline_state"]["image_path"] = image_path

    print_pipeline_summary(session)

    log = InteractionLog(output_dir / "interaction_log.json")

    return session, orc, log


def repl(session, orc, log):
    turn = 0
    print(f"\n{_bold('Ready for HITL commands.')} Type 'quit' to exit, 'help' for commands.\n")

    while True:
        try:
            user_input = input(f"{_blue('You > ')}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "help":
            print("Commands: show detections, why was X blurred, use avatar/blur/pixelate,")
            print("          make stronger, ignore X, don't protect X, undo, quit")
            continue
        if user_input.lower() in ("show detections", "show", "status"):
            print_pipeline_summary(session)
            continue

        turn += 1
        t0 = time.perf_counter()

        try:
            result = session.handle_message(user_input)
        except Exception as e:
            result = {"response_text": f"Error: {e}", "intent": {"action": "error"}}

        elapsed = (time.perf_counter() - t0) * 1000

        print_result(result, elapsed)
        log.add(turn, user_input, result, elapsed)

    orc.close()
    print(f"\nInteraction log saved: {log.log_path}")


def run_demo(session, orc, log, pause=0.8):
    print(f"\n{_bold('Running scripted demo...')}\n")

    for i, cmd in enumerate(DEMO_SCRIPT):
        print(f"\n{_bold(f'[Demo {i+1}/{len(DEMO_SCRIPT)}]')}")
        print(f"{_blue('You > ')} {cmd}")

        t0 = time.perf_counter()
        try:
            result = session.handle_message(cmd)
        except Exception as e:
            result = {"response_text": f"Error: {e}", "intent": {"action": "error"}}
        elapsed = (time.perf_counter() - t0) * 1000

        print_result(result, elapsed)
        log.add(i + 1, cmd, result, elapsed)

        if pause > 0:
            time.sleep(pause)

    orc.close()
    print(f"\n{_bold('Demo complete.')} Log: {log.log_path}")


def main():
    parser = argparse.ArgumentParser(description="Interactive HITL test for Coordinator Agent")
    parser.add_argument("image", nargs="?", default="data/test_images/sample3.png",
                        help="Image to process")
    parser.add_argument("--demo", action="store_true", help="Run scripted demo instead of REPL")
    parser.add_argument("--demo-pause", type=float, default=0.8, help="Pause between demo commands (seconds)")
    parser.add_argument("--no-fallback", action="store_true", help="Enable full VLM pipeline")
    parser.add_argument("--output-dir", default="data/hitl_demo", help="Output directory")
    args = parser.parse_args()

    result = run_interactive(
        args.image,
        fallback_only=not args.no_fallback,
        output_dir=args.output_dir,
    )

    if result is None:
        return

    session, orc, log = result

    if args.demo:
        run_demo(session, orc, log, pause=args.demo_pause)
    else:
        repl(session, orc, log)


if __name__ == "__main__":
    main()
