import time
import json
from PIL import Image
from typing import List, Dict, Optional
import easyocr
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langgraph.errors import GraphRecursionError
from langchain_core.messages import HumanMessage, ToolMessage

from utils.models import (
    DetectionResults,
    RiskAssessment,
    RiskAnalysisResult,
    RiskLevel,
    RiskType,
    PrivacyProfile,
    BoundingBox
)
from agents.local_wrapper import VisionLLM
from utils.config import get_risk_color
from agents.tools import (
    FaceRiskAssessmentTool,
    TextRiskAssessmentTool,
    ObjectRiskAssessmentTool,
    SpatialRelationshipTool,
    ConsentInferenceTool,
    RiskEscalationTool,
    FalsePositiveFilterTool,
    ConsistencyValidationTool,
    BatchReclassifyTool,
    SplitAssessmentTool,
    GetCurrentAssessmentsTool,
    ValidateAssessmentsTool,
)


class RiskAssessmentAgent:
    """
    Agent 2: Two-Phase Risk Assessment Agent

    Phase 1: Deterministic tool-based assessment (fast, <1 second)
        - Runs face/text/object risk tools in fixed order
        - Spatial analysis, escalation, filtering, validation

    Phase 2: LangGraph agent review with VLM tool calling
        - VLM sees annotated image + Phase 1 assessments
        - LangGraph create_react_agent dynamically calls tools
        - Pre-model hook trims messages to prevent context overflow
        - Tools modify assessments in-place
    """

    def __init__(
        self,
        config,
        privacy_profile: Optional[PrivacyProfile] = None,
        reasoning_mode: str = "balanced",
        vlm_backend: str = "llama-cpp"
    ):
        """
        Initialize the Two-Phase Risk Assessment Agent.

        Args:
            config: System configuration
            privacy_profile: User privacy preferences
            reasoning_mode: "fast" | "balanced" | "thorough"
                - fast: Phase 1 only (no VLM review)
                - balanced: Phase 1 + VLM review (default)
                - thorough: Phase 1 + VLM review with more iterations
            vlm_backend: "mlx" | "llama-cpp" | "ollama"
                - mlx: Use vllm-mlx server (default, fastest on Apple Silicon)
                - llama-cpp: Use llama-server
                - ollama: Use Ollama server
        """
        self.config = config
        self.privacy_profile = privacy_profile if privacy_profile else PrivacyProfile()
        self.reasoning_mode = reasoning_mode
        self.vlm_backend = vlm_backend
        self.vlm_model = self._get_vlm_model()

        # Backend-specific config
        backend_config = {
            "llama-cpp": {"base_url": "http://localhost:8080"},
            "ollama": {"base_url": "http://localhost:11434"},
            "mlx": {"base_url": "http://localhost:8000"},
        }
        base_url = backend_config.get(vlm_backend, backend_config["mlx"])["base_url"]

        # VLM for Phase 2 agentic review
        self.vlm = VisionLLM(
            model=self.vlm_model,
            base_url=base_url,
            backend=vlm_backend
        )

        # Phase 1 deterministic tools
        self.tools = self._get_tools()

        print(f"\n[RiskAssessmentAgent] Initialized")
        print(f"  Phase 1: Deterministic tool-based assessment")
        print(f"  Phase 2: LangGraph agent review (model: {self.vlm_model}, backend: {self.vlm_backend})")
        print(f"  Reasoning Mode: {self.reasoning_mode}")
        print(f"  Privacy Mode: {self.privacy_profile.default_mode}")
        print(f"  Phase 1 Tools: {len(self.tools)}")
        for t in self.tools:
            print(f"    - {t.name}")

    def _get_vlm_model(self) -> str:
        """Select VLM model based on reasoning mode and backend."""
        if self.vlm_backend == "llama-cpp":
            # llama-server serves a single model; name must match what server reports
            mode_to_model = {
                "fast": "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf",
                "balanced": "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf",
                "thorough": "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf"
            }
        elif self.vlm_backend == "mlx":
            mode_to_model = {
                "fast": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
                "balanced": "./Qwen3-VL-30B-A3B-Thinking-4bit",
                "thorough": "./Qwen3-VL-30B-A3B-Thinking-4bit"
            }
        else:  # ollama
            mode_to_model = {
                "fast": "llama3.2-vision:11b",
                "balanced": "qwen3-vl:30b-a3b-thinking",
                "thorough": "qwen3-vl:30b-a3b-thinking"
            }
        return mode_to_model.get(self.reasoning_mode, mode_to_model["balanced"])

    def _get_tools(self, assessments: Optional[List[Dict]] = None, image_path: str = None) -> List:
        """
        Initialize Phase 1 and Phase 2 tool instances.

        Phase 1 tools are always created (deterministic assessment).
        Phase 2 tools are only created when assessments are provided,
        since they need a mutable reference to modify in-place.

        Args:
            assessments: Mutable list of assessment dicts from Phase 1 (modified in-place).
            image_path: Path to original image (for OCR re-crop in SplitAssessmentTool).

        Returns:
            List of BaseTool instances
        """
        # Phase 1: Deterministic assessment tools
        tools = [
            FaceRiskAssessmentTool(self.config, self.privacy_profile),
            TextRiskAssessmentTool(self.config, self.privacy_profile),
            ObjectRiskAssessmentTool(self.config, self.privacy_profile),
            SpatialRelationshipTool(),
            ConsentInferenceTool(),
            RiskEscalationTool(),
            FalsePositiveFilterTool(),
            ConsistencyValidationTool(),
        ]

        # Phase 2: VLM review tools (only when assessments list is available)
        if assessments is not None:
            # Initialize OCR reader for precise bbox splitting
            if not hasattr(self, '_ocr_reader'):
                try:
                    self._ocr_reader = easyocr.Reader(
                        ["en"],
                        gpu=self.config.system.device == "cuda",
                        verbose=False
                    )
                except Exception:
                    self._ocr_reader = None

            tools.extend([
                BatchReclassifyTool(assessments=assessments, config=self.config),
                SplitAssessmentTool(
                    assessments=assessments,
                    config=self.config,
                    ocr_reader=self._ocr_reader,
                    image_path=image_path
                ),
                GetCurrentAssessmentsTool(assessments=assessments),
                ValidateAssessmentsTool(assessments=assessments),
            ])

        return tools

    def run(
        self,
        detections: DetectionResults,
        annotated_image: Optional[Image.Image] = None
    ) -> RiskAnalysisResult:
        """
        Main risk assessment pipeline.

        Phase 1: Deterministic tool-based assessment (always runs)
        Phase 2: Agentic VLM review (skipped in fast mode or if no image)

        Args:
            detections: Detection results from DetectionAgent
            annotated_image: Annotated image with bounding boxes for VLM review

        Returns:
            RiskAnalysisResult with complete risk assessments
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Risk Assessment Agent - Starting")
        print(f"{'='*60}")

        # Early exit if nothing to assess
        if detections.total_detections == 0:
            print("  No detections found, skipping risk assessment")
            processing_time = (time.time() - start_time) * 1000
            result = RiskAnalysisResult(
                image_path=detections.image_path,
                risk_assessments=[],
                overall_risk_level=RiskLevel.LOW,
                faces_pending_identity=0,
                confirmed_risks=0,
                processimg_time_ms=processing_time
            )
            self._print_summary(result, processing_time)
            return result

        # Build image context
        image = Image.open(detections.image_path)
        image_width, image_height = image.size
        image_context = {
            "width": image_width,
            "height": image_height,
            "total_faces": len(detections.faces),
            "total_texts": len(detections.text_regions),
            "total_objects": len(detections.objects)
        }

        print(f"\nImage Context:")
        print(f"  Dimensions: {image_width}x{image_height}")
        print(f"  Total elements: {detections.total_detections}")
        print(f"    - Faces: {image_context['total_faces']}")
        print(f"    - Texts: {image_context['total_texts']}")
        print(f"    - Objects: {image_context['total_objects']}")

        # Phase 1: Deterministic tool-based assessment
        print(f"\n{'-'*60}")
        print(f"Phase 1: Deterministic Tool-Based Assessment")
        print(f"{'-'*60}")
        assessments = self._tool_based_assessment(detections, image_context)
        print(f"Phase 1 complete: {len(assessments)} assessments")

        # Phase 2: Agentic VLM review
        if self.reasoning_mode != "fast" and annotated_image is not None and assessments:
            assessments = self._vlm_agent_review(
                assessments, annotated_image, image_path=detections.image_path
            )

        # Build final result
        return self._build_result(assessments, detections.image_path, start_time)

    # Phase 1: Deterministic Assessment 

    def _tool_based_assessment(
        self,
        detections: DetectionResults,
        image_context: Dict
    ) -> List[Dict]:
        """
        Phase 1: Run tools in fixed order for deterministic assessment.

        Pipeline: Individual assessment -> Spatial analysis -> Escalation -> Filter -> Validate
        """
        assessments = []

        # Step 1: Individual assessments
        for face in detections.faces:
            face_dict = face.model_dump()
            face_dict.pop("embedding", None)
            if "attributes" in face_dict:
                face_dict["attributes"].pop("embedding", None)
            face_dict["image_width"] = image_context["width"]
            face_dict["image_height"] = image_context["height"]

            result = self.tools[0]._run(json.dumps(face_dict))  # FaceRiskAssessmentTool
            assessment = json.loads(result)
            if "error" not in assessment:
                assessments.append(assessment)

        for text in detections.text_regions:
            result = self.tools[1]._run(json.dumps(text.model_dump()))  # TextRiskAssessmentTool
            assessment = json.loads(result)
            if "error" not in assessment:
                assessments.append(assessment)

        for obj in detections.objects:
            result = self.tools[2]._run(json.dumps(obj.model_dump()))  # ObjectRiskAssessmentTool
            assessment = json.loads(result)
            if "error" not in assessment and not assessment.get("filtered", False):
                assessments.append(assessment)

        # Step 2: Spatial analysis (if multiple elements)
        if detections.total_detections >= 2:
            detections_dict = {
                "faces": [f.model_dump() for f in detections.faces],
                "texts": [t.model_dump() for t in detections.text_regions],
                "objects": [o.model_dump() for o in detections.objects]
            }
            spatial_result = self.tools[3]._run(json.dumps(detections_dict))  # SpatialRelationshipTool
            spatial_data = json.loads(spatial_result)

            escalations = spatial_data.get("escalations", [])
            if escalations:
                escalation_input = {"assessments": assessments, "escalations": escalations}
                esc_result = self.tools[5]._run(json.dumps(escalation_input))  # RiskEscalationTool
                assessments = json.loads(esc_result).get("assessments", assessments)

        # Step 3: Filter + validate
        filter_result = self.tools[6]._run(json.dumps({"assessments": assessments}))  # FalsePositiveFilterTool
        assessments = json.loads(filter_result).get("assessments", assessments)

        validation_result = self.tools[7]._run(json.dumps({"assessments": assessments}))  # ConsistencyValidationTool
        assessments = json.loads(validation_result).get("assessments", assessments)

        return assessments

    #  Phase 2: Agentic VLM Review 

    def _build_agent(
        self,
        assessments: List[Dict],
        annotated_image: Image.Image,
        image_path: str = None
    ):
        """
        Build the Phase 2 LangGraph agent with multimodal input.

        Uses langchain's create_agent with middleware:
        - MessageTrimMiddleware: trims old messages, keeps image + recent context
        - ModelCallLimitMiddleware: caps iterations to prevent runaway loops
        - Tools modify assessments in-place

        Args:
            assessments: Phase 1 assessment dicts (tools will modify these in-place)
            annotated_image: Annotated image with bounding boxes
            image_path: Path to original image (for OCR re-crop in split tool)

        Returns:
            Tuple of (compiled_agent, image_b64, max_iters)
        """
        # Create all tools with assessments bound, then extract Phase 2 tools
        all_tools = self._get_tools(assessments, image_path=image_path)
        phase2_tools = all_tools[len(self.tools):]

        # Resize image to reduce vision token usage
        max_dim = 768
        w, h = annotated_image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            annotated_image = annotated_image.resize((new_w, new_h), Image.LANCZOS)

        # Encode image for multimodal prompt
        image_b64 = self.vlm._image_to_base64(annotated_image)

        # Scale max iterations based on assessment count and reasoning mode
        if self.reasoning_mode == "thorough":
            max_iters = max(10, min(len(assessments), 25))
        else:  # balanced
            max_iters = max(10, min(len(assessments), 15))

        system_prompt = (
            "You are a visual privacy risk reviewer. Verify EVERY Phase 1 assessment against the image.\n\n"
            "WORKFLOW (follow strictly in this order):\n"
            "1. SPLIT composites FIRST: Items marked COMPOSITE(split) contain 'Label: Value' — "
            "call split_assessment for each one. After each split, indices shift.\n"
            "2. REFRESH: Call get_current_assessments to see updated indices after splits.\n"
            "3. RECLASSIFY: Call batch_reclassify ONCE with ALL severity corrections.\n"
            "4. VALIDATE: Call validate_assessments ONCE after all corrections.\n\n"
            "SEVERITY RULES:\n"
            "- Faces with consent=none → CRITICAL (NEVER downgrade)\n"
            "- Label-only text (no real data after colon): '#:', 'Password:', 'Account:' → LOW\n"
            "- Actual sensitive values (numbers, SSNs, PINs) → CRITICAL\n"
            "- COMPOSITE text ('PIN: 4821', 'Bank Account: 8765') → SPLIT, not reclassify\n"
            "- Devices (laptop, phone): LOW unless sensitive content visible on screen\n\n"
            "CRITICAL RULE — SPLIT vs RECLASSIFY:\n"
            "- If text has 'Label: ActualData' (data after colon) → SPLIT into label(low) + value(critical)\n"
            "- If text is label-only (nothing meaningful after colon) → reclassify to LOW\n"
            "- Example: 'PIN: 4821' → SPLIT. '#:' → reclassify LOW.\n"
            "- Do NOT reclassify composites to LOW — that loses the sensitive value.\n\n"
            "IMPORTANT:\n"
            "- Items marked COMPOSITE(split) MUST be split, not reclassified.\n"
            "- Items already starting with 'Text label:' or 'Text value:' are already split — skip.\n"
            "- After split_assessment, indices shift. Use new_indices from the response.\n"
            "- Be concise between tool calls."
        )

        # Middleware: trim old messages to prevent context overflow.
        # Keeps the first message (image + assessments) and last 12 messages
        # (6 tool call/result pairs). The model only sees trimmed messages;
        # state retains the full history.
        class MessageTrimMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                messages = request.messages
                if len(messages) > 14:
                    trimmed = [messages[0]] + messages[-12:]
                    return handler(request.override(messages=trimmed))
                return handler(request)

        agent = create_agent(
            model=self.vlm.llm,
            tools=phase2_tools,
            system_prompt=system_prompt,
            middleware=[
                MessageTrimMiddleware(),
                ModelCallLimitMiddleware(run_limit=max_iters),
            ],
        )

        print(f"  Phase 2 LangGraph agent built:")
        print(f"    LLM: {self.vlm_model}")
        print(f"    Max iterations: {max_iters}")
        print(f"    Tools: {[t.name for t in phase2_tools]}")

        return agent, image_b64, max_iters
    

    def _build_assessment_summary(self, assessments: List[Dict]) -> str:
        """Format Phase 1 assessments as text for the VLM review prompt.

        Includes consent_status for faces and contains_screen for objects
        so the VLM has enough context to make correct severity decisions.
        """
        lines = []
        for i, a in enumerate(assessments):
            bbox_raw = a.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox_raw, dict):
                bbox_display = f"[{bbox_raw.get('x', 0)}, {bbox_raw.get('y', 0)}, {bbox_raw.get('width', 0)}, {bbox_raw.get('height', 0)}]"
            else:
                bbox_display = str(bbox_raw)

            parts = [
                f"[{i}] {a.get('element_type', '?')}",
                a.get('element_description', '?'),
                f"severity={a.get('severity', '?')}",
            ]

            # Faces: include consent status so VLM knows not to downgrade
            if a.get('element_type') == 'face':
                consent = a.get('consent_status', 'unknown')
                if hasattr(consent, 'value'):
                    consent = consent.value
                parts.append(f"consent={consent}")

            # Text: detect composite "Label: Value" patterns for VLM guidance
            if a.get('element_type') == 'text':
                import re as _re
                desc_text = a.get('element_description', '')
                raw = desc_text
                for prefix in ["Text: ", "Text label: ", "Text value: "]:
                    if raw.startswith(prefix):
                        raw = raw[len(prefix):]
                        break
                # Check "Label: Value" (with space) or "Label:Digits" (no space)
                has_colon_space = ": " in raw and len(raw.split(": ", 1)[1].strip()) >= 2
                has_colon_digits = bool(_re.search(r':\s*\d{2,}', raw))
                if has_colon_space or has_colon_digits:
                    parts.append("COMPOSITE(split)")
                elif raw.endswith(":") or raw.endswith(": ") or (": " in raw and len(raw.split(": ", 1)[1].strip()) < 2):
                    parts.append("label_only")

            # Objects: flag screen devices so VLM can escalate if needed
            if a.get('element_type') == 'object':
                has_screen = a.get('factors', {}).get('contains_screen',
                             a.get('contains_screen', False))
                if has_screen:
                    parts.append("screen_device")

            parts.append(f"bbox={bbox_display}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def _vlm_agent_review(
        self,
        assessments: List[Dict],
        annotated_image: Image.Image,
        image_path: str = None
    ) -> List[Dict]:
        """
        Phase 2: Run the LangGraph agent review.

        Builds the agent via _build_agent, invokes it with the image and
        assessment summary as a multimodal HumanMessage. The agent calls
        tools that modify assessments in-place.
        """
        print(f"\n{'-'*60}")
        print(f"Phase 2: LangGraph Agent Review")
        print(f"{'-'*60}")

        try:
            # Build Phase 2 agent
            agent, image_b64, max_iters = self._build_agent(
                assessments, annotated_image, image_path=image_path
            )

            # Build input message with image + assessment summary
            assessment_summary = self._build_assessment_summary(assessments)
            input_message = HumanMessage(content=[
                {"type": "text", "text": f"Assessments to review:\n{assessment_summary}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ])

            print(f"  Starting VLM agent review ({len(assessments)} assessments)...")

            # ModelCallLimitMiddleware handles graceful stop.
            # recursion_limit is a hard safety net above the middleware limit.
            result = agent.invoke(
                {"messages": [input_message]},
                config={"recursion_limit": 2 * max_iters + 5},
            )

            # Extract tool calls from the full message history
            tool_calls = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(tc["name"])

            print(f"  VLM agent made {len(tool_calls)} tool calls: {tool_calls}")
            print(f"  Phase 2 complete: {len(assessments)} assessments after review")

            return assessments  # Modified in-place by tools

        except GraphRecursionError:
            print(f"  VLM agent hit max iterations ({max_iters})")
            print(f"  Returning assessments as modified so far")
            return assessments

        except Exception as e:
            print(f"  VLM agent review failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Keeping Phase 1 results unchanged")
            return assessments

    # ==================== Result Building ====================

    def _build_result(
        self,
        assessment_dicts: List[Dict],
        image_path: str,
        start_time: float
    ) -> RiskAnalysisResult:
        """Convert assessment dicts into a RiskAnalysisResult."""
        final_assessments = []

        for d in assessment_dicts:
            try:
                # Handle bbox as both list and dict
                bbox_raw = d.get("bbox", [0, 0, 0, 0])
                if isinstance(bbox_raw, dict):
                    bbox = BoundingBox(**bbox_raw)
                elif isinstance(bbox_raw, list):
                    bbox = BoundingBox(x=bbox_raw[0], y=bbox_raw[1], width=bbox_raw[2], height=bbox_raw[3])
                else:
                    bbox = BoundingBox(x=0, y=0, width=0, height=0)

                risk_assessment = RiskAssessment(
                    detection_id=d.get("detection_id", "unknown"),
                    element_type=d.get("element_type", "unknown"),
                    element_description=d.get("element_description", "Unknown"),
                    risk_type=RiskType(d.get("risk_type", "context_exposure")),
                    severity=RiskLevel(d.get("severity", "low")),
                    color_code=d.get("color_code", "#808080"),
                    reasoning=d.get("reasoning", d.get("factors", {}).get("escalation_reason", "Tool-based assessment")),
                    user_sensitivity_applied=d.get("user_sensitivity_applied", "medium"),
                    bbox=bbox,
                    requires_protection=d.get("requires_protection", False),
                    legal_requirement=d.get("legal_requirement", False),
                    person_id=d.get("person_id"),
                    person_label=d.get("person_label"),
                    classification=d.get("classification"),
                    consent_status=d.get("consent_status"),
                    consent_confidence=d.get("consent_confidence", 0.0)
                )
                final_assessments.append(risk_assessment)
            except Exception as e:
                print(f"  Skipping malformed assessment: {e}")

        overall_risk = self._calculate_overall_risk(final_assessments)

        faces_pending_identity = sum(
            1 for a in final_assessments
            if a.element_type == "face" and a.consent_status == "unclear"
        )
        confirmed_risks = sum(1 for a in final_assessments if a.requires_protection)
        processing_time = (time.time() - start_time) * 1000

        result = RiskAnalysisResult(
            image_path=image_path,
            risk_assessments=final_assessments,
            overall_risk_level=overall_risk,
            faces_pending_identity=faces_pending_identity,
            confirmed_risks=confirmed_risks,
            processimg_time_ms=processing_time
        )

        self._print_summary(result, processing_time)
        return result

    def _calculate_overall_risk(self, assessments: List[RiskAssessment]) -> RiskLevel:
        """Calculate overall risk level for the image."""
        if not assessments:
            return RiskLevel.LOW

        critical_count = sum(1 for a in assessments if a.severity == RiskLevel.CRITICAL)
        high_count = sum(1 for a in assessments if a.severity == RiskLevel.HIGH)
        medium_count = sum(1 for a in assessments if a.severity == RiskLevel.MEDIUM)

        if critical_count > 0:
            return RiskLevel.CRITICAL
        elif high_count >= 2:
            return RiskLevel.CRITICAL
        elif high_count >= 1:
            return RiskLevel.HIGH
        elif medium_count >= 3:
            return RiskLevel.HIGH
        elif medium_count >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _print_summary(self, result: RiskAnalysisResult, processing_time: float):
        """Print result summary."""
        print(f"\n{'='*60}")
        print(f"Risk Assessment Complete")
        print(f"{'='*60}")
        print(f"  Processing time: {processing_time:.2f}ms")
        print(f"  Overall risk: {result.overall_risk_level.value.upper()}")
        print(f"  Total assessments: {len(result.risk_assessments)}")
        print(f"    - Critical: {len(result.get_critical_risks())}")
        print(f"    - High: {len(result.get_high_risks())}")
        print(f"    - Medium: {len(result.get_by_serverity(RiskLevel.MEDIUM))}")
        print(f"    - Low: {len(result.get_by_serverity(RiskLevel.LOW))}")
        print(f"  Requires protection: {result.confirmed_risks}")
        print(f"  Faces pending identity: {result.faces_pending_identity}")
        print(f"{'='*60}\n")
