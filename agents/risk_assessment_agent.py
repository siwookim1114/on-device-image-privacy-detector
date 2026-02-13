import time
import json
from PIL import Image
from typing import List, Dict, Optional
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

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
    ReclassifyAssessmentTool,
    SplitAssessmentTool,
    GetCurrentAssessmentsTool,
    ValidateAssessmentsTool,
    FinalizeReviewTool
)


class RiskAssessmentAgent:
    """
    Agent 2: Two-Phase Risk Assessment Agent

    Phase 1: Deterministic tool-based assessment (fast, <1 second)
        - Runs face/text/object risk tools in fixed order
        - Spatial analysis, escalation, filtering, validation

    Phase 2: Agentic VLM review with tool calling 
        - VLM sees annotated image + Phase 1 assessments
        - Tool-calling agent dynamically calls tools to reclassify, split, validate
        - Tools modify assessments in-place
    """

    def __init__(
        self,
        config,
        privacy_profile: Optional[PrivacyProfile] = None,
        reasoning_mode: str = "balanced"
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
        """
        self.config = config
        self.privacy_profile = privacy_profile if privacy_profile else PrivacyProfile()
        self.reasoning_mode = reasoning_mode
        self.vlm_model = self._get_vlm_model()

        # VLM for Phase 2 agentic review
        self.vlm = VisionLLM(
            model=self.vlm_model,
            base_url="http://localhost:11434"
        )

        # Phase 1 deterministic tools
        self.tools = self._get_tools()

        print(f"\n[RiskAssessmentAgent] Initialized")
        print(f"  Phase 1: Deterministic tool-based assessment")
        print(f"  Phase 2: Agentic VLM review (model: {self.vlm_model})")
        print(f"  Reasoning Mode: {self.reasoning_mode}")
        print(f"  Privacy Mode: {self.privacy_profile.default_mode}")
        print(f"  Phase 1 Tools: {len(self.tools)}")
        for t in self.tools:
            print(f"    - {t.name}")

    def _get_vlm_model(self) -> str:
        """Select VLM model based on reasoning mode."""
        mode_to_model = {
            "fast": "llama3.2-vision:11b",
            "balanced": "qwen3-vl:30b-a3b-thinking",
            "thorough": "qwen3-vl:30b-a3b-thinking"
        }
        return mode_to_model.get(self.reasoning_mode, "llama3.2-vision:11b")

    def _get_tools(self, assessments: Optional[List[Dict]] = None) -> List:
        """
        Initialize Phase 1 and Phase 2 tool instances.

        Phase 1 tools are always created (deterministic assessment).
        Phase 2 tools are only created when assessments are provided,
        since they need a mutable reference to modify in-place.

        Args:
            assessments: Mutable list of assessment dicts from Phase 1 (modified in-place).
                         None when called from __init__ (Phase 1 only).

        Returns:
            List of BaseTool instances
        """
        # Phase 1: Deterministic assessment tools (always available)
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
            tools.extend([
                ReclassifyAssessmentTool(assessments=assessments, config=self.config),
                SplitAssessmentTool(assessments=assessments, config=self.config),
                GetCurrentAssessmentsTool(assessments=assessments),
                ValidateAssessmentsTool(assessments=assessments),
                FinalizeReviewTool(assessments=assessments),
            ])

        return tools

    # ==================== Main Pipeline ====================

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
            assessments = self._vlm_agent_review(assessments, annotated_image)

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
        annotated_image: Image.Image
    ) -> AgentExecutor:
        """
        Build the Phase 2 tool-calling agent with multimodal prompt.

        Creates:
        1. Phase 2 review tools (from tools.py) that modify assessments in-place
        2. Multimodal prompt with annotated image + Phase 1 assessment summary
        3. Tool-calling agent + AgentExecutor

        Args:
            assessments: Phase 1 assessment dicts (tools will modify these in-place)
            annotated_image: Annotated image with bounding boxes

        Returns:
            AgentExecutor with tools bound
        """
        # Create all tools with assessments bound, then extract Phase 2 tools
        # Phase 1 tools = indices 0-7, Phase 2 tools = indices 8+
        all_tools = self._get_tools(assessments)
        phase2_tools = all_tools[len(self.tools):]  # Everything after Phase 1

        # Encode image for multimodal prompt
        image_b64 = self.vlm._image_to_base64(annotated_image)

        # Determine max iterations based on reasoning mode
        max_iters = 5 if self.reasoning_mode == "balanced" else 10

        # Build multimodal prompt:
        #   - System: /no_think + guidelines + Phase 1 assessments (via {input})
        #   - HumanMessage: annotated image (concrete, multimodal)
        #   - agent_scratchpad: tool call history (managed by AgentExecutor)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
                "/no_think\n"
                "You are a visual privacy risk assessment reviewer.\n\n"
                "You are looking at an annotated image with colored bounding boxes:\n"
                "- RED boxes: detected faces\n"
                "- GREEN boxes: detected text regions\n"
                "- BLUE boxes: detected objects\n\n"
                "Phase 1 (deterministic tools) produced these risk assessments:\n"
                "{input}\n\n"
                "Your job: VISUALLY VERIFY each assessment against the image.\n\n"
                "Available tools:\n"
                "- reclassify_assessment: Change severity if visual evidence contradicts Phase 1\n"
                "- split_assessment: Split a text block that contains both label and value\n"
                "- get_current_assessments: See updated state after changes\n"
                "- validate_assessments: Check consistency after reclassifications\n"
                "- finalize_review: Call when you are done reviewing\n\n"
                "Rules:\n"
                "- Only reclassify if you can clearly SEE visual evidence in the image\n"
                "- Faces without consent should remain critical\n"
                "- Text labels without actual values (e.g., 'Password:') should be low severity\n"
                "- Text with actual sensitive values (e.g., '4821') should be critical/high\n"
                "- After making changes, call validate_assessments\n"
                "- When satisfied with all assessments, call finalize_review"
            ),
            HumanMessage(content=[
                {"type": "text", "text": (
                    "Here is the annotated image. Review each assessment against "
                    "what you see and use tools to make corrections. "
                    "Call finalize_review when done."
                )},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"}
            ]),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create tool calling agent (native structured tool calls, no text parsing)
        agent = create_tool_calling_agent(
            llm=self.vlm.llm,
            tools=phase2_tools,
            prompt=prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=phase2_tools,
            verbose=True,
            max_iterations=max_iters,
            return_intermediate_steps=True
        )

        print(f"  Phase 2 agent built:")
        print(f"    LLM: {self.vlm_model}")
        print(f"    Max iterations: {max_iters}")
        print(f"    Tools: {[t.name for t in phase2_tools]}")

        return agent_executor
    

    def _build_assessment_summary(self, assessments: List[Dict]) -> str:
        """Format Phase 1 assessments as text for the VLM review prompt."""
        lines = []
        for i, a in enumerate(assessments):
            bbox_raw = a.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox_raw, dict):
                bbox_display = f"[{bbox_raw.get('x', 0)}, {bbox_raw.get('y', 0)}, {bbox_raw.get('width', 0)}, {bbox_raw.get('height', 0)}]"
            else:
                bbox_display = str(bbox_raw)

            lines.append(
                f"[{i}] {a.get('element_type', '?')} | "
                f"{a.get('element_description', '?')} | "
                f"severity={a.get('severity', '?')} | "
                f"protection={a.get('requires_protection', False)} | "
                f"bbox={bbox_display}"
            )
        return "\n".join(lines)

    def _vlm_agent_review(
        self,
        assessments: List[Dict],
        annotated_image: Image.Image
    ) -> List[Dict]:
        """
        Phase 2: Run the agentic VLM review.

        Builds the tool-calling agent via _build_agent, runs it, and returns
        the modified assessments. On any error, returns Phase 1 assessments unchanged.
        """
        print(f"\n{'-'*60}")
        print(f"Phase 2: Agentic VLM Review")
        print(f"{'-'*60}")

        try:
            # Build Phase 2 agent
            agent_executor = self._build_agent(assessments, annotated_image)

            # Run the agent (assessments are modified in-place by tools)
            print(f"  Starting VLM agent review ({len(assessments)} assessments)...")
            assessment_summary = self._build_assessment_summary(assessments)
            result = agent_executor.invoke({"input": assessment_summary})

            # Log what the agent did
            steps = result.get("intermediate_steps", [])
            tool_calls = [action.tool for action, _ in steps]
            print(f"  VLM agent made {len(steps)} tool calls: {tool_calls}")
            print(f"  Phase 2 complete: {len(assessments)} assessments after review")

            return assessments  # Modified in-place by tools

        except Exception as e:
            print(f"  VLM agent review failed: {e}")
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
