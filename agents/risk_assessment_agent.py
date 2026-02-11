import time
import json
import re
from PIL import Image
from typing import List, Dict, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

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
from agents.tools import (
    FaceRiskAssessmentTool,
    TextRiskAssessmentTool,
    ObjectRiskAssessmentTool,
    SpatialRelationshipTool,
    ConsentInferenceTool,
    RiskEscalationTool,
    FalsePositiveFilterTool,
    ConsistencyValidationTool
)

class RiskAssessmentAgent:
    """
    Agent 2: ReAct-Based Risk Assessment Agent

    Uses dynamic tool selection via ReAct (Reasoning + Acting) pattern.
    The agent autonomously decides which tools to use and when.
    """

    def __init__(
        self,
        config,
        privacy_profile: Optional[PrivacyProfile] = None,
        reasoning_mode: str = "balanced"
    ):
        """
        Initialize the ReAct-based Risk Assessment Agent.

        Args:
            config: System configuration
            privacy_profile: User privacy preferences
            reasoning_mode: "fast" | "balanced" | "thorough"
                - fast: VLM only for uncertain cases
                - balanced: VLM for HIGH/CRITICAL risks (default)
                - thorough: VLM for all elements
        """
        self.config = config
        self.privacy_profile = privacy_profile if privacy_profile else PrivacyProfile()
        self.reasoning_mode = reasoning_mode
        self.vlm_model = self._get_vlm_model()

        # Initialize VLM
        self.vlm = VisionLLM(
            model=self.vlm_model,
            base_url="http://localhost:11434"
        )

        # Get all tools
        self.tools = self._get_tools()

        # Create ReAct agent
        self.agent_executor = self._build_agent()

        print(f"\n[RiskAssessmentAgent] Initialized as ReAct Agent")
        print(f"  VLM Model: {self.vlm_model}")
        print(f"  Reasoning Mode: {self.reasoning_mode}")
        print(f"  Privacy Mode: {self.privacy_profile.default_mode}")
        print(f"  Available Tools: {len(self.tools)}")
        for tool in self.tools:
            print(f"    - {tool.name}")

    def _get_vlm_model(self) -> str:
        """Select VLM model based on reasoning mode."""
        mode_to_model = {
            "fast": "llama3.2-vision:11b",
            "balanced": "qwen3-vl:30b-a3b-thinking",
            "thorough": "llama3.2-vision:11b"
        }
        return mode_to_model.get(self.reasoning_mode, "llama3.2-vision:11b")

    def _get_tools(self) -> List:
        """Initialize all tool instances (already LangChain BaseTool compatible)."""
        return [
            FaceRiskAssessmentTool(self.config, self.privacy_profile),
            TextRiskAssessmentTool(self.config, self.privacy_profile),
            ObjectRiskAssessmentTool(self.config, self.privacy_profile),
            SpatialRelationshipTool(),
            ConsentInferenceTool(),
            RiskEscalationTool(),
            FalsePositiveFilterTool(),
            ConsistencyValidationTool()
        ]

    def _build_agent(self) -> AgentExecutor:
        """Create ReAct agent with LangChain."""

        # FIX #1: ReAct prompt must have NO leading whitespace.
        # LangChain's ReAct parser expects "Thought:", "Action:", "Action Input:",
        # "Observation:", "Final Answer:" at the START of a line.
        template = (
            "You are a privacy risk assessment expert AI agent.\n"
            "Your task is to analyze detected elements (faces, text, objects) in an image and assess privacy risks using the available tools.\n"
            "\n"
            "Available tools:\n"
            "{tools}\n"
            "\n"
            "Tool names: {tool_names}\n"
            "\n"
            "Use the following format:\n"
            "\n"
            "Thought: Consider what needs to be done next\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action (must be valid JSON string)\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now have completed the risk assessment\n"
            "Final Answer: a short summary of the completed risk assessment\n"
            "\n"
            "Current Task:\n"
            "{input}\n"
            "\n"
            "Current Detection Data:\n"
            "{detection_summary}\n"
            "\n"
            "Begin! Remember to:\n"
            "1. Assess each detected face, text, and object using the appropriate risk assessment tool\n"
            "2. Analyze spatial relationships if multiple elements exist\n"
            "3. Apply escalations if spatial risks found\n"
            "4. Filter false positives\n"
            "5. Validate consistency\n"
            "\n"
            "{agent_scratchpad}"
        )

        prompt = PromptTemplate.from_template(template)

        # Create agent
        agent = create_react_agent(
            llm=self.vlm.llm,
            tools=self.tools,
            prompt=prompt
        )

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=20,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        return agent_executor

    def run(
        self,
        detections: DetectionResults,
        annotated_image: Optional[Image.Image] = None  # Reserved for future VLM image reasoning
    ) -> RiskAnalysisResult:
        """
        Main risk assessment pipeline using ReAct agent.

        Args:
            detections: Detection results from DetectionAgent
            annotated_image: Optional annotated image for VLM reasoning

        Returns:
            RiskAnalysisResult with complete risk assessments
        """
        _ = annotated_image  # Stored for future multimodal reasoning
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"ReAct Risk Assessment Agent - Starting")
        print(f"{'='*60}")

        # FIX #7: Early exit if nothing to assess
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

        # Prepare detection summary for agent
        detection_summary = self._prepare_detection_summary(detections, image_context)

        # Create task for agent (FIX #1: no leading whitespace)
        task = (
            f"Perform comprehensive privacy risk assessment on the detected elements.\n"
            f"Reasoning Mode: {self.reasoning_mode}\n"
            f"Privacy Profile: {self.privacy_profile.default_mode}\n\n"
            f"Follow this workflow:\n"
            f"1. Assess each detected face using assess_face_risk tool "
            f"(pass face data as JSON with image_width={image_width}, image_height={image_height})\n"
            f"2. Assess each detected text using assess_text_risk tool\n"
            f"3. Assess each detected object using assess_object_risk tool\n"
            f"4. If multiple elements detected, use analyze_spatial_relationship with all detections\n"
            f"5. Apply risk escalations if spatial risks found using apply_risk_escalations\n"
            f"6. Filter false positives using filter_false_positives\n"
            f"7. Validate consistency using validate_consistency\n\n"
            f"When finished, provide a Final Answer summarizing the risk assessment."
        )
        try:
            # Run ReAct agent
            print(f"\n{'-'*60}")
            print(f"Agent Execution Starting...")
            print(f"{'-'*60}\n")

            result = self.agent_executor.invoke({
                "input": task,
                "detection_summary": detection_summary
            })            

            # Extract assessments from intermediate_steps (reliable tool outputs)
            # instead of parsing the Final Answer (unreliable LLM-generated JSON)
            assessments = self._extract_from_intermediate_steps(result.get("intermediate_steps", []))

            # If no assessments extracted from steps, try parsing Final Answer
            if not assessments:
                print("  No assessments from intermediate steps, trying Final Answer...")
                assessments = self._try_parse_final_answer(result.get("output", ""))

            # Convert to RiskAnalysisResult
            risk_result = self._build_result(assessments, detections.image_path, start_time)
            return risk_result

        except Exception as e:
            print(f"\nAgent execution failed: {e}")
            print(f"Falling back to tool-based assessment...")

            # FIX #6: Fallback still uses tools directly
            return self._fallback_with_tools(detections, image_context, start_time)

    def _prepare_detection_summary(
        self,
        detections: DetectionResults,
        image_context: Dict
    ) -> str:
        """Prepare detection summary for agent context."""

        # FIX #2: Exclude embeddings from serialization (512-dim float arrays blow up context)
        face_dicts = []
        for f in detections.faces:
            d = f.model_dump()
            d.pop("embedding", None)
            if "attributes" in d:
                d["attributes"].pop("embedding", None)
            face_dicts.append(d)

        text_dicts = [t.model_dump() for t in detections.text_regions]
        object_dicts = [o.model_dump() for o in detections.objects]

        # No leading whitespace in the summary
        summary = (
            f"Image Dimensions: {image_context['width']}x{image_context['height']}\n\n"
            f"Detected Elements:\n"
            f"- Faces: {len(detections.faces)}\n"
            f"- Text Regions: {len(detections.text_regions)}\n"
            f"- Objects: {len(detections.objects)}\n\n"
            f"Face Data (JSON):\n{json.dumps(face_dicts, indent=2)}\n\n"
            f"Text Data (JSON):\n{json.dumps(text_dicts, indent=2)}\n\n"
            f"Object Data (JSON):\n{json.dumps(object_dicts, indent=2)}\n\n"
            f"Image Context (for consent inference):\n{json.dumps(image_context, indent=2)}"
        )

        return summary

    def _extract_from_intermediate_steps(self, steps: List) -> List[Dict]:
        """
        FIX #4: Extract assessment dicts from intermediate tool call results.

        Each step is a tuple of (AgentAction, tool_output_string).
        The tool outputs are JSON strings from our deterministic tools,
        which are far more reliable than the LLM's Final Answer.
        """
        assessments = []

        for action, output in steps:
            tool_name = action.tool

            try:
                result = json.loads(output) if isinstance(output, str) else output
            except (json.JSONDecodeError, TypeError):
                continue

            # Collect individual risk assessments
            if tool_name in ["assess_face_risk", "assess_text_risk", "assess_object_risk"]:
                if "error" not in result and not result.get("filtered", False):
                    assessments.append(result)

            # Spatial analysis results are consumed by the agent's next tool call
            # (apply_risk_escalations), so we don't need to store them here

            # If escalation tool was used, grab the updated assessments
            elif tool_name == "apply_risk_escalations":
                updated = result.get("assessments", [])
                if updated:
                    assessments = updated

            # If filter tool was used, grab the filtered assessments
            elif tool_name == "filter_false_positives":
                filtered = result.get("assessments", [])
                if filtered:
                    assessments = filtered

            # If consistency tool was used, grab the validated assessments
            elif tool_name == "validate_consistency":
                validated = result.get("assessments", [])
                if validated:
                    assessments = validated

        return assessments

    def _try_parse_final_answer(self, output: str) -> List[Dict]:
        """Try to extract assessment dicts from the LLM's Final Answer string."""
        # Try to find JSON in the output (LLM might wrap it in markdown ```json blocks)
        json_match = re.search(r'\{[\s\S]*"assessments"[\s\S]*\}', output)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return result.get("assessments", [])
            except json.JSONDecodeError:
                pass

        return []

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
                # FIX #5: Handle bbox as both list and dict
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

    def _fallback_with_tools(
        self,
        detections: DetectionResults,
        image_context: Dict,
        start_time: float
    ) -> RiskAnalysisResult:
        """
        FIX #6: Fallback runs tools directly in a fixed order.

        Still uses the same tools for accurate results,
        just without the LLM choosing the order.
        """
        print("\nRunning fallback tool-based assessment...")
        assessments = []

        # Phase 1: Individual assessments
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

        # Phase 2: Spatial analysis (if multiple elements)
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

        # Phase 3: Filter + validate
        filter_result = self.tools[6]._run(json.dumps({"assessments": assessments}))  # FalsePositiveFilterTool
        assessments = json.loads(filter_result).get("assessments", assessments)

        validation_result = self.tools[7]._run(json.dumps({"assessments": assessments}))  # ConsistencyValidationTool
        assessments = json.loads(validation_result).get("assessments", assessments)

        return self._build_result(assessments, detections.image_path, start_time)

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
