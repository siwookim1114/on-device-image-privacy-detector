# Re-export all tool classes so existing `from agents.tools import X` still works.

from agents.tools.detection_tools import (
    FaceDetectionTool,
    TextDetectionTool,
    ObjectDetectionTool,
)
from agents.tools.risk_tools import (
    FaceRiskAssessmentTool,
    TextRiskAssessmentTool,
    ObjectRiskAssessmentTool,
    SpatialRelationshipTool,
    ConsentInferenceTool,
    RiskEscalationTool,
    FalsePositiveFilterTool,
    ConsistencyValidationTool,
    ReclassifyAssessmentTool,
    BatchReclassifyTool,
    SplitAssessmentTool,
    GetCurrentAssessmentsTool,
    ValidateAssessmentsTool,
)
from agents.tools.strategy_tools import (
    ModifyStrategyTool,
    BatchModifyStrategiesTool,
    GetCurrentStrategiesTool,
    FinalizeStrategiesTool,
)
from agents.tools.execution_tools import (
    PatchRegionTool,
    AddProtectionTool,
    GetProtectionStatusTool,
    FinalizeVerificationTool,
)
