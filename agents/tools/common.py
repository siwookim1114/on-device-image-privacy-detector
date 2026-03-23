# Shared imports and utilities for all tool modules
import json
import re
import numpy as np
from typing import Any, List, Dict, Optional, ClassVar, Set, Tuple, Type, Union

from PIL import Image
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from utils.models import (
    RiskLevel,
    RiskType,
    PrivacyProfile,
    BoundingBox,
    RiskAssessment,
    FaceDetection,
    TextDetection,
    ObjectDetection,
    DetectionResults,
    ConsentStatus,
    PersonClassification,
    ReclassifyAssessmentInput,
    ReclassifyItem,
    BatchReclassifyInput,
    SplitPart,
    SplitAssessmentInput,
    ModifyStrategyInput,
    ModifyStrategyItem,
    BatchModifyStrategiesInput,
    ObfuscationMethod,
    PatchRegionInput,
    AddProtectionInput,
)
from utils.config import get_risk_color

# Valid obfuscation methods for validation
VALID_METHODS = {"blur", "pixelate", "solid_overlay", "inpaint", "avatar_replace", "generative_replace", "none"}

# Method strength ranking for safety guards
METHOD_STRENGTH = {
    "none": 0, "blur": 1, "pixelate": 2,
    "avatar_replace": 3, "inpaint": 3, "generative_replace": 3,
    "solid_overlay": 4,
}

# Screen device keywords for challenge detection
SCREEN_KEYWORDS = {"laptop", "tv", "cell phone", "monitor", "phone", "screen"}


def _parse_tool_input(raw_input) -> dict:
    """
    Parse tool input that may be a JSON string (fallback/ReAct) or a dict (tool calling API).
    Tool calling agents pass args as dicts; direct _run() calls pass JSON strings.
    """
    if isinstance(raw_input, dict):
        return raw_input
    if isinstance(raw_input, str):
        return json.loads(raw_input)
    return json.loads(str(raw_input))
