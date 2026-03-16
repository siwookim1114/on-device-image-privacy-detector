# On-Device Image Privacy-Aware Detector

An on-device, multi-agent orchestrated privacy-aware image analysis and protection framework. The system automatically detects sensitive visual elements (faces, text, screen content), assesses privacy risks, determines consent through identity learning, recommends context-aware obfuscation strategies, and applies precise pixel-level protection — all running locally on Apple Silicon without cloud APIs.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Flow](#pipeline-flow)
- [Agent Deep Dive](#agent-deep-dive)
  - [Agent 1: Detection](#agent-1-detection)
  - [Agent 2: Risk Assessment](#agent-2-risk-assessment)
  - [Agent 2.5: Consent Identity](#agent-25-consent-identity)
  - [Agent 3: Strategy](#agent-3-strategy)
  - [Precision Segmentation (SAM)](#precision-segmentation-sam)
- [Two-Phase Agent Architecture](#two-phase-agent-architecture)
- [Comparison with ImagoObscura](#comparison-with-imagoobscura)
- [Data Models](#data-models)
- [Configuration](#configuration)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Overview

Sharing images online exposes privacy risks that most users overlook — bystander faces, visible PII on documents, screen content, and contextual information. Existing tools either require manual selection (Photoshop, Canva) or focus narrowly on faces only.

This framework provides **end-to-end automated privacy protection** through a multi-agent pipeline that:

1. **Detects** all privacy-sensitive elements (faces, text/PII, screen devices, objects)
2. **Assesses** risk severity using deterministic rules + VLM visual review
3. **Identifies** known individuals and applies consent-based policies
4. **Recommends** context-aware obfuscation strategies with ethical mode enforcement
5. **Segments** precise regions using SAM for natural-looking protection
6. **Applies** targeted obfuscation (blur, pixelate, solid overlay, inpaint, avatar replace)

### Key Design Principles

- **On-device processing** — All models run locally on Apple Silicon (Metal GPU). No data leaves the device.
- **Two-phase architecture** — Each agent uses fast deterministic rules (Phase 1, <1s) followed by optional VLM visual review (Phase 2) for accuracy.
- **Safety-first** — Multiple layers of protection: deterministic rules, VLM prompt guidance, and hard-coded tool guards ensure no sensitive element is left unprotected.
- **Consent-aware** — Learned identity database tracks who has given consent, adjusting protection automatically over time.
- **Element-type-aware** — Different detection, assessment, and obfuscation strategies for faces vs text vs screens vs objects.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AGENT 1: DETECTION                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ MTCNN +      │  │ EasyOCR      │  │ YOLOv8n                  │  │
│  │ FaceNet      │  │ Text/OCR     │  │ Object Detection         │  │
│  │ (512D embed) │  │ Detection    │  │ (screen device classify)  │  │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘  │
│         │                 │                        │                 │
│         ▼                 ▼                        ▼                 │
│    FaceDetection[]   TextDetection[]        ObjectDetection[]       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ DetectionResults + Annotated Image
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AGENT 2: RISK ASSESSMENT                                           │
│                                                                     │
│  Phase 1 (Deterministic, <1s):                                      │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────┐ ┌────────┐          │
│  │Face    │→│Text    │→│Object  │→│Spatial  │→│Consent │→ ...      │
│  │Risk    │ │Risk    │ │Risk    │ │Relation │ │Infer   │           │
│  └────────┘ └────────┘ └────────┘ └─────────┘ └────────┘          │
│                                                                     │
│  Phase 1.5 (Screen Verification):                                   │
│  ┌─────────────────────────────────────────────┐                   │
│  │ Crop each screen device → VLM: "ON or OFF?" │                   │
│  │ ON → escalate to MEDIUM | OFF → stay LOW    │                   │
│  └─────────────────────────────────────────────┘                   │
│  Phase 1.5b (Composite Auto-Split):                                 │
│  ┌─────────────────────────────────────────────┐                   │
│  │ "Bank Account: 8765 4321" → split into      │                   │
│  │   label ("Bank Account:") LOW                │                   │
│  │   value ("8765 4321") CRITICAL               │                   │
│  └─────────────────────────────────────────────┘                   │
│                                                                     │
│  Phase 2 (VLM Agentic Review, optional):                            │
│  ┌─────────────────────────────────────────────┐                   │
│  │ VLM sees annotated image + assessments       │                   │
│  │ Tool calling: reclassify, split, validate    │                   │
│  │ Middleware: MessageTrim + ModelCallLimit      │                   │
│  └─────────────────────────────────────────────┘                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ RiskAnalysisResult
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AGENT 2.5: CONSENT IDENTITY                                        │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ Face Embedding   │→│ MongoDB Match     │→│ Risk Adjustment   │   │
│  │ Comparison       │ │ (cosine sim >0.85)│ │ by Consent Status │   │
│  └─────────────────┘  └──────────────────┘  └──────────────────┘   │
│                                                                     │
│  Classifications: PRIMARY_SUBJECT | KNOWN_CONTACT | BYSTANDER       │
│  Consent Status:  EXPLICIT | ASSUMED | NONE | UNCLEAR               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ RiskAnalysisResult (+ identity fields)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AGENT 3: STRATEGY                                                  │
│                                                                     │
│  Phase 1 (Rule-Based Defaults, <100ms):                             │
│  ┌─────────────────────────────────────────────┐                   │
│  │ Face rules (consent × severity → method)     │                   │
│  │ Text rules (severity × requires_protection)  │                   │
│  │ Object rules (severity → method)             │                   │
│  │ Ethical mode enforcement                     │                   │
│  └─────────────────────────────────────────────┘                   │
│                                                                     │
│  Phase 2 (VLM Strategy Review, optional):                           │
│  ┌─────────────────────────────────────────────┐                   │
│  │ VLM reviews strategies against actual image  │                   │
│  │ Safety net: catches Agent 2 misclassifications│                  │
│  │ Tool guards: cannot weaken CRITICAL, cannot  │                   │
│  │   remove bystander protection, etc.          │                   │
│  └─────────────────────────────────────────────┘                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ StrategyRecommendations
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PRECISION SEGMENTATION (SAM)                                       │
│  ┌─────────────────────────────────────────────┐                   │
│  │ MobileSAM (~40MB, on-device)                 │                   │
│  │ Selective: only elements needing protection   │                   │
│  │ Faces → point prompt → precise face mask      │                   │
│  │ Screens → box+point prompt → panel mask        │                   │
│  │ Text → skip (bbox redaction is cleaner)       │                   │
│  └─────────────────────────────────────────────┘                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Precise masks + refined bboxes
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AGENT 4: PROTECTION APPLICATION [Planned]                          │
│  ┌─────────────────────────────────────────────┐                   │
│  │ Apply obfuscation using SAM masks:           │                   │
│  │   Faces → mask-based blur (natural contour)  │                   │
│  │   Text  → bbox solid overlay (clean redact)  │                   │
│  │   Screens → mask-based blur (screen only)    │                   │
│  └─────────────────────────────────────────────┘                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                        PROTECTED IMAGE
                    + Risk Map + JSON Reports
                    + Provenance Log (audit trail)
```

---

## Pipeline Flow

### End-to-End Data Flow

```
Image → Detections (bbox + embeddings + OCR text)
      → Risk Assessments (severity + risk type per element)
      → Identity Matching (consent status per face)
      → Strategy Recommendations (method + parameters per element)
      → Precise Segmentation (pixel masks for faces/screens)
      → Protected Image (obfuscation applied)
```

### Processing Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| **AUTO** | All protections applied automatically | Quick sharing |
| **HYBRID** | Critical items auto-protected, others need user approval | Recommended default |
| **MANUAL** | All items need user approval | Maximum control |

### Ethical Modes

| Mode | Allowed Methods | Constraint |
|------|----------------|------------|
| **STRICT** | blur, pixelate, solid_overlay | No deception — viewer knows content was hidden |
| **BALANCED** | + avatar_replace, inpaint | Context-preserving transformations |
| **CREATIVE** | + generative_replace | Full power, requires watermark disclosure |

---

## Agent Deep Dive

### Agent 1: Detection

**Purpose:** Detect all privacy-sensitive visual elements in the image.

**Tools:**
| Tool | Model | Output |
|------|-------|--------|
| FaceDetectionTool | MTCNN + FaceNet | Face bbox + 512D embedding + landmarks + clarity |
| TextDetectionTool | EasyOCR | Text bbox + content + privacy classification |
| ObjectDetectionTool | YOLOv8n | Object bbox + class + screen device flag |

**Text Privacy Classification:**
EasyOCR detects text, then a regex-based classifier identifies privacy-relevant patterns:
- `phone_number` — Phone number patterns
- `email` — Email addresses
- `ssn` — Social Security Numbers (XXX-XX-XXXX)
- `credit_card` — Credit card numbers
- `numeric_fragment` — 3-8 digit numbers (potential PINs, codes)
- `password_like` — Mixed alphanumeric + special chars
- `name` — Proper names (capitalized words)

**Screen Device Detection:**
YOLO objects with class `laptop`, `tv`, `cell phone`, or `remote` are flagged as `screen_device` with `contains_screen=True` for Phase 1.5 verification.

---

### Agent 2: Risk Assessment

**Purpose:** Assess privacy risk severity for each detected element using a two-phase architecture.

#### Phase 1: Deterministic Risk Pipeline (<1s)

Nine specialized tools run in sequence:

1. **FaceRiskAssessmentTool** — Face size, clarity, position → severity
2. **TextRiskAssessmentTool** — Text type classification → severity (SSN=CRITICAL, label=LOW)
3. **ObjectRiskAssessmentTool** — Object class → severity (screen devices start LOW)
4. **SpatialRelationshipTool** — Proximity analysis with adaptive threshold (20% of image diagonal, clamped [150, 400px])
5. **ConsentInferenceTool** — Basic consent status assignment
6. **RiskEscalationTool** — Context-based severity elevation (e.g., PII near face)
7. **FalsePositiveFilterTool** — Remove low-confidence detections
8. **ConsistencyValidationTool** — Cross-validation sanity checks

#### Phase 1.5a: Screen Device VLM Verification

For each screen device detected by YOLO:
1. Crop the screen region from the original full-resolution image
2. Send the crop to the VLM with a focused YES/NO question: *"Is a display screen with visible content facing the camera?"*
3. YES → escalate severity to MEDIUM | NO → stays LOW

This targeted VLM check is fast (~1-2s per device) and uses strict YES/NO parsing to avoid VLM hallucination.

#### Phase 1.5b: Composite Auto-Split

Text detections containing "Label: Value" patterns (e.g., "Bank Account: 8765 4321") are automatically split into separate assessments:
- **Label** ("Bank Account:") → LOW severity, no protection needed
- **Value** ("8765 4321") → inherits the critical severity of the value type

This ensures labels are never incorrectly protected while values always are.

#### Phase 2: VLM Agentic Review (Optional)

A LangChain ReAct agent with Qwen3-VL reviews the annotated image alongside all Phase 1 assessments. The VLM can:
- **Reclassify** assessment severity (with guards preventing dangerous downgrades)
- **Split** assessments into finer-grained parts
- **Validate** the complete assessment list

**Middleware:**
- `MessageTrimMiddleware` — Keeps first message (image) + last 10 messages to prevent context overflow
- `ModelCallLimitMiddleware` — Caps VLM iterations to prevent infinite loops

**Safety Guards (Phase 2 Tools):**
- Cannot downgrade CRITICAL text items
- Cannot reclassify screen devices (pre-verified by VLM crop)
- Escalation deduplication (one per element × relationship pair)

---

### Agent 2.5: Consent Identity

**Purpose:** Match detected faces against a consent database and adjust risk based on identity and consent history.

**Architecture:** Fully deterministic — no LLM involved. Uses FaceNet embedding cosine similarity against a MongoDB database.

**Face Classification Flow:**

```
Detected face embedding (512D)
        │
        ▼
  ┌─────────────────┐
  │ MongoDB Lookup   │ cosine similarity against all stored embeddings
  │ Threshold: 0.85  │
  └────────┬────────┘
           │
     ┌─────┴──────┐
     │             │
  Match          No Match
     │             │
     ▼             ▼
  Known Person   BYSTANDER
  (check consent   consent=NONE
   history)        severity=CRITICAL
     │
     ├─ PRIMARY_SUBJECT (self) → consent=EXPLICIT → severity=LOW
     ├─ KNOWN_CONTACT (>80% approval rate) → consent=ASSUMED → severity reduced
     └─ KNOWN_CONTACT (<80% approval) → consent=UNCLEAR → severity unchanged
```

**Consent Learning:**
- Tracks `times_appeared`, `times_approved`, `times_protected` per person
- `approval_rate = times_approved / times_appeared`
- After `min_appearances_for_trust` (default: 3) appearances with >80% approval rate, the person is upgraded to `ASSUMED` consent
- `risk_decay_factor` reduces over time with approvals (more trusted → less protection)

**Storage:** MongoDB with Fernet encryption for face embeddings. Max 5 embeddings per person.

---

### Agent 3: Strategy

**Purpose:** Recommend context-aware obfuscation strategies for each assessed risk.

#### Phase 1: Rule-Based Defaults (<100ms)

Maps each risk assessment to an obfuscation strategy using deterministic rule tables:

**Face Rules:**

| Consent | Severity | Method | Parameters |
|---------|----------|--------|------------|
| explicit (self) | any | none | — |
| assumed (known) | low | none | — |
| assumed (known) | medium+ | blur | kernel=15/25/35 |
| none (bystander) | low/medium | blur | kernel=25 |
| none (bystander) | high | blur | kernel=35 |
| none (bystander) | critical | pixelate | block=16 |

**Text Rules:**

| Severity | Method | Parameters |
|----------|--------|------------|
| low | none | — (labels, benign text) |
| medium | blur | kernel=25 |
| high | solid_overlay | color=black |
| critical | solid_overlay | color=black |

**Object Rules:**

| Severity | Method | Parameters |
|----------|--------|------------|
| low | none | — |
| medium | blur | kernel=25 |
| high+ | blur | kernel=35 |

#### Phase 2: VLM Strategy Review (Optional)

The VLM acts as a **safety net** for the previous agents. It reviews all strategies against the actual image and can modify recommendations via tool calling.

**Key VLM Instructions:**
- **Faces:** Trust consent status from data (determined by face-matching agent). Only verify method appropriateness (blur vs pixelate), not identity/consent.
- **Text — Labels vs Values:** Labels ("Social Security #:", "Bank Account:") are descriptors that do NOT need protection. Values ("238-49-6521") contain the actual sensitive data.
- **Screens:** Check if visible screen content is actually sensitive. Desktop backgrounds with no readable content → set to `none`.

**Hard Rules (enforced by tool guards):**
- Cannot weaken CRITICAL severity items
- Cannot weaken HIGH severity text items
- Cannot remove bystander face protection
- Cannot set `none` for items marked `[LOCKED]` (requires_protection=True)

**Soft Rules (VLM judgment):**
- LOW items usually stay at `none` unless VLM visually confirms misclassification
- May strengthen methods for MEDIUM+ items based on visual context

---

### Precision Segmentation (SAM)

**Purpose:** Replace rectangular bounding box obfuscation with precise pixel-level masks for natural-looking protection.

**Model:** MobileSAM (~40MB, on-device)

**Why SAM Runs After Agent 3 (Not After Agent 1):**

A critical design decision — SAM is placed *after* all analysis agents, not immediately after detection. This follows the framework's progressive narrowing philosophy:

```
Agent 1 detects 7 elements
  → Agent 2 assesses risk (6 are LOW)
    → Agent 2.5 identifies faces (1 is the user → consent=explicit)
      → Agent 3 assigns strategies (only 1 needs protection)
        → SAM segments 1 element (not 7)
```

Running SAM after Agent 1 would segment all 7 detections (~700ms), then discard 6 of those masks when downstream agents determine they don't need protection. This is exactly what ImagoObscura does — uniform SAM on everything. Our selective approach segments only what actually needs obfuscation.

No upstream agent benefits from SAM masks: Agent 2 classifies risk based on element type, text content, and spatial relationships (bboxes are sufficient). Agent 2.5 matches face embeddings (its own pipeline). Agent 3 chooses methods based on severity and consent. Only Agent 4 (execution) needs pixel-level masks to apply obfuscation.

**Element-Type-Aware Segmentation:**

| Element | SAM? | Prompt Strategy | Result |
|---------|------|----------------|--------|
| Faces | Yes | Point prompt at face bbox center | Face contour mask — natural blur without rectangular artifacts |
| Screens (TV/monitor) | Yes | Box prompt (YOLO bbox) + point at device center | Screen panel mask — precise display area |
| Screens (laptop) | Yes | Tight box (upper 60% of bbox) + point + largest-mask selection | Screen panel only — excludes keyboard/trackpad |
| Text | No | — | Bbox redaction is cleaner and more standard for text |
| Other objects | No | — | Typically method=none; bbox is fine when protected |

**SAM Prompt Strategies:**
- **Faces:** Single foreground point at face bbox center. SAM naturally segments the most prominent face contour at that location.
- **TV/Monitors:** Combined box + point prompt. The YOLO bbox constrains the region, the center point guides SAM to the display surface.
- **Laptops:** Tight box restricted to the upper 60% of the YOLO bbox (screen region), with a center point. Among candidate masks, the largest mask is selected (the screen panel is the largest flat surface in the constrained region). This prevents segmenting the keyboard/trackpad.

**Selective Processing:** SAM only runs on elements where `method != none`. User's own face (consent=explicit), labels (severity=LOW), and low-risk objects are skipped entirely. Masks are saved as `.npy` files for Agent 4.

---

## Two-Phase Agent Architecture

Every agent in the pipeline follows the same two-phase pattern:

```
Phase 1 (Deterministic)          Phase 2 (VLM Agentic Review)
┌───────────────────┐            ┌──────────────────────────┐
│ Rule-based tools   │            │ VLM sees image +         │
│ Fixed pipeline     │     ──►    │ Phase 1 results          │
│ <1 second          │            │ Tool calling to modify   │
│ Always runs        │            │ Safety guards prevent    │
│ Predictable output │            │   dangerous changes      │
└───────────────────┘            └──────────────────────────┘
```

**Why Two Phases?**
- **Phase 1 alone** handles 80%+ of cases correctly with zero latency
- **Phase 2** catches the edge cases Phase 1 misses (OCR errors, ambiguous content, visual context)
- If Phase 2 fails or times out, Phase 1 results are always safe defaults
- VLM tool guards ensure Phase 2 can never make things worse (cannot weaken critical protections)

**VLM Backend:** Qwen3-VL-30B-A3B-Instruct (Q4_K_M quantization) served via llama.cpp on port 8081 with Metal GPU acceleration and native tool calling (`--jinja` flag).

---

## Comparison with ImagoObscura

[ImagoObscura](https://arxiv.org/abs/2504.11521) is a benchmark privacy protection framework using an ensemble of four AI models. Our framework draws inspiration from their approach while making key architectural improvements:

### Pipeline Comparison

| Step | ImagoObscura | Our Framework |
|------|-------------|---------------|
| **Detection** | Florence 2 (open-vocab) | YOLO + MTCNN + FaceNet + EasyOCR (specialized per element type) |
| **Risk Assessment** | GPT-4o (cloud, single-pass) | Two-phase: deterministic tools + VLM review (on-device) |
| **Consent** | None | Agent 2.5: MongoDB identity database with consent learning |
| **Localization** | Florence 2 re-localization | Not needed — detections already have precise bboxes |
| **Segmentation** | SAM (full, all elements) | MobileSAM (selective: faces + screens only) |
| **Obfuscation** | Stable Diffusion | Direct mask-based blur/pixelate (faster, on-device) |
| **Infrastructure** | Cloud APIs (GPT-4o) | Fully on-device (llama.cpp + Metal GPU) |

### Key Improvements

#### 1. No Re-Localization Step Needed
ImagoObscura requires Florence 2 to re-locate elements after GPT-4o identifies them, because GPT-4o outputs text descriptions without coordinates. Our pipeline maintains coordinate-level tracking from initial detection through every agent — no information loss, no mismatch between what was identified and what was located.

#### 2. Element-Type-Aware + Selective Segmentation
ImagoObscura runs SAM uniformly on all detected elements. Our framework applies SAM selectively — only on elements that actually need protection (`method != none`), and only where pixel-level precision improves the result:
- **Faces** — SAM for precise contour (natural-looking blur)
- **Screens** — SAM with box+point prompts for screen panel isolation (excludes keyboard/bezels)
- **Text** — Rectangular bbox (cleaner for PII redaction, no SAM needed)
- **Low-risk objects** — Skipped entirely (method=none)

SAM is placed *after* all analysis agents (not after detection), so the pipeline progressively narrows scope before the expensive segmentation step. In a typical image with 7 detections but only 1-2 needing protection, this saves 5-6 unnecessary SAM calls compared to ImagoObscura's approach.

#### 3. Consent-Aware Protection
ImagoObscura has no consent system — every face is treated identically. Our Agent 2.5 maintains a learned identity database:
- User's own face → never protected (explicit consent)
- Known contacts with approval history → lighter protection (assumed consent)
- Unknown bystanders → mandatory protection (no consent)

This eliminates unnecessary protection of the user's own face and trusted contacts.

#### 4. Multi-Layer Safety Architecture
ImagoObscura relies on GPT-4o's judgment alone. Our framework has three independent safety layers:
1. **Deterministic rules** (Phase 1) — correct defaults without any LLM
2. **VLM prompt guidance** (Phase 2) — guides VLM toward correct behavior
3. **Tool-level guards** (hard blocks) — prevent VLM from making dangerous mistakes

Even if the VLM completely ignores its instructions, the tool guards prevent weakening critical protections or removing bystander face protection.

#### 5. On-Device Architecture
ImagoObscura requires GPT-4o (cloud API) for risk assessment. Our framework runs entirely on-device:
- VLM: Qwen3-VL-30B-A3B via llama.cpp (Metal GPU)
- Detection: YOLO + MTCNN + EasyOCR (local)
- Segmentation: MobileSAM (~40MB, local)
- Database: MongoDB (local Docker container)

No image data leaves the device — critical for privacy-focused tools.

#### 6. Label vs Value Distinction
Our composite auto-split (Phase 1.5b) automatically separates "Bank Account: 8765 4321" into a LOW-severity label and a CRITICAL-severity value. The VLM is then guided to understand this distinction, preventing over-protection of labels while ensuring values are always redacted.

---

## Data Models

### Core Enums

```python
RiskLevel:      CRITICAL | HIGH | MEDIUM | LOW
RiskType:       IDENTITY_EXPOSURE | INFORMATION_DISCLOSURE | LOCATION_EXPOSURE | CONTEXT_EXPOSURE
ObfuscationMethod: BLUR | PIXELATE | SOLID_OVERLAY | INPAINT | AVATAR_REPLACE | GENERATIVE_REPLACE | NONE
PersonClassification: PRIMARY_SUBJECT | KNOWN_CONTACT | BYSTANDER
ConsentStatus:  EXPLICIT | ASSUMED | NONE | UNCLEAR
ProcessingMode: AUTO | HYBRID | MANUAL
EthicalMode:    STRICT | BALANCED | CREATIVE
```

### Key Models

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `DetectionResults` | Agent 1 output | faces[], text_regions[], objects[], scene_context |
| `RiskAssessment` | Agent 2 output | detection_id, severity, risk_type, bbox, requires_protection, consent_status |
| `RiskAnalysisResult` | Agent 2 aggregate | risk_assessments[], overall_risk_level, confirmed_risks |
| `ProtectionStrategy` | Agent 3 output | detection_id, method, parameters, reasoning, alternatives, execution_priority |
| `StrategyRecommendations` | Agent 3 aggregate | strategies[], total_protections_recommended |
| `PersonEntry` | Agent 2.5 DB record | person_id, label, embeddings[], consent_history, risk_decay_factor |
| `ProvenanceLog` | Audit trail | image hashes, edits applied, authenticity score, deception risk |

---

## Configuration

The system is configured via `configs/config.yaml`:

```yaml
system:
  device: "mps"              # Apple Silicon GPU
  on_device_only: true        # No cloud APIs
  enable_cloud_llm: false

models:
  detection:
    face_detector: "mtcnn"
    text_detector: "easyocr"
    object_detector: "yolo"
    confidence_threshold: 0.7
  face_recognition:
    similarity_threshold: 0.85  # Cosine similarity for face matching
    confident_threshold: 0.90

risk_levels:
  critical:
    color: "#FF0000"
    requires_protection: true
    user_can_override: false
  # ... high, medium, low

ethical_modes:
  strict:
    allowed: [blur, pixelate, solid_overlay]
  balanced:
    allowed: [blur, pixelate, solid_overlay, avatar_replace, inpaint]
  creative:
    allowed: all

learning:
  min_appearances_for_trust: 3
  trust_approval_threshold: 0.8
  risk_decay_per_approval: 0.1
```

---

## Setup & Installation

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 (via conda)
- Docker (for MongoDB)
- ~16GB RAM recommended

### 1. Environment Setup

```bash
# Create conda environment
conda create -n lab_env python=3.11
conda activate lab_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Build llama.cpp Backend

```bash
cd backend-engines/llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
cd ../..
```

### 3. Download VLM Model

```bash
huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF \
  Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf \
  mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf \
  --local-dir ./backend-engines/models
```

### 4. Start Services

```bash
# Terminal 1: Start MongoDB
docker compose up -d

# Terminal 2: Start VLM server
bash start_llama_server.sh
# Serves at http://localhost:8081/v1
# Model: Qwen3-VL-30B-A3B-Instruct Q4_K_M
# Features: Vision + native tool calling (--jinja)
```

### 5. Verify Setup

```bash
conda run -n lab_env python3 tests/check_setup.py
```

---

## Usage

### Run Full Pipeline

```bash
# Full pipeline (requires llama-server + MongoDB)
conda run -n lab_env python3 tests/test_full_pipeline.py

# Phase 1 only (no VLM server needed, deterministic only)
conda run -n lab_env python3 tests/test_full_pipeline.py --fallback-only
```

### Output Files

Results are saved to `data/full_pipeline_results/`:

| File | Description |
|------|-------------|
| `{sample}_risk_results.json` | Complete risk assessments with severity, identity, consent |
| `{sample}_risk_map.png` | Visual risk map with color-coded bounding boxes |
| `{sample}_strategies.json` | Obfuscation strategy recommendations per element |
| `masks/{detection_id}_mask.npy` | SAM segmentation masks for elements needing protection |

### Risk Map Color Coding

| Color | Severity | Meaning |
|-------|----------|---------|
| Red (#FF0000) | CRITICAL | Mandatory protection (bystander faces, SSNs, passwords) |
| Orange (#FF6600) | HIGH | Strong protection recommended |
| Gold (#FFD700) | MEDIUM | Protection recommended (screen content, unclear consent) |
| Light Green (#90EE90) | LOW | No protection needed (user's face, labels, low-risk objects) |

---

## Project Structure

```
on-device-image-privacy-detector/
├── agents/
│   ├── detection_agent.py          # Agent 1: Face/text/object detection
│   ├── risk_assessment_agent.py    # Agent 2: Two-phase risk analysis
│   ├── consent_identity_agent.py   # Agent 2.5: Face matching + consent
│   ├── strategy_agent.py           # Agent 3: Obfuscation strategies
│   ├── local_wrapper.py            # VisionLLM wrapper (llama-cpp/MLX/Ollama)
│   └── tools.py                    # 21 specialized tools for all agents
├── utils/
│   ├── models.py                   # 40+ Pydantic data models & enums
│   ├── config.py                   # YAML configuration loader
│   ├── storage.py                  # MongoDB face database + encryption
│   ├── visualization.py            # JSON export + risk map generation
│   ├── segmentation.py             # MobileSAM precision segmentation
│   └── callback.py                 # LangChain streaming callbacks
├── configs/
│   └── config.yaml                 # System configuration
├── tests/
│   ├── test_full_pipeline.py       # Full integration test
│   ├── test_detection_agent.py     # Detection unit tests
│   ├── test_risk_assessment_agent.py # Risk assessment tests
│   ├── test_consent_identity_agent.py # Consent agent tests
│   └── check_setup.py             # Environment verification
├── data/
│   ├── test_images/                # Sample test images (1-4)
│   └── full_pipeline_results/      # Pipeline outputs
├── backend-engines/
│   ├── llama.cpp/                  # C++ LLM serving engine
│   └── models/                     # GGUF model files
├── docker-compose.yml              # MongoDB service
├── start_llama_server.sh           # VLM server startup
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```
