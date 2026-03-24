import type { RiskAssessment } from '../../types/risk';

import { BboxRect } from './BboxRect';

interface BboxOverlayProps {
  assessments: RiskAssessment[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  /**
   * Uniform zoom applied to the canvas layer.
   * BboxRect coordinates are already in image space — Konva's layer transform
   * handles the scale. This prop is passed through for any consumer that needs
   * to know the effective stroke width on screen, but the overlay itself does
   * not apply an additional transform.
   */
  scale: number;
  /** Horizontal offset of the image layer within the Stage (in canvas pixels). */
  offsetX: number;
  /** Vertical offset of the image layer within the Stage (in canvas pixels). */
  offsetY: number;
}

export function BboxOverlay({
  assessments,
  selectedId,
  onSelect,
  scale: _scale,
  offsetX: _offsetX,
  offsetY: _offsetY,
}: BboxOverlayProps) {
  return (
    <>
      {assessments.map((assessment) => {
        if (!assessment.bbox) return null;

        const label = `${assessment.element_type.toUpperCase()} ${assessment.severity.toUpperCase()}`;

        return (
          <BboxRect
            key={assessment.detection_id}
            bbox={assessment.bbox}
            severity={assessment.severity}
            selected={selectedId === assessment.detection_id}
            label={label}
            onClick={() => onSelect(assessment.detection_id)}
          />
        );
      })}
    </>
  );
}
