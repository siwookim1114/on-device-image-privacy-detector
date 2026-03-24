import { useState } from 'react';
import { Group, Rect, Text } from 'react-konva';
import type { BoundingBox } from '../../types/detection';
import type { RiskLevel } from '../../types/risk';
import { SEVERITY_COLORS } from '../../lib/colors';

interface BboxRectProps {
  bbox: BoundingBox;
  severity: RiskLevel;
  selected: boolean;
  label: string;
  onClick: () => void;
}

/** Lighten a hex/rgba colour for hover and selected states. */
function selectedFill(baseFill: string): string {
  // Replace the alpha component in rgba(r,g,b,alpha) with a higher value.
  return baseFill.replace(/rgba\((.+),\s*[\d.]+\)/, 'rgba($1, 0.22)');
}

function hoverFill(baseFill: string): string {
  return baseFill.replace(/rgba\((.+),\s*[\d.]+\)/, 'rgba($1, 0.15)');
}

const LABEL_FONT_SIZE = 12;
const LABEL_PADDING = 3;
const DEFAULT_STROKE_WIDTH = 1.5;
const SELECTED_STROKE_WIDTH = 2.5;

export function BboxRect({ bbox, severity, selected, label, onClick }: BboxRectProps) {
  const [hovered, setHovered] = useState(false);

  const colors = SEVERITY_COLORS[severity];

  const strokeWidth = selected ? SELECTED_STROKE_WIDTH : DEFAULT_STROKE_WIDTH;
  const dash = selected ? [6, 3] : undefined;

  let fillColor = colors.fill;
  if (selected) {
    fillColor = selectedFill(colors.fill);
  } else if (hovered) {
    fillColor = hoverFill(colors.fill);
  }

  // Label sits just above the bounding box.
  const labelY = bbox.y - LABEL_FONT_SIZE - LABEL_PADDING * 2 - 2;

  return (
    <Group
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Main bounding box rectangle */}
      <Rect
        x={bbox.x}
        y={bbox.y}
        width={bbox.width}
        height={bbox.height}
        stroke={colors.stroke}
        strokeWidth={strokeWidth}
        fill={fillColor}
        dash={dash}
        cornerRadius={2}
        listening={true}
      />

      {/* Label background pill */}
      <Rect
        x={bbox.x}
        y={labelY}
        width={bbox.width}
        height={LABEL_FONT_SIZE + LABEL_PADDING * 2}
        fill={fillColor}
        stroke={colors.stroke}
        strokeWidth={1}
        cornerRadius={2}
        listening={false}
      />

      {/* Label text */}
      <Text
        x={bbox.x + LABEL_PADDING}
        y={labelY + LABEL_PADDING}
        text={label}
        fontSize={LABEL_FONT_SIZE}
        fontFamily="monospace"
        fill={colors.stroke}
        listening={false}
      />
    </Group>
  );
}
