import type { RiskAssessment, RiskLevel } from '../../types/risk';
import { ElementCard } from './ElementCard';

interface ElementListProps {
  assessments: RiskAssessment[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

type GroupKey = 'FACES' | 'TEXT' | 'OBJECTS';

const GROUP_LABELS: Record<GroupKey, string> = {
  FACES: 'Faces',
  TEXT: 'Text',
  OBJECTS: 'Objects',
};

const ELEMENT_TYPE_TO_GROUP: Record<string, GroupKey> = {
  face: 'FACES',
  text: 'TEXT',
  object: 'OBJECTS',
};

const SEVERITY_ORDER: Record<RiskLevel, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3,
};

function sortBySeverity(items: RiskAssessment[]): RiskAssessment[] {
  return [...items].sort(
    (a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity],
  );
}

function CountBadge({ count }: { count: number }) {
  return (
    <span className="ml-1.5 inline-flex items-center justify-center rounded-full bg-gray-700 text-gray-300 text-xs font-medium w-5 h-5 leading-none">
      {count}
    </span>
  );
}

export function ElementList({ assessments, selectedId, onSelect }: ElementListProps) {
  const groups: Record<GroupKey, RiskAssessment[]> = { FACES: [], TEXT: [], OBJECTS: [] };

  for (const a of assessments) {
    const g = ELEMENT_TYPE_TO_GROUP[a.element_type];
    if (g) groups[g].push(a);
  }

  const orderedGroups: GroupKey[] = ['FACES', 'TEXT', 'OBJECTS'];

  return (
    <div className="flex-1 overflow-y-auto">
      {orderedGroups.map((groupKey) => {
        const items = sortBySeverity(groups[groupKey]);
        if (items.length === 0) return null;

        return (
          <div key={groupKey}>
            {/* Sticky group header */}
            <div className="sticky top-0 z-10 flex items-center px-3 py-1.5 bg-gray-900 border-b border-gray-800">
              <span className="text-xs uppercase tracking-wider font-semibold text-gray-500">
                {GROUP_LABELS[groupKey]}
              </span>
              <CountBadge count={items.length} />
            </div>

            {/* Items */}
            <div className="divide-y divide-gray-800/60">
              {items.map((a) => (
                <ElementCard
                  key={a.detection_id}
                  assessment={a}
                  selected={selectedId === a.detection_id}
                  onClick={() => onSelect(a.detection_id)}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
