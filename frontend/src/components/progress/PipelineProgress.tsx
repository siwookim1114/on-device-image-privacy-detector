import type { PipelineStage } from '../../types';

import { STAGE_LABELS } from '../../lib/colors';
import { StageIndicator } from './StageIndicator';

interface PipelineProgressProps {
  currentStage: PipelineStage;
  stageTimings: Partial<Record<string, number>>;
}

const PIPELINE_STAGES: PipelineStage[] = [
  'detection',
  'risk',
  'consent',
  'strategy',
  'sam',
  'execution',
];

const STAGE_ORDER: Record<PipelineStage, number> = {
  detection: 0,
  risk: 1,
  consent: 2,
  strategy: 3,
  sam: 4,
  execution: 5,
  export: 6,
  done: 7,
};

function getStatus(
  stage: PipelineStage,
  currentStage: PipelineStage,
): 'completed' | 'active' | 'pending' {
  const stageIdx = STAGE_ORDER[stage];
  const currentIdx = STAGE_ORDER[currentStage];

  if (currentIdx > stageIdx) return 'completed';
  if (currentIdx === stageIdx) return 'active';
  return 'pending';
}

function ConnectorLine({ completed }: { completed: boolean }) {
  return (
    <div className="flex-1 h-0.5 mx-1 mb-7" aria-hidden="true">
      <div
        className={`h-full rounded-full transition-colors duration-300 ${
          completed ? 'bg-green-500' : 'bg-gray-700'
        }`}
      />
    </div>
  );
}

export function PipelineProgress({ currentStage, stageTimings }: PipelineProgressProps) {
  return (
    <div
      className="flex items-start px-4 py-3 bg-gray-900 border-b border-gray-800"
      role="progressbar"
      aria-label="Pipeline progress"
    >
      {PIPELINE_STAGES.map((stage, idx) => {
        const status = getStatus(stage, currentStage);
        const label: string = STAGE_LABELS[stage] ?? stage;
        const timingMs = stageTimings[stage];

        return (
          <div key={stage} className="flex items-center flex-1 min-w-0">
            <StageIndicator
              stage={stage}
              label={label}
              status={status}
              timingMs={status === 'completed' ? timingMs : undefined}
            />
            {idx < PIPELINE_STAGES.length - 1 && (
              <ConnectorLine completed={status === 'completed'} />
            )}
          </div>
        );
      })}
    </div>
  );
}
