import type { ExecutionReport, TransformationResult } from '../../types/execution';
import type { RiskAssessment } from '../../types/risk';
import { ComparisonView } from './ComparisonView';
import { ActionBar } from './ActionBar';

interface ProtectionPreviewProps {
  originalUrl: string;
  protectedUrl: string;
  executionReport: ExecutionReport;
  assessments: RiskAssessment[];
  onAccept: () => void;
  onReprocess: () => void;
}

type TransformationStatus = TransformationResult['status'];

function statusLabel(status: TransformationStatus): string {
  switch (status) {
    case 'success': return 'Applied';
    case 'failed': return 'Failed';
    case 'skipped': return 'Skipped';
  }
}

function statusClasses(status: TransformationStatus): string {
  switch (status) {
    case 'success': return 'text-green-400';
    case 'failed': return 'text-red-400';
    case 'skipped': return 'text-gray-500';
  }
}

function StatusDot({ status }: { status: TransformationStatus }) {
  const color =
    status === 'success' ? 'bg-green-500' :
    status === 'failed' ? 'bg-red-500' : 'bg-gray-600';
  return <span className={`inline-block w-1.5 h-1.5 rounded-full shrink-0 ${color}`} />;
}

function ReportStatusBadge({ status }: { status: ExecutionReport['status'] }) {
  if (status === 'completed') {
    return (
      <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-green-500/15 text-green-400 border border-green-500/30">
        Completed
      </span>
    );
  }
  if (status === 'partial') {
    return (
      <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-yellow-500/15 text-yellow-400 border border-yellow-500/30">
        Partial
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-red-500/15 text-red-400 border border-red-500/30">
      Failed
    </span>
  );
}

export function ProtectionPreview({
  originalUrl,
  protectedUrl,
  executionReport,
  assessments: _assessments,
  onAccept,
  onReprocess,
}: ProtectionPreviewProps) {
  const transformations = executionReport.transformations_applied;
  const unchanged = executionReport.elements_unchanged;

  const applied = transformations.filter((t) => t.status === 'success').length;
  const failed = transformations.filter((t) => t.status === 'failed').length;
  const skipped = transformations.filter((t) => t.status === 'skipped').length;

  const patchCount = transformations.filter(
    (t) => t.status === 'success' && t.method !== 'none',
  ).length;

  const handleAdjust = () => {
    window.history.back();
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Stats bar */}
      <div className="flex items-center gap-4 px-4 py-2 bg-gray-900 border-b border-gray-800 shrink-0">
        <ReportStatusBadge status={executionReport.status} />
        <span className="text-xs text-gray-400">
          <span className="text-green-400 font-semibold tabular-nums">{applied}</span> applied
        </span>
        <span className="text-xs text-gray-400">
          <span className="text-gray-500 font-semibold tabular-nums">{unchanged.length}</span> unchanged
        </span>
        {failed > 0 && (
          <span className="text-xs text-gray-400">
            <span className="text-red-400 font-semibold tabular-nums">{failed}</span> failed
          </span>
        )}
        {skipped > 0 && (
          <span className="text-xs text-gray-400">
            <span className="text-gray-500 font-semibold tabular-nums">{skipped}</span> skipped
          </span>
        )}
        <span className="ml-auto text-xs text-gray-600 tabular-nums">
          {executionReport.total_execution_time_ms}ms total
        </span>
      </div>

      {/* Main content */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Comparison view — takes the majority of horizontal space */}
        <div className="flex-1 min-w-0 min-h-0 flex flex-col">
          <ComparisonView originalUrl={originalUrl} protectedUrl={protectedUrl} />
        </div>

        {/* Transformation sidebar */}
        <div className="w-64 shrink-0 border-l border-gray-800 bg-gray-900/50 flex flex-col overflow-hidden">
          <div className="px-3 py-2 border-b border-gray-800 shrink-0">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
              Transformations
            </h3>
          </div>

          <div className="flex-1 overflow-y-auto">
            {transformations.length === 0 && (
              <p className="text-xs text-gray-600 px-3 py-4 text-center">No transformations recorded.</p>
            )}

            {transformations.map((t) => (
              <div
                key={t.detection_id}
                className={[
                  'px-3 py-2.5 border-b border-gray-800/50 transition-colors',
                  t.status === 'failed' ? 'bg-red-950/20' : '',
                ].join(' ')}
              >
                <div className="flex items-center gap-2 mb-0.5">
                  <StatusDot status={t.status} />
                  <span className="text-xs text-gray-200 truncate flex-1 leading-tight">
                    {t.element}
                  </span>
                </div>
                <div className="flex items-center justify-between pl-3.5">
                  <span className="text-xs font-mono text-gray-500">{t.method}</span>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-medium ${statusClasses(t.status)}`}>
                      {statusLabel(t.status)}
                    </span>
                    <span className="text-xs text-gray-600 tabular-nums">
                      {t.execution_time_ms}ms
                    </span>
                  </div>
                </div>
                {t.status === 'failed' && t.error_message && (
                  <p className="text-xs text-red-400 mt-1 pl-3.5 leading-tight line-clamp-2">
                    {t.error_message}
                  </p>
                )}
              </div>
            ))}

            {unchanged.length > 0 && (
              <>
                <div className="px-3 py-1.5 bg-gray-900/80 border-b border-gray-800">
                  <span className="text-xs text-gray-600 uppercase tracking-wide font-medium">
                    Unchanged ({unchanged.length})
                  </span>
                </div>
                {unchanged.map((item) => (
                  <div key={item.detection_id} className="px-3 py-2 border-b border-gray-800/50">
                    <div className="flex items-center gap-2">
                      <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-700 shrink-0" />
                      <span className="text-xs text-gray-500 truncate">{item.detection_id}</span>
                    </div>
                    <p className="text-xs text-gray-600 mt-0.5 pl-3.5 leading-tight line-clamp-1">
                      {item.reason}
                    </p>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Action bar */}
      <ActionBar
        onAccept={onAccept}
        onAdjust={handleAdjust}
        onReprocess={onReprocess}
        patchCount={patchCount}
      />
    </div>
  );
}
