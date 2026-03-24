import { useEffect, useCallback, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

import type { RiskLevel } from '../types/risk';
import type { ObfuscationMethod } from '../types/strategy';
import type { TransformationResult } from '../types/execution';
import { getPipelineResults, approvePipeline, getImageUrl } from '../api/pipeline';
import { usePipelineStore } from '../stores/pipelineStore';
import { useCanvasInteraction } from '../hooks/useCanvasInteraction';
import { fitScale } from '../lib/canvas-utils';
import { WorkspaceLayout } from '../components/layout/WorkspaceLayout';
import { Header } from '../components/layout/Header';
import { PipelineProgress } from '../components/progress/PipelineProgress';
import { ImageCanvas } from '../components/canvas/ImageCanvas';
import { BboxOverlay } from '../components/canvas/BboxOverlay';
import { ZoomControls } from '../components/canvas/ZoomControls';
import { DetectionReview } from '../components/detection/DetectionReview';
import { StrategyReview } from '../components/strategy/StrategyReview';
import { ConversationPanel } from '../components/chat/ConversationPanel';
import { ComparisonView } from '../components/protection/ComparisonView';

function SkeletonBar({ w }: { w: string }) {
  return <div className={`h-3 rounded-full bg-gray-800 animate-pulse ${w}`} />;
}

function WorkspaceSkeleton() {
  return (
    <div className="flex h-full">
      <div className="w-80 bg-gray-900 border-r border-gray-800 p-4 space-y-3 shrink-0">
        <SkeletonBar w="w-1/2" />
        <SkeletonBar w="w-full" />
        <SkeletonBar w="w-3/4" />
        <SkeletonBar w="w-full" />
        <SkeletonBar w="w-2/3" />
      </div>
      <div className="flex-1 bg-gray-950 flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-12 h-12 rounded-full border-2 border-blue-500/30 border-t-blue-500 animate-spin" />
          <span className="text-sm text-gray-500">Loading session…</span>
        </div>
      </div>
    </div>
  );
}

function WorkspaceError({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="flex h-full items-center justify-center bg-gray-950">
      <div className="flex flex-col items-center gap-4 max-w-sm text-center px-6">
        <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center">
          <svg
            className="w-6 h-6 text-red-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
            />
          </svg>
        </div>
        <div>
          <p className="text-sm font-semibold text-gray-200">Failed to load session</p>
          <p className="mt-1 text-xs text-gray-500">{message}</p>
        </div>
        <button
          type="button"
          onClick={onRetry}
          className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-950"
        >
          Retry
        </button>
      </div>
    </div>
  );
}

const CHECKPOINT_LABELS: Record<string, string> = {
  risk_review: 'Risk Assessment Review',
  strategy_review: 'Strategy Review',
  execution_verify: 'Execution Verification',
};

interface HitlBannerProps {
  checkpoint: string;
  onApprove: () => void;
  approving: boolean;
}

function HitlBanner({ checkpoint, onApprove, approving }: HitlBannerProps) {
  const label = CHECKPOINT_LABELS[checkpoint] ?? checkpoint;
  return (
    <div
      role="status"
      className="shrink-0 flex items-center justify-between gap-4 px-4 py-2.5 bg-yellow-500/10 border-b border-yellow-500/30"
    >
      <div className="flex items-center gap-2 min-w-0">
        <svg
          className="w-4 h-4 text-yellow-400 shrink-0"
          viewBox="0 0 16 16"
          fill="none"
          aria-hidden="true"
        >
          <path
            d="M8 2.5l5.5 10H2.5L8 2.5z"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinejoin="round"
          />
          <path d="M8 6.5v3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="8" cy="11.5" r="0.75" fill="currentColor" />
        </svg>
        <p className="text-xs text-yellow-300 truncate">
          <span className="font-semibold">Checkpoint:</span>{' '}
          {label} — review the results below before continuing.
        </p>
      </div>
      <button
        type="button"
        onClick={onApprove}
        disabled={approving}
        className="shrink-0 px-3 py-1 rounded-md bg-yellow-500 hover:bg-yellow-400 disabled:bg-yellow-700 disabled:cursor-not-allowed text-gray-900 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2 focus:ring-offset-gray-950"
      >
        {approving ? 'Approving…' : 'Approve'}
      </button>
    </div>
  );
}

type WorkspaceTab = 'detection' | 'strategy' | 'protection';

const TABS: { id: WorkspaceTab; label: string }[] = [
  { id: 'detection', label: 'Detection' },
  { id: 'strategy', label: 'Strategy' },
  { id: 'protection', label: 'Protection' },
];

interface TabBarProps {
  active: WorkspaceTab;
  onChange: (tab: WorkspaceTab) => void;
}

function TabBar({ active, onChange }: TabBarProps) {
  return (
    <div
      className="flex items-center gap-0.5 px-3 py-2 bg-gray-900 border-b border-gray-800 shrink-0"
      role="tablist"
      aria-label="Workspace tabs"
    >
      {TABS.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          aria-selected={active === tab.id}
          type="button"
          onClick={() => onChange(tab.id)}
          className={[
            'px-3 py-1.5 rounded-md text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/50',
            active === tab.id
              ? 'bg-blue-600 text-white'
              : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800',
          ].join(' ')}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function TransformationList({
  transformations,
}: {
  transformations: TransformationResult[];
}) {
  if (transformations.length === 0) {
    return (
      <div className="flex flex-col h-full items-center justify-center px-6 text-center gap-2">
        <p className="text-sm text-gray-500">No transformations recorded yet.</p>
      </div>
    );
  }

  const successCount = transformations.filter((t) => t.status === 'success').length;

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-gray-800 shrink-0">
        <h2 className="text-sm font-semibold text-gray-200">Applied Protections</h2>
        <p className="mt-0.5 text-xs text-gray-500">
          {successCount} of {transformations.length} applied successfully
        </p>
      </div>
      <ul className="flex-1 divide-y divide-gray-800 overflow-y-auto">
        {transformations.map((t) => (
          <li key={t.detection_id} className="px-4 py-3 flex items-start gap-3">
            <span
              className={[
                'mt-0.5 w-2 h-2 rounded-full shrink-0',
                t.status === 'success'
                  ? 'bg-green-400'
                  : t.status === 'failed'
                    ? 'bg-red-400'
                    : 'bg-gray-500',
              ].join(' ')}
              aria-label={t.status}
            />
            <div className="min-w-0">
              <p className="text-xs font-medium text-gray-300 truncate">{t.element}</p>
              <p className="text-xs text-gray-500 mt-0.5">
                {t.method}
                {t.execution_time_ms > 0 && (
                  <span className="ml-2 tabular-nums">{t.execution_time_ms}ms</span>
                )}
              </p>
              {t.error_message !== null && (
                <p className="text-xs text-red-400 mt-0.5 break-words">{t.error_message}</p>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

const ZOOM_STEP = 1.25;

function zoomedFitArgs(
  iw: number,
  ih: number,
  cw: number,
  ch: number,
  currentZoom: number,
  factor: number,
): { iw: number; ih: number } | null {
  if (iw <= 0 || ih <= 0 || cw <= 0 || ch <= 0) return null;
  const baseResult = fitScale(iw, ih, cw, ch);
  const targetZoom = Math.min(10, Math.max(0.05, currentZoom * factor));
  const adjustedIw = (iw * baseResult.scale) / targetZoom;
  const adjustedIh = (ih * baseResult.scale) / targetZoom;
  return { iw: adjustedIw, ih: adjustedIh };
}

function WorkspaceContent({ sessionId }: { sessionId: string }) {
  const navigate = useNavigate();

  const {
    imageUrl,
    currentStage,
    stageTimings,
    hitlCheckpoint,
    riskAnalysis,
    strategies,
    executionReport,
    protectedImageUrl,
    selectedElementId,
    activeTab,
    setSession,
    setStage,
    setHitlCheckpoint,
    setRiskAnalysis,
    setStrategies,
    setExecutionReport,
    setProtectedImageUrl,
    selectElement,
    setActiveTab,
    addTiming,
  } = usePipelineStore();

  const [loading, setLoading] = useState(imageUrl === null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [approving, setApproving] = useState(false);

  const canvasInteraction = useCanvasInteraction();
  const { zoom, fitToScreen } = canvasInteraction;

  const imageDimsRef = useRef({ width: 0, height: 0 });
  const containerDimsRef = useRef({ width: 0, height: 0 });

  const handleImageLoaded = useCallback((w: number, h: number) => {
    imageDimsRef.current = { width: w, height: h };
  }, []);

  const handleContainerResized = useCallback((w: number, h: number) => {
    containerDimsRef.current = { width: w, height: h };
  }, []);

  const handleFitToScreen = useCallback(() => {
    const { width: iw, height: ih } = imageDimsRef.current;
    const { width: cw, height: ch } = containerDimsRef.current;
    if (iw > 0 && ih > 0 && cw > 0 && ch > 0) fitToScreen(iw, ih, cw, ch);
  }, [fitToScreen]);

  const handleZoomIn = useCallback(() => {
    const { width: iw, height: ih } = imageDimsRef.current;
    const { width: cw, height: ch } = containerDimsRef.current;
    const args = zoomedFitArgs(iw, ih, cw, ch, zoom, ZOOM_STEP);
    if (args) fitToScreen(args.iw, args.ih, cw, ch);
  }, [fitToScreen, zoom]);

  const handleZoomOut = useCallback(() => {
    const { width: iw, height: ih } = imageDimsRef.current;
    const { width: cw, height: ch } = containerDimsRef.current;
    const args = zoomedFitArgs(iw, ih, cw, ch, zoom, 1 / ZOOM_STEP);
    if (args) fitToScreen(args.iw, args.ih, cw, ch);
  }, [fitToScreen, zoom]);

  const fetchResults = useCallback(async () => {
    setLoading(true);
    setFetchError(null);

    try {
      const data = (await getPipelineResults(sessionId)) as Record<string, unknown>;

      if (!data || typeof data !== 'object') {
        navigate('/');
        return;
      }

      const originalUrl = getImageUrl(sessionId, 'original');
      setSession(sessionId, originalUrl);

      if (typeof data['current_stage'] === 'string') {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        setStage(data['current_stage'] as any);
      }
      if (data['hitl_checkpoint'] !== undefined) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        setHitlCheckpoint((data['hitl_checkpoint'] as any) ?? null);
      }
      if (data['risk_analysis'] !== null && data['risk_analysis'] !== undefined) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        setRiskAnalysis(data['risk_analysis'] as any);
      }
      if (data['strategies'] !== null && data['strategies'] !== undefined) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        setStrategies(data['strategies'] as any);
      }
      if (data['execution_report'] !== null && data['execution_report'] !== undefined) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        setExecutionReport(data['execution_report'] as any);
        const protUrl =
          typeof data['protected_image_url'] === 'string'
            ? data['protected_image_url']
            : getImageUrl(sessionId, 'protected');
        setProtectedImageUrl(protUrl);
      } else if (typeof data['protected_image_url'] === 'string') {
        setProtectedImageUrl(data['protected_image_url']);
      }
      if (data['stage_timings'] !== null && typeof data['stage_timings'] === 'object') {
        for (const [stage, ms] of Object.entries(
          data['stage_timings'] as Record<string, unknown>,
        )) {
          if (typeof ms === 'number') addTiming(stage, ms);
        }
      }

      if (data['execution_report']) {
        setActiveTab('protection');
      } else if (data['strategies']) {
        setActiveTab('strategy');
      } else {
        setActiveTab('detection');
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to load session results.';
      setFetchError(message);
    } finally {
      setLoading(false);
    }
  }, [
    sessionId,
    navigate,
    setSession,
    setStage,
    setHitlCheckpoint,
    setRiskAnalysis,
    setStrategies,
    setExecutionReport,
    setProtectedImageUrl,
    addTiming,
    setActiveTab,
  ]);

  useEffect(() => {
    void fetchResults();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const handleApprove = useCallback(async () => {
    if (!hitlCheckpoint || approving) return;
    setApproving(true);
    try {
      await approvePipeline(sessionId, hitlCheckpoint);
      setHitlCheckpoint(null);
    } catch {
    } finally {
      setApproving(false);
    }
  }, [hitlCheckpoint, approving, sessionId, setHitlCheckpoint]);

  const handleOverrideSeverity = useCallback(
    (id: string, severity: RiskLevel) => {
      if (!riskAnalysis) return;
      setRiskAnalysis({
        ...riskAnalysis,
        risk_assessments: riskAnalysis.risk_assessments.map((a) =>
          a.detection_id === id ? { ...a, severity } : a,
        ),
      });
    },
    [riskAnalysis, setRiskAnalysis],
  );

  const handleMethodChange = useCallback(
    (detectionId: string, method: ObfuscationMethod) => {
      if (!strategies) return;
      setStrategies({
        ...strategies,
        strategies: strategies.strategies.map((s) =>
          s.detection_id === detectionId
            ? { ...s, recommended_method: method }
            : s,
        ),
      });
    },
    [strategies, setStrategies],
  );

  const handleParamsChange = useCallback(
    (detectionId: string, params: Record<string, unknown>) => {
      if (!strategies) return;
      setStrategies({
        ...strategies,
        strategies: strategies.strategies.map((s) =>
          s.detection_id === detectionId ? { ...s, parameters: params } : s,
        ),
      });
    },
    [strategies, setStrategies],
  );

  const handleDetectionApprove = useCallback(() => {
    setActiveTab('strategy');
  }, [setActiveTab]);

  const handleStrategyApprove = useCallback(() => {
    setActiveTab('protection');
  }, [setActiveTab]);

  const assessments = riskAnalysis?.risk_assessments ?? [];

  useEffect(() => {
    const SEVERITY_KEYS: Partial<Record<string, RiskLevel>> = {
      '1': 'critical',
      '2': 'high',
      '3': 'medium',
      '4': 'low',
    };

    const METHOD_KEYS: Partial<Record<string, ObfuscationMethod>> = {
      b: 'blur',
      x: 'pixelate',
      s: 'solid_overlay',
      n: 'none',
    };

    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;

      const lower = e.key.toLowerCase();

      if (e.key === '/') {
        e.preventDefault();
        const chatTextarea = document.querySelector<HTMLTextAreaElement>(
          '[aria-label="Chat message"]',
        );
        chatTextarea?.focus();
        return;
      }

      if (e.key === 'Escape') {
        selectElement(null);
        return;
      }

      if (e.key === 'Tab' && assessments.length > 0) {
        e.preventDefault();
        const ids = assessments.map((a) => a.detection_id);
        const currentIdx = selectedElementId ? ids.indexOf(selectedElementId) : -1;
        const nextIdx = (currentIdx + 1) % ids.length;
        const nextId = ids[nextIdx];
        if (nextId !== undefined) selectElement(nextId);
        return;
      }

      const severity = SEVERITY_KEYS[e.key];
      if (severity !== undefined && selectedElementId && activeTab === 'detection') {
        handleOverrideSeverity(selectedElementId, severity);
        return;
      }

      const method = METHOD_KEYS[lower];
      if (method !== undefined && selectedElementId && activeTab === 'strategy') {
        handleMethodChange(selectedElementId, method);
        return;
      }
    };

    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [
    assessments,
    selectedElementId,
    activeTab,
    selectElement,
    handleOverrideSeverity,
    handleMethodChange,
  ]);

  const [elapsedMs, setElapsedMs] = useState(0);
  const startTimeRef = useRef<number>(Date.now());

  useEffect(() => {
    startTimeRef.current = Date.now();
    if (currentStage === 'done') return;
    const timer = setInterval(() => {
      setElapsedMs(Date.now() - startTimeRef.current);
    }, 250);
    return () => clearInterval(timer);
  }, [currentStage]);

  if (loading) return <WorkspaceSkeleton />;
  if (fetchError) {
    return <WorkspaceError message={fetchError} onRetry={() => void fetchResults()} />;
  }

  const effectiveImageUrl = imageUrl ?? '';

  const sidebarContent =
    activeTab === 'detection' ? (
      <DetectionReview
        assessments={assessments}
        selectedId={selectedElementId}
        onSelectElement={(id) => selectElement(id)}
        onOverrideSeverity={handleOverrideSeverity}
        onApprove={handleDetectionApprove}
      />
    ) : activeTab === 'strategy' ? (
      <StrategyReview
        strategies={strategies?.strategies ?? []}
        riskAssessments={assessments}
        onMethodChange={handleMethodChange}
        onParamsChange={handleParamsChange}
        onApprove={handleStrategyApprove}
      />
    ) : (
      <TransformationList
        transformations={executionReport?.transformations_applied ?? []}
      />
    );

  const canvasContent = (
    <>
      <Header
        sessionId={sessionId}
        currentStage={currentStage}
        elapsedMs={elapsedMs}
      />
      <PipelineProgress currentStage={currentStage} stageTimings={stageTimings} />

      {hitlCheckpoint !== null && (
        <HitlBanner
          checkpoint={hitlCheckpoint}
          onApprove={() => void handleApprove()}
          approving={approving}
        />
      )}

      <TabBar active={activeTab} onChange={setActiveTab} />

      {activeTab === 'protection' ? (
        <div className="flex-1 overflow-hidden min-h-0 bg-gray-950">
          {effectiveImageUrl !== '' && protectedImageUrl !== null ? (
            <ComparisonView
              originalUrl={effectiveImageUrl}
              protectedUrl={protectedImageUrl}
            />
          ) : effectiveImageUrl !== '' ? (
            <div className="relative w-full h-full flex items-center justify-center">
              <img
                src={effectiveImageUrl}
                alt="Original — protection not yet applied"
                className="max-w-full max-h-full object-contain"
              />
              <div className="absolute inset-0 flex items-end justify-center pb-6 pointer-events-none">
                <span className="text-xs font-medium text-gray-400 bg-gray-900/80 backdrop-blur-sm px-3 py-1.5 rounded-full">
                  Protection not yet applied
                </span>
              </div>
            </div>
          ) : (
            <div className="flex h-full items-center justify-center">
              <p className="text-sm text-gray-500">No image available</p>
            </div>
          )}
        </div>
      ) : (
        /* Detection / Strategy tabs: Konva canvas with bbox overlay */
        <div className="relative flex-1 min-h-0 overflow-hidden">
          {effectiveImageUrl !== '' ? (
            <ImageCanvas
              imageUrl={effectiveImageUrl}
              onImageLoaded={handleImageLoaded}
              onContainerResized={handleContainerResized}
              interaction={canvasInteraction}
            >
              <BboxOverlay
                assessments={assessments}
                selectedId={selectedElementId}
                onSelect={(id) => selectElement(id)}
                scale={zoom}
                offsetX={canvasInteraction.offsetX}
                offsetY={canvasInteraction.offsetY}
              />
            </ImageCanvas>
          ) : (
            <div className="flex h-full items-center justify-center">
              <p className="text-sm text-gray-500">No image loaded</p>
            </div>
          )}

          <ZoomControls
            zoom={zoom}
            onZoomIn={handleZoomIn}
            onZoomOut={handleZoomOut}
            onFitToScreen={handleFitToScreen}
          />
        </div>
      )}
    </>
  );

  return (
    <WorkspaceLayout
      sidebar={sidebarContent}
      canvas={canvasContent}
      panel={<ConversationPanel />}
    />
  );
}

export function Workspace() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();

  useEffect(() => {
    if (!sessionId) navigate('/');
  }, [sessionId, navigate]);

  if (!sessionId) return null;

  return <WorkspaceContent sessionId={sessionId} />;
}
