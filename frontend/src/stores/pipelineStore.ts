import { create } from 'zustand';
import type { DetectionResults } from '../types/detection';
import type { RiskAnalysisResult } from '../types/risk';
import type { StrategyRecommendations } from '../types/strategy';
import type { ExecutionReport } from '../types/execution';
import type { PipelineStage, HitlCheckpoint } from '../types/pipeline';

interface PipelineState {
  sessionId: string | null;
  imageUrl: string | null;
  currentStage: PipelineStage;
  hitlCheckpoint: HitlCheckpoint | null;
  detections: DetectionResults | null;
  riskAnalysis: RiskAnalysisResult | null;
  strategies: StrategyRecommendations | null;
  executionReport: ExecutionReport | null;
  protectedImageUrl: string | null;
  selectedElementId: string | null;
  activeTab: 'detection' | 'strategy' | 'protection';
  stageTimings: Record<string, number>;

  setSession: (sessionId: string, imageUrl: string) => void;
  setStage: (stage: PipelineStage) => void;
  setHitlCheckpoint: (cp: HitlCheckpoint | null) => void;
  setDetections: (d: DetectionResults) => void;
  setRiskAnalysis: (r: RiskAnalysisResult) => void;
  setStrategies: (s: StrategyRecommendations) => void;
  setExecutionReport: (e: ExecutionReport) => void;
  setProtectedImageUrl: (url: string) => void;
  selectElement: (id: string | null) => void;
  setActiveTab: (tab: 'detection' | 'strategy' | 'protection') => void;
  addTiming: (stage: string, ms: number) => void;
  reset: () => void;
}

const initialState = {
  sessionId: null, imageUrl: null, currentStage: 'detection' as PipelineStage,
  hitlCheckpoint: null, detections: null, riskAnalysis: null, strategies: null,
  executionReport: null, protectedImageUrl: null, selectedElementId: null,
  activeTab: 'detection' as const, stageTimings: {},
};

export const usePipelineStore = create<PipelineState>((set) => ({
  ...initialState,
  setSession: (sessionId, imageUrl) => set({ sessionId, imageUrl }),
  setStage: (stage) => set({ currentStage: stage }),
  setHitlCheckpoint: (cp) => set({ hitlCheckpoint: cp }),
  setDetections: (d) => set({ detections: d }),
  setRiskAnalysis: (r) => set({ riskAnalysis: r }),
  setStrategies: (s) => set({ strategies: s }),
  setExecutionReport: (e) => set({ executionReport: e }),
  setProtectedImageUrl: (url) => set({ protectedImageUrl: url }),
  selectElement: (id) => set({ selectedElementId: id }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  addTiming: (stage, ms) => set((s) => ({ stageTimings: { ...s.stageTimings, [stage]: ms } })),
  reset: () => set(initialState),
}));
