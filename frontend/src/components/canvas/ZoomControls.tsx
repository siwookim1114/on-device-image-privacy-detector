interface ZoomControlsProps {
  zoom: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitToScreen: () => void;
}

export function ZoomControls({ zoom, onZoomIn, onZoomOut, onFitToScreen }: ZoomControlsProps) {
  const zoomPercent = Math.round(zoom * 100);

  return (
    <div
      className="absolute bottom-3 right-3 flex items-center gap-1 rounded-lg bg-gray-800/80 backdrop-blur p-1 select-none"
      style={{ zIndex: 10 }}
    >
      <button
        type="button"
        onClick={onZoomOut}
        aria-label="Zoom out"
        className="w-7 h-7 flex items-center justify-center rounded text-gray-200 hover:bg-gray-700 active:bg-gray-600 text-sm font-mono transition-colors"
      >
        -
      </button>

      <span className="w-14 text-center text-xs font-mono text-gray-300 tabular-nums">
        {zoomPercent}%
      </span>

      <button
        type="button"
        onClick={onZoomIn}
        aria-label="Zoom in"
        className="w-7 h-7 flex items-center justify-center rounded text-gray-200 hover:bg-gray-700 active:bg-gray-600 text-sm font-mono transition-colors"
      >
        +
      </button>

      <div className="w-px h-5 bg-gray-600 mx-0.5" aria-hidden="true" />

      <button
        type="button"
        onClick={onFitToScreen}
        aria-label="Fit to screen"
        className="px-2 h-7 flex items-center justify-center rounded text-gray-300 hover:bg-gray-700 active:bg-gray-600 text-xs font-mono transition-colors"
      >
        Fit
      </button>
    </div>
  );
}
