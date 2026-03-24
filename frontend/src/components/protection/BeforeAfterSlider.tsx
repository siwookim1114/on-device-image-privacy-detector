import { useState, useRef, useCallback, useEffect } from 'react';

interface BeforeAfterSliderProps {
  originalUrl: string;
  protectedUrl: string;
}

const MIN_PERCENT = 5;
const MAX_PERCENT = 95;

function clamp(value: number): number {
  return Math.min(MAX_PERCENT, Math.max(MIN_PERCENT, value));
}

export function BeforeAfterSlider({ originalUrl, protectedUrl }: BeforeAfterSliderProps) {
  const [dividerPercent, setDividerPercent] = useState(50);
  const [dragging, setDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const updateFromClientX = useCallback((clientX: number) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const x = clientX - rect.left;
    const percent = clamp((x / rect.width) * 100);
    setDividerPercent(percent);
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setDragging(true);
    updateFromClientX(e.clientX);
  }, [updateFromClientX]);

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    e.preventDefault();
    const touch = e.touches[0];
    if (!touch) return;
    setDragging(true);
    updateFromClientX(touch.clientX);
  }, [updateFromClientX]);

  useEffect(() => {
    if (!dragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      updateFromClientX(e.clientX);
    };

    const handleTouchMove = (e: TouchEvent) => {
      const touch = e.touches[0];
      if (!touch) return;
      updateFromClientX(touch.clientX);
    };

    const handleUp = () => setDragging(false);

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleUp);
    window.addEventListener('touchmove', handleTouchMove, { passive: true });
    window.addEventListener('touchend', handleUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleUp);
    };
  }, [dragging, updateFromClientX]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden select-none"
      style={{ cursor: dragging ? 'ew-resize' : 'default' }}
    >
      {/* Original image — full width, clipped to left portion */}
      <img
        src={originalUrl}
        alt="Original"
        className="absolute inset-0 w-full h-full object-contain"
        draggable={false}
      />

      {/* Protected image — clipped to right portion via clip-path */}
      <img
        src={protectedUrl}
        alt="Protected"
        className="absolute inset-0 w-full h-full object-contain"
        style={{ clipPath: `inset(0 0 0 ${dividerPercent}%)` }}
        draggable={false}
      />

      {/* Divider line */}
      <div
        className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_4px_rgba(0,0,0,0.8)]"
        style={{ left: `${dividerPercent}%`, transform: 'translateX(-50%)' }}
      />

      {/* Drag handle */}
      <div
        className="absolute top-1/2 flex items-center justify-center bg-white rounded-full shadow-lg
                   border-2 border-gray-200 touch-none"
        style={{
          left: `${dividerPercent}%`,
          transform: 'translate(-50%, -50%)',
          width: 44,
          height: 44,
          cursor: 'ew-resize',
        }}
        onMouseDown={handleMouseDown}
        onTouchStart={handleTouchStart}
        role="slider"
        aria-label="Comparison divider"
        aria-valuenow={Math.round(dividerPercent)}
        aria-valuemin={MIN_PERCENT}
        aria-valuemax={MAX_PERCENT}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft') setDividerPercent((p) => clamp(p - 2));
          if (e.key === 'ArrowRight') setDividerPercent((p) => clamp(p + 2));
        }}
      >
        {/* Arrow icons */}
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
          <path d="M6 10l-3 3-3-3M6 10l-3-3-3 3" stroke="#374151" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" transform="translate(0,0)" />
          <path d="M7 7l-2 3 2 3" stroke="#374151" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M13 7l2 3-2 3" stroke="#374151" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      {/* Labels */}
      <span className="absolute top-3 left-3 text-xs font-semibold text-white
                       bg-black/50 backdrop-blur-sm px-2 py-0.5 rounded pointer-events-none">
        Original
      </span>
      <span className="absolute top-3 right-3 text-xs font-semibold text-white
                       bg-black/50 backdrop-blur-sm px-2 py-0.5 rounded pointer-events-none">
        Protected
      </span>
    </div>
  );
}
