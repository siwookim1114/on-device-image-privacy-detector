import { useState, useRef, useCallback, type ReactNode } from 'react';

interface WorkspaceLayoutProps {
  sidebar: ReactNode;
  canvas: ReactNode;
  panel: ReactNode;
}

const SIDEBAR_MIN = 240;
const SIDEBAR_MAX = 400;
const SIDEBAR_DEFAULT = 320;
const PANEL_WIDTH = 360;

function PanelToggleIcon({ open }: { open: boolean }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`w-4 h-4 transition-transform duration-200 ${open ? '' : 'rotate-180'}`}
    >
      <polyline points="15 18 9 12 15 6" />
    </svg>
  );
}

export function WorkspaceLayout({ sidebar, canvas, panel }: WorkspaceLayoutProps) {
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT);
  const [panelOpen, setPanelOpen] = useState(true);
  const dragging = useRef(false);
  const dragStartX = useRef(0);
  const dragStartWidth = useRef(SIDEBAR_DEFAULT);

  const onDragStart = useCallback((e: React.MouseEvent) => {
    dragging.current = true;
    dragStartX.current = e.clientX;
    dragStartWidth.current = sidebarWidth;

    const onMove = (ev: MouseEvent) => {
      if (!dragging.current) return;
      const delta = ev.clientX - dragStartX.current;
      const next = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, dragStartWidth.current + delta));
      setSidebarWidth(next);
    };

    const onUp = () => {
      dragging.current = false;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [sidebarWidth]);

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left Sidebar — fixed width, resizable via drag handle */}
      <div
        className="relative flex flex-col bg-gray-900 overflow-y-auto overflow-x-hidden shrink-0"
        style={{ width: sidebarWidth }}
      >
        {sidebar}

        {/* Drag handle */}
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize sidebar"
          className="absolute top-0 right-0 w-1 h-full cursor-col-resize group z-10"
          onMouseDown={onDragStart}
        >
          <div className="w-full h-full bg-transparent group-hover:bg-blue-500/40 transition-colors duration-150" />
        </div>
      </div>

      {/* Center Canvas — flex-1 */}
      <div className="flex-1 flex flex-col bg-gray-950 overflow-hidden min-w-0">
        {canvas}
      </div>

      {/* Right Panel toggle button — sits at the seam */}
      <button
        onClick={() => setPanelOpen((v) => !v)}
        className="flex items-center justify-center w-5 bg-gray-900 border-l border-gray-800 text-gray-500 hover:text-gray-300 hover:bg-gray-800 transition-colors duration-150 shrink-0 z-10"
        aria-label={panelOpen ? 'Collapse right panel' : 'Expand right panel'}
        title={panelOpen ? 'Collapse panel' : 'Expand panel'}
      >
        <PanelToggleIcon open={panelOpen} />
      </button>

      {/* Right Panel — collapsible */}
      <div
        className={`
          flex flex-col bg-gray-900 border-l border-gray-800 shrink-0 overflow-y-auto overflow-x-hidden
          transition-[width] duration-200 ease-in-out
        `}
        style={{ width: panelOpen ? PANEL_WIDTH : 0 }}
        aria-hidden={!panelOpen}
      >
        <div style={{ width: PANEL_WIDTH, minWidth: PANEL_WIDTH }}>
          {panel}
        </div>
      </div>
    </div>
  );
}
