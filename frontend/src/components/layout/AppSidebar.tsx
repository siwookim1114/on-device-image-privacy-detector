import { useState, type ReactNode } from 'react';
import { NavLink, useLocation } from 'react-router-dom';

interface NavItem {
  to: string;
  label: string;
  icon: ReactNode;
}

function HomeIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
      <path d="M3 9.75L12 3l9 6.75V21a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9.75z" />
      <path d="M9 22V12h6v10" />
    </svg>
  );
}

function ConsentIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
      <circle cx="12" cy="8" r="4" />
      <path d="M4 20c0-4 3.582-7 8-7s8 3 8 7" />
      <path d="M16 11l1.5 1.5L20 10" />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
      <circle cx="12" cy="12" r="9" />
      <polyline points="12 7 12 12 15 15" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function ChevronRightIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4">
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}

const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Home', icon: <HomeIcon /> },
  { to: '/consent', label: 'Consent', icon: <ConsentIcon /> },
  { to: '/history', label: 'History', icon: <HistoryIcon /> },
];

export function AppSidebar() {
  const [expanded, setExpanded] = useState(false);
  const location = useLocation();

  // On small screens (< 1024px) always collapse to icon-only
  // The toggle button controls expanded state for large screens

  const sidebarWidth = expanded ? 'w-[200px]' : 'w-16';

  return (
    <aside
      className={`
        flex flex-col h-full shrink-0
        bg-gray-900 border-r border-gray-800
        transition-[width] duration-200 ease-in-out
        ${sidebarWidth}
        lg:${sidebarWidth}
        max-lg:w-16
      `}
    >
      {/* Logo / Branding */}
      <div className="flex items-center h-14 px-3 border-b border-gray-800 shrink-0 overflow-hidden">
        <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-blue-600 text-white shrink-0">
          <ShieldIcon />
        </div>
        {expanded && (
          <span className="ml-3 text-sm font-semibold text-gray-100 whitespace-nowrap truncate">
            Privacy Guard
          </span>
        )}
      </div>

      {/* Nav Items */}
      <nav className="flex flex-col gap-1 p-2 flex-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const isActive =
            item.to === '/'
              ? location.pathname === '/'
              : location.pathname.startsWith(item.to);

          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={`
                flex items-center gap-3 rounded-lg px-3 py-2.5
                transition-colors duration-150 group
                ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:bg-gray-800 hover:text-gray-100'
                }
              `}
              title={!expanded ? item.label : undefined}
            >
              <span className="shrink-0">{item.icon}</span>
              {expanded && (
                <span className="text-sm font-medium whitespace-nowrap truncate">
                  {item.label}
                </span>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Bottom: Settings + Toggle */}
      <div className="flex flex-col gap-1 p-2 border-t border-gray-800 shrink-0">
        <button
          className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-gray-400 hover:bg-gray-800 hover:text-gray-100 transition-colors duration-150 w-full"
          title={!expanded ? 'Settings' : undefined}
          aria-label="Settings"
        >
          <span className="shrink-0"><SettingsIcon /></span>
          {expanded && (
            <span className="text-sm font-medium whitespace-nowrap">Settings</span>
          )}
        </button>

        {/* Collapse/Expand toggle — hidden on < 1024px (always icon-only there) */}
        <button
          onClick={() => setExpanded((v) => !v)}
          className="hidden lg:flex items-center justify-center w-full rounded-lg px-3 py-2 text-gray-500 hover:bg-gray-800 hover:text-gray-300 transition-colors duration-150"
          aria-label={expanded ? 'Collapse sidebar' : 'Expand sidebar'}
          title={expanded ? 'Collapse' : 'Expand'}
        >
          <span
            className={`transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
          >
            <ChevronRightIcon />
          </span>
        </button>
      </div>
    </aside>
  );
}
