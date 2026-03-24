import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppSidebar } from './components/layout/AppSidebar';
import { Dashboard } from './pages/Dashboard';
import { Workspace } from './pages/Workspace';
import HistoryPageFull from './pages/History';
import { Consent } from './pages/Consent';
import { Onboarding } from './pages/Onboarding';

function isOnboardingComplete(): boolean {
  return localStorage.getItem('onboarding_complete') === 'true';
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Onboarding — full screen, no sidebar */}
        <Route path="/onboarding" element={<Onboarding />} />

        {/* Main app — sidebar layout */}
        <Route
          path="*"
          element={
            <div className="flex h-screen bg-gray-950 text-gray-100 overflow-hidden">
              <AppSidebar />
              {/*
                main fills the remaining horizontal space and matches h-screen so that
                child pages with h-full (WorkspaceLayout) correctly fill the viewport.
                Pages that need scrolling (Dashboard, Consent) use overflow-auto on
                their own root element.
              */}
              <main className="flex-1 flex flex-col min-w-0 overflow-auto">
                <Routes>
                  <Route
                    path="/"
                    element={
                      isOnboardingComplete() ? <Dashboard /> : <Navigate to="/onboarding" replace />
                    }
                  />
                  <Route path="/process/:sessionId" element={<Workspace />} />
                  <Route path="/consent" element={<Consent />} />
                  <Route path="/history" element={<HistoryPageFull />} />
                </Routes>
              </main>
            </div>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
