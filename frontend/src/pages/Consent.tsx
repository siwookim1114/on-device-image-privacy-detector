import { FaceRegistration } from '../components/consent/FaceRegistration';
import { FaceDatabase } from '../components/consent/FaceDatabase';

function LockIcon() {
  return (
    <svg
      className="w-3.5 h-3.5 text-green-400 shrink-0"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      aria-hidden="true"
    >
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M7 11V7a5 5 0 0110 0v4" />
    </svg>
  );
}

export function Consent() {
  return (
    <div className="min-h-full px-6 py-10 max-w-5xl mx-auto space-y-10">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-100 tracking-tight">Consent Management</h1>
        <p className="mt-1.5 text-sm text-gray-400">
          Register faces and manage consent permissions for privacy-aware image protection.
        </p>
      </div>

      {/* Two-column layout on wider screens */}
      <div className="flex flex-col lg:flex-row gap-8 items-start">
        {/* Left: Register form */}
        <div className="w-full lg:w-auto lg:shrink-0">
          <FaceRegistration />
        </div>

        {/* Right: Divider on large screens */}
        <div
          className="hidden lg:block w-px self-stretch bg-gray-800 shrink-0"
          aria-hidden="true"
        />

        {/* Right: Face database browser */}
        <div className="flex-1 min-w-0 w-full">
          <FaceDatabase />
        </div>
      </div>

      {/* Privacy notice */}
      <footer className="flex items-center gap-2 rounded-xl bg-gray-900 border border-gray-800 px-4 py-3">
        <LockIcon />
        <p className="text-xs text-gray-500">
          All face data is stored locally with encryption. No data leaves this device.
        </p>
      </footer>
    </div>
  );
}
