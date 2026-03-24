import { useState } from 'react';

import { downloadArtifact } from '../../api/history';

interface DownloadButtonProps {
  sessionId: string;
  type: 'protected' | 'risk_json' | 'provenance';
  label: string;
}

const MIME_TYPES: Record<DownloadButtonProps['type'], string> = {
  protected: 'image/png',
  risk_json: 'application/json',
  provenance: 'application/json',
};

const FILE_EXTENSIONS: Record<DownloadButtonProps['type'], string> = {
  protected: 'png',
  risk_json: 'json',
  provenance: 'json',
};

export function DownloadButton({ sessionId, type, label }: DownloadButtonProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDownload = async () => {
    setLoading(true);
    setError(null);
    try {
      const blob: Blob = await downloadArtifact(sessionId, type);
      const typedBlob = new Blob([blob], { type: MIME_TYPES[type] });
      const url = URL.createObjectURL(typedBlob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `${sessionId}_${type}.${FILE_EXTENSIONS[type]}`;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(url);
    } catch {
      setError('Download failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="inline-flex flex-col items-start gap-1">
      <button
        onClick={() => void handleDownload()}
        disabled={loading}
        className="inline-flex items-center gap-1.5 text-sm text-blue-400 hover:text-blue-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        title={`Download ${label}`}
      >
        {loading ? (
          <svg
            className="w-4 h-4 animate-spin"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden="true"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
        ) : (
          <svg
            className="w-4 h-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
          </svg>
        )}
        {label}
      </button>
      {error !== null && (
        <span className="text-xs text-red-400">{error}</span>
      )}
    </div>
  );
}
