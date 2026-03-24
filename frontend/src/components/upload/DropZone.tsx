import { useRef, useState, useCallback } from 'react';

interface DropZoneProps {
  onFileSelect: (file: File) => void;
}

type DragState = 'default' | 'hover' | 'error';

const MAX_SIZE_BYTES = 20 * 1024 * 1024; // 20 MB
const ACCEPTED_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function DropZone({ onFileSelect }: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragState, setDragState] = useState<DragState>('default');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ url: string; name: string; size: string } | null>(null);

  const validateAndSelect = useCallback(
    (file: File) => {
      if (!ACCEPTED_TYPES.has(file.type)) {
        setDragState('error');
        setErrorMessage('Unsupported file type. Please upload PNG, JPEG, or WebP.');
        return;
      }
      if (file.size > MAX_SIZE_BYTES) {
        setDragState('error');
        setErrorMessage(`File exceeds 20 MB limit (${formatBytes(file.size)}).`);
        return;
      }
      setErrorMessage(null);
      setDragState('default');
      setPreview({
        url: URL.createObjectURL(file),
        name: file.name,
        size: formatBytes(file.size),
      });
      onFileSelect(file);
    },
    [onFileSelect],
  );

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragState('hover');
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragState('default');
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragState('default');
    const file = e.dataTransfer.files[0];
    if (file) validateAndSelect(file);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) validateAndSelect(file);
    e.target.value = '';
  };

  const borderClass =
    dragState === 'hover'
      ? 'border-blue-500 bg-blue-500/5'
      : dragState === 'error'
        ? 'border-red-500'
        : 'border-gray-700';

  return (
    <div className="space-y-3">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload image by clicking or dragging a file here"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
        }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`min-h-[300px] rounded-xl border-2 border-dashed transition-colors duration-150 cursor-pointer flex flex-col items-center justify-center gap-4 px-6 py-8 select-none ${borderClass}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="hidden"
          onChange={handleInputChange}
          aria-hidden="true"
        />

        {preview ? (
          <div className="flex flex-col items-center gap-3 w-full" onClick={(e) => e.stopPropagation()}>
            <img
              src={preview.url}
              alt="Selected file preview"
              className="max-h-48 max-w-full rounded-lg object-contain border border-gray-700"
            />
            <div className="text-center">
              <p className="text-sm font-medium text-gray-200 truncate max-w-xs">{preview.name}</p>
              <p className="text-xs text-gray-500 mt-0.5">{preview.size}</p>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setPreview(null);
                setDragState('default');
                setErrorMessage(null);
              }}
              className="text-xs text-blue-400 hover:text-blue-300 underline underline-offset-2 transition-colors"
            >
              Change image
            </button>
          </div>
        ) : (
          <>
            {/* Upload icon */}
            <div
              className={`w-14 h-14 rounded-full flex items-center justify-center transition-colors ${dragState === 'hover' ? 'bg-blue-500/15' : 'bg-gray-800'}`}
            >
              <svg
                className={`w-7 h-7 transition-colors ${dragState === 'hover' ? 'text-blue-400' : 'text-gray-400'}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                />
              </svg>
            </div>

            <div className="text-center space-y-1">
              <p className="text-sm font-medium text-gray-200">
                Drop image here or{' '}
                <span className="text-blue-400 hover:text-blue-300 transition-colors">click to browse</span>
              </p>
              <p className="text-xs text-gray-500">PNG, JPEG, WebP — up to 20 MB</p>
            </div>
          </>
        )}
      </div>

      {errorMessage && (
        <div
          role="alert"
          className="flex items-start gap-2 rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3"
        >
          <svg
            className="w-4 h-4 text-red-400 mt-0.5 shrink-0"
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
          <p className="text-xs text-red-400">{errorMessage}</p>
        </div>
      )}
    </div>
  );
}
