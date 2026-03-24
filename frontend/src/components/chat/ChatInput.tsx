import { useState, useRef, type KeyboardEvent, type ChangeEvent } from 'react';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
  placeholder: string;
}

export function ChatInput({ onSend, disabled, placeholder }: ChatInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function handleSend() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleInput(e: ChangeEvent<HTMLTextAreaElement>) {
    setValue(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }

  return (
    <div
      className={[
        'flex items-end gap-2 rounded-lg border px-3 py-2 transition-colors',
        'bg-gray-800 border-gray-700',
        disabled ? 'opacity-60' : 'focus-within:border-gray-600',
      ].join(' ')}
    >
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
        aria-label="Chat message"
        className={[
          'flex-1 bg-transparent text-sm text-gray-200 placeholder-gray-600',
          'resize-none outline-none leading-relaxed',
          'disabled:cursor-not-allowed',
          'min-h-[1.5rem] max-h-[7.5rem]',
        ].join(' ')}
      />

      <button
        type="button"
        onClick={handleSend}
        disabled={disabled || !value.trim()}
        aria-label="Send message"
        className={[
          'shrink-0 w-7 h-7 rounded-md flex items-center justify-center transition-colors',
          'focus:outline-none focus:ring-2 focus:ring-blue-500/50',
          disabled || !value.trim()
            ? 'text-gray-600 cursor-not-allowed'
            : 'text-blue-400 hover:text-blue-300 hover:bg-gray-700',
        ].join(' ')}
      >
        <svg className="w-4 h-4" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path
            d="M14 8L2 2l3 6-3 6 12-6z"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </div>
  );
}
