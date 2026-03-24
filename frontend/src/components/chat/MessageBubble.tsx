interface MessageBubbleProps {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

function formatTime(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** Render text with **bold** and preserved newlines. No external deps. */
function FormattedContent({ text }: { text: string }) {
  const lines = text.split('\n');
  return (
    <>
      {lines.map((line, lineIdx) => {
        // Split on **text** bold markers
        const parts = line.split(/(\*\*[^*]+\*\*)/g);
        return (
          <span key={lineIdx}>
            {parts.map((part, partIdx) => {
              if (part.startsWith('**') && part.endsWith('**')) {
                return (
                  <strong key={partIdx} className="font-semibold">
                    {part.slice(2, -2)}
                  </strong>
                );
              }
              return <span key={partIdx}>{part}</span>;
            })}
            {lineIdx < lines.length - 1 && <br />}
          </span>
        );
      })}
    </>
  );
}

export function MessageBubble({ role, content, timestamp }: MessageBubbleProps) {
  if (role === 'system') {
    return (
      <div className="flex justify-center my-1" role="status" aria-label="System message">
        <div className="text-xs text-gray-500 px-3 py-1 text-center max-w-xs">
          <FormattedContent text={content} />
          <span className="block text-gray-600 text-[10px] mt-0.5">{formatTime(timestamp)}</span>
        </div>
      </div>
    );
  }

  const isUser = role === 'user';

  return (
    <div className={['flex gap-2 my-1', isUser ? 'flex-row-reverse' : 'flex-row'].join(' ')}>
      {/* Avatar dot */}
      <div
        className={[
          'w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-1 text-[10px] font-bold',
          isUser ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300',
        ].join(' ')}
        aria-hidden="true"
      >
        {isUser ? 'U' : 'AI'}
      </div>

      <div className={['flex flex-col gap-0.5 max-w-[80%]', isUser ? 'items-end' : 'items-start'].join(' ')}>
        <div
          className={[
            'rounded-2xl px-3 py-2 text-sm leading-relaxed',
            isUser
              ? 'bg-blue-600 text-white rounded-tr-sm'
              : 'bg-gray-800 text-gray-200 rounded-tl-sm',
          ].join(' ')}
        >
          <FormattedContent text={content} />
        </div>
        <span className="text-[10px] text-gray-600 px-1 tabular-nums">{formatTime(timestamp)}</span>
      </div>
    </div>
  );
}
