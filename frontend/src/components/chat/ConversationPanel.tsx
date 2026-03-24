import { useEffect, useRef } from 'react';

import { useConversationStore } from '../../stores/conversationStore';
import { usePipelineStore } from '../../stores/pipelineStore';
import { sendChat } from '../../api/coordinator';
import { MessageBubble } from './MessageBubble';
import { ChatInput } from './ChatInput';
import { SuggestionChips } from './SuggestionChips';

const DEFAULT_SUGGESTIONS = [
  'Make protection stronger',
  'Why was this protected?',
  'Show detection details',
];

function TypingIndicator() {
  return (
    <div className="flex gap-2 my-1 flex-row">
      <div
        className="w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-1 text-[10px] font-bold bg-gray-700 text-gray-300"
        aria-hidden="true"
      >
        AI
      </div>
      <div className="flex items-center gap-1 bg-gray-800 rounded-2xl rounded-tl-sm px-3 py-2.5">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
    </div>
  );
}

export function ConversationPanel() {
  const { messages, isTyping, addMessage, setTyping } = useConversationStore();
  const sessionId = usePipelineStore((s) => s.sessionId);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, isTyping]);

  async function handleSend(message: string) {
    if (!sessionId) return;
    addMessage('user', message);
    setTyping(true);

    try {
      const response = await sendChat(sessionId, message);
      addMessage('assistant', response.response_text);

    } catch {
      addMessage('system', 'Could not reach the server. Please try again.');
    } finally {
      setTyping(false);
    }
  }

  function handleSuggestion(suggestion: string) {
    void handleSend(suggestion);
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full bg-gray-900/50">
      {/* Suggestion chips */}
      <div className="border-b border-gray-800 shrink-0">
        <SuggestionChips
          suggestions={DEFAULT_SUGGESTIONS}
          onSelect={handleSuggestion}
        />
      </div>

      {/* Message list */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-3 py-3 space-y-0.5"
        aria-label="Conversation"
        aria-live="polite"
        aria-relevant="additions"
      >
        {isEmpty && !isTyping && (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-center px-4">
            <div className="w-10 h-10 rounded-full bg-gray-800 flex items-center justify-center">
              <svg className="w-5 h-5 text-gray-500" viewBox="0 0 20 20" fill="none" aria-hidden="true">
                <path
                  d="M17 9.5C17 13.09 13.866 16 10 16c-.73 0-1.43-.09-2.09-.26L4 17l1.12-3.32A6.37 6.37 0 013 9.5C3 5.91 6.134 3 10 3s7 2.91 7 6.5z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <p className="text-sm text-gray-500">
              Ask questions about the privacy analysis or request changes to the strategy.
            </p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <MessageBubble
            key={idx}
            role={msg.role}
            content={msg.content}
            timestamp={msg.timestamp}
          />
        ))}

        {isTyping && <TypingIndicator />}
      </div>

      {/* Chat input */}
      <div className="border-t border-gray-800 px-3 py-3 shrink-0">
        <ChatInput
          onSend={handleSend}
          disabled={isTyping || !sessionId}
          placeholder={sessionId ? 'Ask about the analysis...' : 'Start a session to chat'}
        />
      </div>
    </div>
  );
}
