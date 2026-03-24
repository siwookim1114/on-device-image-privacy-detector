import { create } from 'zustand';
import type { ParsedIntent } from '../types/coordinator';

interface Message { role: 'user' | 'assistant' | 'system'; content: string; timestamp: number; }

interface ConversationState {
  messages: Message[];
  isTyping: boolean;
  lastIntent: ParsedIntent | null;
  addMessage: (role: Message['role'], content: string) => void;
  setTyping: (t: boolean) => void;
  setLastIntent: (i: ParsedIntent | null) => void;
  clear: () => void;
}

export const useConversationStore = create<ConversationState>((set) => ({
  messages: [], isTyping: false, lastIntent: null,
  addMessage: (role, content) => set((s) => ({ messages: [...s.messages, { role, content, timestamp: Date.now() }] })),
  setTyping: (t) => set({ isTyping: t }),
  setLastIntent: (i) => set({ lastIntent: i }),
  clear: () => set({ messages: [], isTyping: false, lastIntent: null }),
}));
