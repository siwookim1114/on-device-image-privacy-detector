interface SuggestionChipsProps {
  suggestions: string[];
  onSelect: (suggestion: string) => void;
}

export function SuggestionChips({ suggestions, onSelect }: SuggestionChipsProps) {
  if (suggestions.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2 px-3 py-2" role="list" aria-label="Quick suggestions">
      {suggestions.map((suggestion) => (
        <button
          key={suggestion}
          type="button"
          role="listitem"
          onClick={() => onSelect(suggestion)}
          className={[
            'px-3 py-1 rounded-full text-sm text-gray-300 transition-colors',
            'bg-gray-800 hover:bg-gray-700 border border-gray-700 hover:border-gray-600',
            'focus:outline-none focus:ring-2 focus:ring-blue-500/40',
          ].join(' ')}
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
}
