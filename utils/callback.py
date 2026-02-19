import re
import sys
from langchain_core.callbacks import BaseCallbackHandler


class Phase2ReviewCallback(BaseCallbackHandler):
    """Callback that prints VLM output during Phase 2 review.

    Shows:
    - Streaming tokens as they arrive (for real-time visibility)
    - Tool calls (name + truncated args)
    - Tool results (truncated)
    """

    def __init__(self):
        self._streaming = False

    def on_llm_new_token(self, token: str, **kwargs):
        """Print each token as it streams in."""
        if token:
            if not self._streaming:
                print("\n  [VLM] ", end="", flush=True)
                self._streaming = True
            print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        """Print tool calls after each LLM generation."""
        if self._streaming:
            print()  # newline after streamed tokens
            self._streaming = False

        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, 'message', None)
                    if not msg:
                        continue

                    # Show tool calls
                    tool_calls = getattr(msg, 'tool_calls', None)
                    if tool_calls:
                        for tc in tool_calls:
                            name = tc.get('name', 'unknown')
                            args = tc.get('args', {})
                            args_short = str(args)
                            if len(args_short) > 150:
                                args_short = args_short[:150] + "..."
                            print(f"  [Tool Call] {name}({args_short})")

        except Exception as e:
            print(f"  [Callback Error] {e}")

    def on_tool_end(self, output, **kwargs):
        """Print truncated tool result."""
        output_str = str(output)
        if len(output_str) > 200:
            output_str = output_str[:200] + "..."
        print(f"  [Tool Result] {output_str}")
