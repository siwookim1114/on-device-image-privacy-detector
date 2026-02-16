import re
from langchain_core.callbacks import BaseCallbackHandler


class Phase2ReviewCallback(BaseCallbackHandler):
    """Callback that prints concise VLM output during Phase 2 review.

    Shows only:
    - Tool calls (name + truncated args)
    - Tool results (truncated)
    - VLM thinking (truncated summary)
    - VLM text responses
    """

    def on_llm_end(self, response, **kwargs):
        """Print thinking summary and tool calls after each LLM generation."""
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, 'message', None)
                    if not msg:
                        continue

                    # Show reasoning/thinking (truncated)
                    extra = getattr(msg, 'additional_kwargs', {}) or {}
                    reasoning = extra.get('reasoning_content', '')
                    if reasoning and reasoning.strip():
                        display = reasoning.strip()
                        display = display[:300] + "..." if len(display) > 300 else display
                        print(f"\n  [Thinking] {display}")

                    # Show text content (not thinking blocks)
                    content = (msg.content or "").strip()
                    if content:
                        # Strip <think> blocks
                        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                        if think_match:
                            thinking = think_match.group(1).strip()
                            visible = content[think_match.end():].strip()
                            if thinking and not reasoning:
                                display = thinking[:300] + "..." if len(thinking) > 300 else thinking
                                print(f"\n  [Thinking] {display}")
                            if visible:
                                display = visible[:300] + "..." if len(visible) > 300 else visible
                                print(f"  [VLM] {display}")
                        else:
                            display = content[:300] + "..." if len(content) > 300 else content
                            print(f"\n  [VLM] {display}")

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
