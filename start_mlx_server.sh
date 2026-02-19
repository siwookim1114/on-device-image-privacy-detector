#!/bin/bash
# Start the MLX VLM server for Phase 2 risk assessment

MODEL_PATH="./Qwen3-VL-30B-A3B-Thinking-4bit"
PORT=8000
CONDA_ENV="lab_env"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH"
    echo "Run the quantization first:"
    echo "  python3 -m mlx_vlm convert \\"
    echo "    --hf-path Qwen/Qwen3-VL-30B-A3B-Thinking \\"
    echo "    --mlx-path $MODEL_PATH \\"
    echo "    -q --q-bits 4"
    exit 1
fi

# Check if something is already LISTENING on this port (not just any connection)
if lsof -iTCP:$PORT -sTCP:LISTEN > /dev/null 2>&1; then
    echo "Port $PORT is already in use. Server might already be running."
    echo "To kill it: kill \$(lsof -t -iTCP:$PORT -sTCP:LISTEN)"
    exit 1
fi

echo "Starting MLX VLM server..."
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"
echo "  Host: 127.0.0.1 (localhost only)"
echo "  Features: Vision + Tool Calling (Hermes parser)"
echo "  Stop with: Ctrl+C"
echo ""

conda run -n "$CONDA_ENV" vllm-mlx serve "$MODEL_PATH" --port $PORT \
  --host 127.0.0.1 \
  --enable-auto-tool-choice --tool-call-parser hermes
