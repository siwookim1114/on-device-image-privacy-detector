#!/bin/bash
# Start llama-server for Phase 2 VLM risk assessment
# Uses llama.cpp with native tool calling via --jinja

PORT=8080
LLAMA_SERVER="./llama.cpp/build/bin/llama-server"
MODEL="./models/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf"
MMPROJ="./models/mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf"

# Check if llama-server binary exists
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "llama-server not found at $LLAMA_SERVER"
    echo "Build first:"
    echo "  cd llama.cpp"
    echo "  cmake -B build -DGGML_METAL=ON"
    echo "  cmake --build build --config Release -j\$(sysctl -n hw.ncpu)"
    exit 1
fi

# Check if model files exist
if [ ! -f "$MODEL" ]; then
    echo "Model not found at $MODEL"
    echo "Download first:"
    echo "  huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf --local-dir ./models"
    exit 1
fi

if [ ! -f "$MMPROJ" ]; then
    echo "Vision projector not found at $MMPROJ"
    echo "Download first:"
    echo "  huggingface-cli download Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf --local-dir ./models"
    exit 1
fi

# Check if something is already LISTENING on this port (not just any connection)
if lsof -iTCP:$PORT -sTCP:LISTEN > /dev/null 2>&1; then
    echo "Port $PORT is already in use. Server might already be running."
    echo "To kill it: kill \$(lsof -t -iTCP:$PORT -sTCP:LISTEN)"
    exit 1
fi

echo "Starting llama-server..."
echo "  Model: Qwen3-VL-30B-A3B-Instruct Q4_K_M"
echo "  Vision: mmproj-F16"
echo "  Port: $PORT"
echo "  Features: Vision + Tool Calling (--jinja)"
echo "  GPU: Metal (all layers offloaded)"
echo "  Stop with: Ctrl+C"
echo ""

$LLAMA_SERVER --jinja --flash-attn on \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  -ngl 99 \
  -c 32768 \
  --host 127.0.0.1 \
  --port $PORT
