#!/bin/bash

# Function to start server and wait for it to be ready
start_server() {
    local model=$1
    local alias=$2
    local port=$3
    local log_file="$LLAMA_LOG_DIR/server_${port}.log"

    echo "Starting server with model: $model"
    echo "Using CUDA devices: $CUDA_DEVICES"
    echo "Log file: $log_file"

    sleep 1

    mkdir -p "$LLAMA_LOG_DIR"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$LLAMA_SERVER" \
        -m "$model" \
        -a "$alias" \
        -t 16 \
        -ngl 999 \
        -n 72000 \
        --ctx-size 72000 \
        -np 1 \
        --temp 0.6 \
        --port "$port" \
        --jinja > "$log_file" 2>&1 &
        
        # \
        # >> "$log_file" 2>&1 &

    local server_pid=$!
    echo "Server PID: $server_pid"

    sleep 5

    #local url="http://localhost:${port}/v1/chat/completions"
    local url="http://127.0.0.1:${port}/v1/chat/completions"

    local start_time
    start_time=$(date +%s)
    local timeout=${LLAMA_TIMEOUT:-300}
    local health_prompt="ping"

    echo "Waiting for llama-server to be ready..."
    local attempts=0

    while true; do
        attempts=$((attempts + 1))

        # bail if server died
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Server process died unexpectedly"
            tail -n 20 "$log_file" 2>/dev/null || echo "No log file found"
            return 1
        fi

        # capture body + status on separate lines
        local raw
        
        #raw=$(timeout 10 curl -s -w '\n%{http_code}' -X POST "$url" \
        raw=$(timeout 10 curl --noproxy "localhost,127.0.0.1,::1" -s -w '\n%{http_code}' -X POST "$url" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"default\",
                \"messages\":[{\"role\":\"user\",\"content\":\"$health_prompt\"}],
                \"max_tokens\": 16
            }" 2>/dev/null || echo -e "\n000")

        local http_code body
        http_code=${raw##*$'\n'}
        body=${raw%$'\n'$http_code}

        # basic ready condition: HTTP 200 + non-empty body
        if [[ "$http_code" == "200" && -n "$body" ]]; then
            # try to extract content if this is OpenAI-style JSON
            local content trimmed
            content=$(echo "$body" | jq -r '.choices[0].message.content // empty' 2>/dev/null || echo "")
            trimmed=$(echo "$content" | xargs)

            # if jq works and we got a real answer, optionally require != prompt
            if [[ -n "$trimmed" && "$trimmed" != "null" && "$trimmed" != "$health_prompt" ]]; then
                echo "llama-server is fully ready (content: '$trimmed')"
                break
            fi

            # fallback: treat as ready if body is non-empty even if jq failed
            if [[ -z "$content" ]]; then
                echo "llama-server is fully ready (non-empty body, jq failed or different schema)"
                break
            fi
        fi

        if (( attempts % 10 == 0 )); then
            echo "Still waiting... (attempt $attempts, HTTP $http_code)"
            echo "Last body (truncated): '$(echo "$body" | head -c 200)'"
        fi

        local now
        now=$(date +%s)
        if (( now - start_time > timeout )); then
            echo "Timed out waiting for llama-server after ${timeout} seconds"
            echo "Server logs:"
            tail -n 50 "$log_file" 2>/dev/null || echo "No log file found"
            kill "$server_pid" 2>/dev/null || true
            return 1
        fi

        sleep 3
    done

    echo "Llama-server ready!"
    export LLAMA_SERVER_PID=$server_pid
    return 0
}

# Function to kill server
kill_server() {
    local pid=$1
    local port=$2

    echo "Stopping server processes on port $port..."
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        wait "$pid" 2>/dev/null
        echo "Server $pid killed successfully"
    else
        echo "Server process $pid not found or already terminated"
    fi
}