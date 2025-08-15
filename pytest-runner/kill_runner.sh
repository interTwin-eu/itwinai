#!/bin/bash

stop_pid() {
  local name=$1
  local file=$2

  if [ -f "$file" ]; then
    local pid
    pid=$(cat "$file")
    if ps -p "$pid" > /dev/null 2>&1; then
      kill "$pid" && echo "Stopped $name (PID: $pid)"
    else
      echo "$name not running."
    fi
    rm -f "$file"
  else
    echo "No $name PID file found."
  fi
}

stop_pid "Runner.Listener" "runner-listener.pid"
stop_pid "helper script" "runner-helper.pid"