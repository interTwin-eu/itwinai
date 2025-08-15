#!/bin/bash

check_pid() {
  local name=$1
  local file=$2

  if [ -f "$file" ]; then
    local pid
    pid=$(cat "$file")
    if ps -p "$pid" > /dev/null 2>&1; then
      echo "$name is running (PID: $pid)"
      return 0
    else
      echo "$name not running."
      return 1
    fi
  else
    echo "No $name PID file found."
    return 1
  fi
}

# Check for the helper script existence
check_pid "Helper script" "runner-helper.pid"
helper_status=$?

# Check for the listener script existence
check_pid "Runner.Listener" "runner-listener.pid"
listener_status=$?

# Return exit status
exit $(( helper_status != 0 || listener_status != 0 ))