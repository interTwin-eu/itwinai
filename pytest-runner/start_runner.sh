#!/bin/bash

LOGFILE="log"
RUNNER_DIR="/p/project1/intertwin/krochak1/actions-runner"
HELPER="$RUNNER_DIR/run.sh"

# launch the runner, get PID and write it file
rm -f "$LOGFILE"
nohup "$HELPER" > "$LOGFILE" 2>&1 &
HELPER_PID=$!
echo $HELPER_PID > runner-helper.pid
disown

# Wait briefly to allow Runner.Listener to start
sleep 2

# Get the PID of Runner.Listener via pgrep
LISTENER_PID=$(pgrep -f "$RUNNER_DIR/bin/Runner.Listener run")
if [ -n "$LISTENER_PID" ]; then
  echo "$LISTENER_PID" > runner-listener.pid
else
  echo "ERROR: Runner.Listener process not found."
  rm -f runner-helper.pid
  exit 1
fi

echo "Runner started."
echo "Helper PID: $HELPER_PID"
echo "Listener PID: $LISTENER_PID"