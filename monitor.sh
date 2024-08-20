#!/bin/bash

while true; do
    if ! pgrep -f "main_bot_script.py" > /dev/null; then
        echo "Bot not running, restarting..."
        python3 main_bot_script.py &  
    fi
    sleep 60  # Check every 60 seconds
done
