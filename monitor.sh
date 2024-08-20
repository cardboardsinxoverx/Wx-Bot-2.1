#!/bin/bash

while true; do
    if ! pgrep -f "your_bot_script.py" > /dev/null; then
        echo "Bot not running, restarting..."
        python3 your_bot_script.py &  # Adjust the command as needed
    fi
    sleep 60  # Check every 60 seconds
done