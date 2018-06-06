#!/bin/bash

source sms-env/bin/activate
TIMEOUT=120

while true; do
    flask deploy
    if [[ "$?" == "0" ]]; then
        break
    fi
    echo Deploy command failed, retrying in 5 secs...
    sleep 5
done

exec gunicorn -b :8080 --access-logfile - --error-logfile - manage:app --timeout $TIMEOUT
