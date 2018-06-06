#!/bin/bash
TIMEOUT=120

exec gunicorn -b :5000 --access-logfile - --error-logfile - manage:app --timeout $TIMEOUT
