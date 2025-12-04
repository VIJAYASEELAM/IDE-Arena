#!/bin/bash
if [ -n "$1" ]; then
    echo "Running Task $1..."
    npx jest /app/tasks/$1/task_tests.js
else
    echo "Running Project Tests..."
    npm test
fi