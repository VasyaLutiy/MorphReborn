#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <number_of_subprocesses>"
  exit 1
fi

num_subprocesses=$1

for ((i=1; i<=num_subprocesses; i++)); do
  python ollama_cli.py &
done

wait
