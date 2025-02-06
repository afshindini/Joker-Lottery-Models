#!/bin/bash -l
set -e
if [ "$#" -eq 0 ]; then
  exec joker_lottery_models --help
else
  exec "$@"
fi
