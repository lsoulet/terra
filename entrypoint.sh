#!/bin/sh
set -e

ray start --head --dashboard-host=0.0.0.0 --port=6379

exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
