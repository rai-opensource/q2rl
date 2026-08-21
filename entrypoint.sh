#!/bin/bash
git config --global --add safe.directory /workspace

if [ ! -f "/root/.setup_done" ]; then
    echo "Installing dependencies..."
    uv sync
    uv pip install -e /opt/mj_envs
    touch /root/.setup_done
fi

source /opt/venv/bin/activate
exec "${@:-bash}"
