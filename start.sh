#!/bin/bash

mkdir -p /src/flux-cache/public

if [ ! -f /src/flux-cache/model_index.json ]; then
    python /src/scripts/download_hf_model.py
fi

python -m cog.server.http