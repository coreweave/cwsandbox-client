# SPDX-FileCopyrightText: 2025 CoreWeave, Inc.
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-PackageName: cwsandbox-client

"""Transcribe an audio file with Qwen3-ASR on a single-GPU sandbox.

Usage:
    python examples/audio_transcription.py path/to/audio.wav
"""

import sys
from pathlib import Path

from cwsandbox import ResourceOptions, Sandbox

IMAGE = "qwenllm/qwen3-asr@sha256:fb75b775f089e06e5a1aaebffd421e37505cc630d50c86d889d95ffa45a7e16a"

TRANSCRIBE = """
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0"
)
print(model.transcribe(audio="/tmp/audio.wav")[0].text)
"""


def main() -> None:
    audio = Path(sys.argv[1]).read_bytes()
    with Sandbox.run(
        container_image=IMAGE,
        resources=ResourceOptions(
            requests={"cpu": "8", "memory": "32Gi"},
            limits={"cpu": "8", "memory": "32Gi"},
            gpu={"count": 1, "type": "rtxp6000-8x"},
        ),
    ) as sandbox:
        sandbox.write_file("/tmp/audio.wav", audio).result()
        result = sandbox.exec(["python3", "-c", TRANSCRIBE], check=True).result()
        print(result.stdout.strip())


if __name__ == "__main__":
    main()
