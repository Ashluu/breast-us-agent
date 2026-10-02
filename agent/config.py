"""P3 agent configuration -- checkpoint paths, manifest paths, orchestrator model name.

Kept as one small module (not a YAML like the P1/P2 configs under configs/) because this
phase has far fewer knobs than a training run: there is nothing here to sweep, only "which
checkpoint/manifest am I probing right now". If that changes (e.g. comparing orchestrator
models becomes its own experiment), promote this to YAML + argparse the way
bjepa/train_ssl.py does -- don't grow this file ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    # Orchestrator LLM -- served locally via Ollama, no API cost (P3_AGENT_NOTES.md
    # constraint). Local derivative of the community build
    # `oamazonasgabriel/lfm2.5-2.6b:q4_k_m-8gbGPU` (the hf.co/... tag does not exist). The
    # community Modelfile sets `stop` to bare "tool_call_start" etc., which cuts generation
    # as soon as the model begins a tool call, so tool_calls always came back empty. Rebuild with:
    #   printf 'FROM oamazonasgabriel/lfm2.5-2.6b:q4_k_m-8gbGPU\nPARAMETER stop <|im_end|>\n' > Modelfile
    #   ollama create lfm2.5-2.6b-agent -f Modelfile
    ollama_model: str = "lfm2.5-2.6b-agent"
    ollama_base_url: str = "http://localhost:11434"

    # Vision encoders -- same architecture/loading path as bjepa/probe.py, so a checkpoint
    # written by train_ssl.py drops in here unchanged.
    encoder_name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k"
    image_size: tuple[int, int] = (224, 224)

    # Encoder-only copy of the thesis v3 ultrasound checkpoint (keys: encoder, cfg), 343 MB.
    # Not in git: download it from the Hugging Face Hub, see README "Weights".
    us_checkpoint: str = "checkpoints/us_ijepa_v3_encoder.pt"
    mg_checkpoint: str = "checkpoints/p1_mm_ijepa_armC.pt"   # present locally, confirmed 2026-09-20

    us_manifest: str = "manifests/bus_bra.csv"
    mg_manifest: str = "manifests/ddsm.csv"      # confirmed 2026-09-20: this is the one with image_path/risk/split

    train_split: str = "train"
    val_split: str = "validation"


CONFIG = AgentConfig()
