---
license: other
tags: [medical-imaging, ultrasound, self-supervised, i-jepa, vit]
---

# us-ijepa-v3-encoder

I-JEPA ViT-B/16 encoder for breast ultrasound, from an MSc thesis. Encoder weights only
(`encoder` state dict and the training `cfg`), 343 MB, float32. Research use; not a
clinical tool. Unpublished work: please do not redistribute.

- Input: 224 x 224 grayscale ultrasound, repeated to 3 channels, scaled to [0, 1].
- Output: patch tokens `[B, 196, 768]`; the thesis probes use the mean over patches.
- Frozen linear probe, BUS-BRA (patient-grouped split): validation AUROC 0.814, test 0.870
  on whole frames.

Used by the agent repository, which loads it from `checkpoints/us_ijepa_v3_encoder.pt`.
