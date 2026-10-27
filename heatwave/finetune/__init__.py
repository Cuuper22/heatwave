"""Fine-tuning of Timestral (Mistral, Oct 2026) on cluster power and thermal series.

Modules:
  model     differentiable forecast pass built on the Timestral modules, loss, trainable-parameter modes
            (full, head + last blocks, hand-written LoRA), checkpoint export in the Hub format
  sim       the e003 simulator on the PUBLIC environment (sim-scenario-v3), leakage-safe known-future covariates
  data      training / validation / held-out windows from the simulator, public GPU telemetry, measured laptop runs
            and grid series, with the fixed splits of the project
  train     the training loop (AdamW, warmup + cosine, clipping, early stopping, checkpoint + resume)
  evaluate  zero-shot versus fine-tuned scores on the held-out windows
"""
