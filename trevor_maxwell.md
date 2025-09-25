# Trevor Maxwell — notes
- yyyy-mm-dd: Initial CPU runs; L=256 assisted stabilized at ~0.89; strict 512 micro-runs promising.
- yyyy-mm-dd: Added diagnostics (addr entropy, residual); preparing 2048 configs.
- yyyy-mm-dd: Implemented confidence-gated writes + adaptive oracle mix; L=512 CPU sweeps with mem τ ≥ 0.30 deliver 0.94 post-warm, write_gate_ratio ≈ 0.33, lift-off epoch 1.
- yyyy-mm-dd: Soft-floor adaptive mix sweep (k0 ∈ {0.4,0.5,0.6}, k1 ∈ {0.8,1.0,1.2}) holds 0.94 post-warm with read_cos 0.47–0.69; 2048-token quick test started (partial due to walltime).
