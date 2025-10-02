# External Teams Using EMMA-Adjacent Architectures

This note summarizes how other research groups have deployed components similar to EMMA's DEQ backbone, VSA memory, and liquid/continuous-time cells. It also distills their debugging practices so we can map them onto our workflow.

## Deep Equilibrium Models (DEQs)
- **Key group:** Carnegie Mellon University & Facebook AI Research (Bai, Kolter, Koltun, *Deep Equilibrium Models*, NeurIPS 2019).
- **Build practices:**
  - Used Broyden's method with Anderson acceleration fallbacks to solve the fixed-point layer; weights maintained spectral norm constraints to guarantee convergence.
  - Warm-started each forward pass with the previous sequence state when operating on videos and language to reduce solver iterations.
- **Debug tactics reported in follow-up repos/papers:**
  - Log the residual norm `||f(z) - z||` per iteration, early stopping on stagnation.
  - Track the number of solver iterations and clamp maximum steps to avoid exploding costs; visualize residual curves to catch divergence.
  - Maintain a lightweight explicit network during bring-up to compare logits against the DEQ output and isolate implicit-layer bugs.
- **Implications for EMMA:** continue recording residual norms and `avg_fp_iters`, and build parity tests against a truncated explicit stack when introducing new nonlinearities.

- **Key group:** University of Washington & NVIDIA (Chen et al., *Implicit Vision Transformers*, CVPR 2022).
- **Build practices:**
  - Hybridized DEQ solvers with layer-wise residual adapters to stabilize large-patch ViT inputs.
  - Cached Jacobian-vector products from adjacent layers to amortize backward-pass costs during high-resolution training.
  - Deployed mixed-precision fixed-point solves but forced fp32 accumulation within Broyden steps to prevent NaN cascades.
- **Debug tactics:**
  - Instrumented solver restarts and flagged any batch element that exceeded two restarts for targeted sample inspection.
  - Compared implicit-layer gradients against explicit unrolled transformers on a nightly basis via finite-difference probes.
  - Logged per-frequency error spectra to ensure the implicit solver preserved high-frequency details during diffusion-style fine-tuning.
- **Implications for EMMA:** we should add gradient-consistency tests between the DEQ module and a 4-layer explicit reference, and trace restart frequencies to automatically quarantine problematic sequences.

- **Key group:** University of Toronto Vector Institute (Geng et al., *Deep Equilibrium Graph Neural Networks*, ICLR 2021).
- **Build practices:**
  - Constrained graph Laplacian eigenvalues via spectral normalization to secure fixed-point convergence on large meshes.
  - Used continuation methods—starting from shallow unrolled GNNs and gradually tightening implicit solves—to avoid local minima.
- **Debug tactics:**
  - Visualized per-node residual heatmaps to localize structural bottlenecks.
  - Ran sinkhorn-normalized gradient checks to ensure message-passing symmetry, catching bugs caused by graph batching.
- **Implications for EMMA:** adopt residual heatmap tooling for long-context tokens to flag sections of the sequence that fail to equilibrate, and explore continuation schedules when rolling out new DEQ nonlinearities.

## Vector Symbolic Architecture (VSA) Memory
- **Key groups:** University of Waterloo's Centre for Theoretical Neuroscience & Applied Brain Research (Eliasmith et al., *Spaun* model, Science 2012); Osborn & Plate (*Distributed Ternary VSA*, Frontiers 2020).
- **Build practices:**
  - Bucketed memory with cleanup (auto-associative) layers to denoise retrieved symbols.
  - Norm clipping and re-normalization after each write to keep hypervectors on the unit hypersphere.
  - Logging collision statistics and slot occupancy to tune binding dimensionality.
- **Debug tactics:**
  - Replay memory traces to inspect when collisions cause retrieval failure.
  - Plot cosine similarity distributions between stored vectors and probes to detect drift.
  - Use ablation writes (zeroing specific slots) to confirm address disentanglement.
- **Implications for EMMA:** our existing bucket entropy and `write_cos` metrics mirror these practices; we should add replay/debug utilities that emit per-slot cosine histograms and support slot-ablation tests inside `VSAMemory`.

- **Key group:** Sandia National Laboratories (Rasmussen et al., *Hyperdimensional Computing for Real-Time Signal Analysis*, IEEE Transactions on Emerging Topics 2021).
- **Build practices:**
  - Employed streaming VSA memories with sliding-window binding to support continuous sensor feeds.
  - Quantized hypervectors to 8-bit ternary representations for FPGA deployment, monitoring Hamming distance to track degradation.
  - Integrated adaptive cleanup thresholds driven by online collision statistics.
- **Debug tactics:**
  - Captured bit-flip histograms across hardware runs to catch failing memory banks early.
  - Replayed windowed signals through a CPU reference implementation each night for regression testing.
  - Benchmarked retrieval quality as a function of dimensionality to set safe deployment margins.
- **Implications for EMMA:** incorporate nightly CPU-vs-GPU regression jobs for `VSAMemory`, and expose hooks for ternary/quantized experiments so we can port memories into edge-serving prototypes.

- **Key group:** University of Amsterdam & Google DeepMind (Komer et al., *Neural Program Search with VSA*, arXiv 2023).
- **Build practices:**
  - Bound program tokens into composite hypervectors with learned role/filler decompositions to reduce superposition errors.
  - Applied entropy-regularized attention to gate memory writes, effectively learning sparsity patterns.
- **Debug tactics:**
  - Logged attention entropy and write sparsity to diagnose runaway writes.
  - Performed targeted probing by binding diagnostic vectors that should map to known functions—failures highlighted corruption.
- **Implications for EMMA:** extend our logging to include write-mask entropy and create diagnostic probes (e.g., identity, shift, negate) to detect corruption during curriculum experiments.

## Liquid / Continuous-Time Cells
- **Key groups:** MIT CSAIL & Austrian Institute of Technology (Hasani et al., *Liquid Time-Constant Networks*, Nature Machine Intelligence 2021); ETH Zürich & NVIDIA (Gu et al., *Combining Recurrent Neural Networks and Neural ODEs*, ICML 2020).
- **Build practices:**
  - Parameterize ODE-based neurons with learnable time constants and integrate them with adaptive-step solvers (Dormand–Prince, Tsit5) for stiff dynamics.
  - Apply curriculum training from short to long sequences to keep gradients stable.
  - Clamp or regularize time constants to avoid negative values and catastrophic oscillations.
- **Debug tactics:**
  - Monitor hidden-state trajectories and their derivatives to verify solver stability.
  - Record solver step counts and reject-rate statistics when using adaptive integrators.
  - Compare continuous-time outputs against discrete GRU/LSTM baselines to confirm correctness.
- **Implications for EMMA:** extend logging around the `LiquidCell` to capture hidden-state norms over time, and introduce comparison runs versus a GRU fallback when tweaking the cell dynamics.

- **Key group:** Stanford University & Google Research (Rubanova et al., *Latent ODEs for Irregularly-Sampled Time Series*, NeurIPS 2019).
- **Build practices:**
  - Leveraged adjoint-based backpropagation through continuous-time latent dynamics with adaptive solvers (Dopri5).
  - Initialized continuous latent states from encoder posteriors conditioned on sparse observations to reduce burn-in instability.
- **Debug tactics:**
  - Compared adjoint gradients with truncated backprop through time on synthetic datasets to ensure parity.
  - Monitored KL divergence trends to verify latent encoders remained calibrated when solver tolerances changed.
- **Implications for EMMA:** add adjoint-vs-truncated gradient checks for the liquid cell when we toggle solver tolerances, and track latent regularization metrics alongside solver stats.

- **Key group:** University of Maryland & NASA FDL (Rusch et al., *Liquid Time-Constant Networks for Adaptive Control*, AIAA 2022).
- **Build practices:**
  - Deployed LTCs inside control loops with hard real-time deadlines, forcing deterministic solver step caps.
  - Mixed implicit Euler steps with learned damping terms to guarantee passivity in physical systems.
- **Debug tactics:**
  - Verified Lyapunov stability numerically at each training checkpoint.
  - Collected per-trajectory Jacobian eigenvalues to detect emerging chaotic behavior.
- **Implications for EMMA:** we should add optional stability-check routines (e.g., eigenvalue scans) for long-horizon planning experiments and allow deterministic stepping modes when integrating with simulators.

- **Key group:** University of Tübingen & Bosch Center for AI (Bittner et al., *Neural Controlled Differential Equations*, ICLR 2021).
- **Build practices:**
  - Modeled control signals as signatures and fed them to neural CDEs trained with log-ODE integration.
  - Used rough-path data augmentation to generalize across irregular sampling rates.
- **Debug tactics:**
  - Ran signature truncation ablations to observe the effect on trajectory reconstruction accuracy.
  - Inspected solver tolerance sweeps to confirm stability margins before deployment in robotics benches.
- **Implications for EMMA:** incorporate tolerance-sweep experiments into our CI for liquid components and investigate signature-inspired augmentations for multimodal sensor fusion tasks.

## Cross-Cutting Tooling from External Teams
- **Unified dashboards:** Both the DEQ and liquid-model communities lean on WandB/Neptune dashboards with custom panels for solver metrics. We should curate a shared EMMA dashboard template that bundles residuals, solver steps, memory collisions, and latent norms in one view.
- **Reproducible notebooks:** Several teams (e.g., Kolter lab, Vector Institute) maintain Colab-style notebooks that replay core debugging scenarios. Cloning this approach for EMMA would let new contributors step through fixed-point failures, memory collisions, and solver tolerance sweeps quickly.
- **Automated alerts:** Hardware-focused VSA groups emit alerts when collision rates or Hamming distances exceed thresholds. Implementing similar alert hooks in our experiment runner would shorten the feedback loop when remote jobs diverge.

## Next Experiments for EMMA
1. Stand up a nightly regression matrix that pairs DEQ implicit solves with explicit baselines, logging gradient discrepancies and restart counts.
2. Prototype ternary and quantized VSA writes using a CPU reference to validate cleanup robustness before porting to accelerators.
3. Build an adjoint-gradient parity test harness for `LiquidCell`, sweeping solver tolerances and logging stability metrics.
4. Ship a unified monitoring dashboard template and alerting thresholds for all implicit components, borrowing layout ideas from the external teams cited above.

## Combined Recommendations for EMMA
1. **Telemetry Parity:** Extend our experiment logging so every run captures DEQ residual curves, memory collision histograms, and liquid-state norms in a single artifact bundle.
2. **Reference Baselines:** Maintain shallow explicit baselines (stacked residual blocks, GRU memory) to re-run configs when EMMA diverges; this mirrors DEQ and liquid-team validation.
3. **Memory Replay Tools:** Add CLI hooks to `VSAMemory` for slot ablation and replay, enabling Spaun-style debugging when cosine metrics flag issues.
4. **Solver Safety Nets:** Keep spectral norm enforcement plus optional Anderson acceleration fallback in the DEQ solver, and gate long-step experiments behind monitors for iteration growth.
5. **Curriculum Scheduling:** Borrow the LTC teams' sequence-length curriculum—start with shorter contexts before moving to 2048+ tokens to reduce destabilizing gradients.
