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

## Combined Recommendations for EMMA
1. **Telemetry Parity:** Extend our experiment logging so every run captures DEQ residual curves, memory collision histograms, and liquid-state norms in a single artifact bundle.
2. **Reference Baselines:** Maintain shallow explicit baselines (stacked residual blocks, GRU memory) to re-run configs when EMMA diverges; this mirrors DEQ and liquid-team validation.
3. **Memory Replay Tools:** Add CLI hooks to `VSAMemory` for slot ablation and replay, enabling Spaun-style debugging when cosine metrics flag issues.
4. **Solver Safety Nets:** Keep spectral norm enforcement plus optional Anderson acceleration fallback in the DEQ solver, and gate long-step experiments behind monitors for iteration growth.
5. **Curriculum Scheduling:** Borrow the LTC teams' sequence-length curriculum—start with shorter contexts before moving to 2048+ tokens to reduce destabilizing gradients.
