# EMMA Roadmap (Medium-Term)

_Last revised: 2025-09-21_

## Goals (Next 6–8 Weeks)
1. **Stabilize Training Loop**
   - Extract additional schedules (oracle mix, mem-injection ramp) into modular policies.
   - Add regression tests covering scheduler metrics and warm-start behavior.
2. **Expand Evaluation Probes**
   - Introduce configurable probe suites (addr entropy histograms, read/write cosine tracking, bucket coverage plots).
   - Automate probe execution post-run and save under `experiments/results/probes/`.
3. **Broaden Hardware Support**
   - Decide on GPU path: prototype GPU-backed VSA read/write (no-grad) or explicitly document CPU-only strategy.
   - Profile DEQ iterations on MPS/GPU and tune defaults accordingly.
4. **Documentation & Onboarding**
   - Convert architecture diagrams to lightweight web versions for quick review.
   - Add “quick experiment” tutorial covering config overrides and probe activation.
5. **Automation / CI**
   - Run smoke tests (`pytest tests`) in CI or via pre-commit shorthand.
   - Add linting/formatting (black, isort) if desired.

## Milestones
- **M1 (2 weeks)**: Scheduler refactor complete, tests passing, roadmap entries updated.
- **M2 (4 weeks)**: Probe suite available with CLI wrapper; documentation updated.
- **M3 (6 weeks)**: Hardware decision documented, GPU path prototype evaluated.
- **M4 (8 weeks)**: CI running smoke tests on every push/hand-off.

## Workstreams & Owners
- _Training Loop_: core maintainers (define policies, ensure tests).
- _Probes & Analysis_: whoever owns evaluation / writing.
- _Infrastructure_: contributor with access to CI / automation.

Update this roadmap as deliverables land or priorities shift.
