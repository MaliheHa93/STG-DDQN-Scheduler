# Validation status

Validation updated on 12 August 2026:

- Python compilation: passed for the complete package.
- Regression suite: 23 non-neural tests passed, including rejected-decision
  runtime accounting and a curve-only assertion that detects any shaded plot
  collection. The optional PyTorch shape test was skipped in the packaging
  runtime because PyTorch was unavailable; the frozen behavior checkpoint and
  neural path had already been validated when the checkpoint was created.
- Simulator self-test: passed.
- Compact STG-DDQN behavior checkpoint: trained for 10 episodes and frozen with
  SHA-256 `9c903591cc1f52a156d3138083e0e429741eedd452ba41cc2762d1c0aa869899`.
- Paired-seed behavior sweep: 135 measured synthetic runs completed for
  STG-DDQN, IM, and DLAPM: 54 traffic, 45 mobility, and 36 scalability runs,
  using seeds 31, 47, and 73 at every operating point.
- Scalability timing: rerun sequentially with one simulator worker and one
  BLAS/OpenMP thread after all competing experiment processes had ended.
- Raw/summary accounting and strict curve validator: passed with zero duplicate
  run keys, complete paired grids, nondegenerate curves, and nonzero migration.
- Python vector-PDF and 600-dpi PNG generation: passed. The combined PDF was
  rendered with Poppler and the combined and separate plots were visually
  inspected for clipping, legends, backgrounds, curve separation, and absence
  of shaded confidence bands.
- Selective-remapping stress check: completed with repeated make-before-break migrations and no CPU/bandwidth reservation leak.
- ZIP integrity: checked during final packaging.

The behavior curves use synthetic mobility, three seeds, a compact topology, and
10 training episodes. They validate measured behavior and plotting, but are not
final manuscript evidence. The final experiment still requires the SUMO CSV,
full configuration, 15 training episodes, and ten seeds declared through
`configs/paper.yaml`. MATLAB code is included but could not be executed in the
build runtime because MATLAB/Octave was unavailable.
