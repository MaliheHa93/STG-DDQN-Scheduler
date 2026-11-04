# Changes from the earlier SM-SFC package

The revised manuscript no longer describes the security-constrained SM-SFC method. This package is therefore a separate implementation with these substantive changes:

- Removed security levels, security scores, compromised-node events, and security baselines because they do not appear in the revised paper.
- Replaced finite per-VNF jobs with concurrently active streaming SFCs whose CPU and service bandwidth stay reserved for the complete lifetime.
- Replaced one-node actions with complete placement vectors and loop-free ordered service routes.
- Added initial placement, RETAIN, route-only update, and selective-remapping actions.
- Added bounded path/action generation with Pareto ordering and a complete fallback search that prevents false blocking caused by `Mmax`.
- Added sequential per-VNF state transfer, temporary destination reservations, make-before-break cutover, and delayed RL transitions.
- Replaced the flat NumPy network with a PyTorch graph-attention encoder, temporal self-attention, candidate-conditioned Q network, replay, and soft-target Double DQN.
- Replaced old security/deadline plots with traffic acceptance, mobility continuity, migration volume, and scalability runtime panels.
- Restored the manuscript units and ranges: service delay in milliseconds, bandwidth in MB/s, and 80–250 ms latency requirements.

The old archive is intentionally left unchanged so results from the two different research models cannot be mixed.

