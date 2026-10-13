# Research and copied implementations

| Reference | What this lab uses | What it does not establish |
|---|---|---|
| [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464), ICLR 2025 | Gated recurrence and the motivation for separating input processing from fixed-size state | CPU speed, this shared-loop variant's quality, or full language-model results |
| [Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927), ICLR 2024 | Multi-query associative recall; score only retrieval queries; test input lengths separately | General language performance, large-vocabulary recall or a reproduction of all paper experiments |

The primary papers were retrieved with alphaXiv. The Zoology venue was also checked against its [official published PDF](https://openreview.net/pdf/80fadc4fc3c3c1a12600b7f7e0e3d13e04ac334b.pdf). Paper methods guide the experiment; local measurements establish this implementation's results.

This recipe uses vocabulary 128, two key/value pairs, no random distractors, a uniform query-gap distribution and 16-token training inputs. Held-out lengths 16, 32, 64 increase the gap without increasing the number of stored pairs. This is a small, easier MQAR condition, not a test of recall capacity as the number of facts grows. Initialization seeds vary; synthetic data sets are fixed independently. Exact input/label row overlap is checked before training.

One and two loops have the same unique parameter count and identical initial parameter bytes for each seed. Two loops perform more block work and retain separate states. This is a parameter-matched comparison, not a compute-matched comparison. The final epoch is fixed before training; validation does not choose an epoch or configuration.

## Source identities and licenses

- FLA: [`1fe52b5d05c53601885610fab3d39157647d836d`](https://github.com/fla-org/flash-linear-attention/tree/1fe52b5d05c53601885610fab3d39157647d836d), MIT. Original source and [manifest](../src/recurrent_state_lab/_vendor/fla/manifest.json) are included. NVIDIA's separately licensed research implementation is not copied.
- Zoology: [`1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`](https://github.com/HazyResearch/zoology/tree/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb), Apache 2.0. Original source and [manifest](../src/recurrent_state_lab/_vendor/zoology/manifest.json) are included.
- New adapters, CLI, checks and report are MIT. Both upstream licenses remain in the distribution; copied bodies retain their original terms.

The selected original recurrence, convolution, RMS, gated RMS and gate functions were executed before composing the CPU layer. Original MQAR labels were checked against the earlier key/value dictionary, and the original embedding/MLP ran with finite gradients. No full upstream GPU test suite was executed.
