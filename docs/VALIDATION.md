# Validation and measured scope

Validated locally on 2026-10-14 JST using Windows 11, an AMD Ryzen AI 7 350 CPU and two PyTorch threads. Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.5.4, einops 0.8.1 and psutil 7.2.2. The runtime was installed from the built wheel and imported outside the checkout. A fresh-machine dependency installation and Linux/GPU execution remain unverified.

## Checks

- 176 tests passed with no failures, errors or skips: 151 backend/state/gradient/dispatch checks and 25 data/model/CLI/report checks.
- Original FLA recurrence, convolution, gate and RMS references ran before composition; original Zoology data, embedding and MLP bodies ran before the learned experiment.
- All 20 runtime source files in the installed package match the candidate. Copied source files match the fixed upstream manifests.
- Four default training runs completed. Repeating from the final installed runtime reproduced all unique parameter bytes, every recorded training-curve value and all held-out scores exactly. The initial run predates an admission-guard addition; it is not used as the final-code execution receipt.
- Each tensor checkpoint was loaded with `weights_only=True` and strict state loading. All three held-out scores for every checkpoint were evaluated again and matched.

## Held-out recall

The recipe is fixed before training: 12 epochs, 1024 training sequences, 128 validation sequences, 256 held-out sequences at each length, two queries per sequence, seeds 7/19 and one/two shared loops. Models train at length 16. These are separate input/label rows, not independently collected real-world data.

| Seed | Shared loops | Parameters | 16 tokens | 32 tokens | 64 tokens |
|---|---|---|---|---|---|
| 7 | 1 | 28,292 | 434/512 (84.77%) | 441/512 (86.13%) | 454/512 (88.67%) |
| 7 | 2 | 28,292 | 500/512 (97.66%) | 499/512 (97.46%) | 497/512 (97.07%) |
| 19 | 1 | 28,292 | 502/512 (98.05%) | 506/512 (98.83%) | 504/512 (98.44%) |
| 19 | 2 | 28,292 | 498/512 (97.27%) | 498/512 (97.27%) | 501/512 (97.85%) |

Two loops improve seed 7 but are lower than one loop for seed 19. Two initialization seeds do not establish statistically reliable superiority. The unchanged number of key/value pairs makes this a length/gap check, not a recall-capacity comparison.

## State and latency

The actual returned tensor storage for one sequence is 4608 bytes with one loop and 9216 bytes with two loops at every tested length. Two-sequence controls use 9216/18432 bytes. These figures exclude weights, full-sequence outputs, training graphs and allocator memory. Input-prefill work still grows with input length.

Full, split and single-token logits agreed within 3e-4 absolute/relative tolerance. The largest observed absolute error over the learned controls was 1.4305115e-05. Reset repeats exactly, independent batch ordering agrees and state offsets equal the processed input length. The backend tests also compare parameter/input/initial-state gradients and interleaved requests.

Every timing repetition is in the [offline report](demo-report.html) and [data](demo-report.json). Token timings use supplied tokens, not autoregressive language generation. Before/after RSS readings are not peak measurements. Timing is a local reference measurement, not a production or GPU throughput claim.

## Incomplete checks

GitHub CI is prepared but has not run; no repository publication has occurred. The complete upstream GPU suites, mixed precision, padding/packed inputs, GVA, a pretrained language model, fresh-machine installation and broader recall tasks were not tested.

The actual report was checked at 1440px and 390px, with all four seed/loop combinations, score-row agreement, keyboard selection, source identity, reduced motion and print visibility. No script errors or network requests were observed. Wide tables scroll within their containers on smaller screens. The README screenshot is the rendered final report.
