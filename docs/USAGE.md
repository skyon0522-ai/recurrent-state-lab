# Run and inspect an experiment

Use Python 3.12 and a CPU build of PyTorch. The first run does not download any pretrained model or send data to a service. It trains four small models locally and retains their weights and reports.

```sh
python -m recurrent_state_lab.cli --output runs/demo
```

The directory must be new. A complete run contains:

| File | Purpose |
|---|---|
| `protocol.json` | Fixed recipe written before training |
| `source-identity.json` | Runtime source hashes and copied version manifests |
| `datasets.json` | Input/label row identities, counts and split lengths |
| `run-seed*-loops*.json` | Each model's training history, held-out scores, controls and all timing repetitions |
| `model-seed*-loops*.pt` | Locally learned tensor checkpoint |
| `report.json`, `report.html` | Combined data and offline interactive report |
| `failure.json` | Incomplete run's error; earlier outputs are retained |

For a short installation check:

```sh
python -m recurrent_state_lab.cli --output runs/smoke --epochs 1 --train-examples 32 --seeds 7
```

Smoke accuracy is not model-quality evidence. The fixed recipe supports one/two seeds, 1–32 epochs and up to 4096 training examples. Dimensions and timing repetitions remain fixed. The full default run trains at length 16 and evaluates separate rows at lengths 16, 32, 64.

Open `report.html` in a browser. Choose an initialization seed and one/two shared loops. Compare query accuracy, prefill time, supplied-token time, exact state bytes and state-carry controls. All measurements are retained. No external asset or API is required by the report.

## Resume inference from a tensor checkpoint

```python
import torch
from recurrent_state_lab.zoology import RecallModel, mqar

checkpoint = torch.load("runs/demo/model-seed7-loops1.pt",
                        map_location="cpu", weights_only=True)
model = RecallModel(loops=checkpoint["loops"]).eval()
model.load_state_dict(checkpoint["state_dict"], strict=True)
tokens = mqar(examples=1, length=32, seed=43).inputs
with torch.no_grad():
    first_logits, state = model.stream(tokens[:, :12])
    next_logits, state = model.stream(tokens[:, 12:], state)
    reset_logits, _ = model.stream(tokens, state=None)
```

Use checkpoints you trust. Keep state with the same model and batch order; reuse with another layer is rejected. A reset is explicit. Retaining gradient-enabled state also retains a computation graph; tensor payload alone is not a training-memory bound.

## Supported scope

CPU float32; equal-length dense sequences; a small fixed MQAR recipe. No padded/packed inputs, GVA, mixed precision or GPU backend. The experiment is a reference and debugging tool, not a pretrained chatbot or a production serving engine. See [implementation](IMPLEMENTATION.md), [research](RESEARCH.md) and [validation](VALIDATION.md) for the exact scope.
