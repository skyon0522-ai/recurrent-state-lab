"""Bounded learned MQAR evaluation; fixed protocol, independent held-out inputs."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import time

import psutil
import torch
from torch.nn import functional as F

from .zoology import RecallModel, mqar


@dataclass(frozen=True)
class Protocol:
    seeds: tuple = (7, 19)
    loops: tuple = (1, 2)
    epochs: int = 12
    train_examples: int = 1024
    validation_examples: int = 128
    test_examples: int = 256
    train_length: int = 16
    test_lengths: tuple = (16, 32, 64)
    pairs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.003
    threads: int = 2
    repetitions: int = 3

    def validate(self):
        if not self.seeds or len(self.seeds) > 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('Use one or two distinct integer seeds.')
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in self.seeds):
            raise ValueError('Seeds must be non-negative integers.')
        for name, maximum in (('epochs', 32), ('train_examples', 4096),
                              ('validation_examples', 512), ('test_examples', 512)):
            x = getattr(self, name)
            if not isinstance(x, int) or isinstance(x, bool) or not 1 <= x <= maximum:
                raise ValueError(f'{name} must be an integer between1 and{maximum}.')
        fixed = dict(loops=(1, 2), train_length=16, test_lengths=(16, 32, 64),
                     pairs=2, batch_size=32, learning_rate=0.003, threads=2, repetitions=3)
        if any(getattr(self, name) != value for name, value in fixed.items()):
            raise ValueError('Keep the named dimensions, learning rate and timing protocol fixed.')


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def row_hashes(data):
    return [hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
            for x, y in zip(data.inputs, data.labels)]


def datasets(protocol):
    train = mqar(examples=protocol.train_examples, seed=107)
    validation = mqar(examples=protocol.validation_examples, seed=108)
    heldout = {length: mqar(examples=protocol.test_examples, length=length, seed=109 + index)
               for index, length in enumerate(protocol.test_lengths)}
    all_sets = {'train': train, 'validation': validation,
                **{f'test-{length}': data for length, data in heldout.items()}}
    rows = {name: row_hashes(data) for name, data in all_sets.items()}
    names = list(rows)
    for index, name in enumerate(names):
        for other in names[index+1:]:
            if set(rows[name]) & set(rows[other]):
                raise ValueError(f'Exact row overlap between{name} and{other}; stop the experiment.')
    manifest = {name: {'examples': len(data.inputs), 'tokens': data.inputs.shape[1],
                      'query_labels': int((data.labels != -100).sum()),
                      'row_sha256': rows[name]}
                for name, data in all_sets.items()}
    return train, validation, heldout, manifest


def score(model, data, batch_size=32):
    correct = total = 0
    with torch.no_grad():
        for index in range(0, len(data.inputs), batch_size):
            targets = data.labels[index:index+batch_size]
            predictions = model(data.inputs[index:index+batch_size]).argmax(-1)
            mask = targets != -100
            correct += int((predictions[mask] == targets[mask]).sum())
            total += int(mask.sum())
    return dict(correct=correct, queries=total, accuracy=correct / total)


def train_model(protocol, seed, loops, train, validation):
    torch.manual_seed(seed)
    model = RecallModel(loops=loops)
    initial_hash = parameter_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol.learning_rate, weight_decay=0)
    rng = torch.Generator().manual_seed(seed + 1000)
    curve = []
    start = time.perf_counter()
    for epoch in range(protocol.epochs):
        model.train()
        permutation = torch.randperm(len(train.inputs), generator=rng)
        loss_sum = count = 0
        for indices in permutation.split(protocol.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train.inputs[indices])
            loss = F.cross_entropy(logits.reshape(-1, 128), train.labels[indices].reshape(-1))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss; keep the failed run and stop.')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1,
                                                 error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            count += len(indices)
        model.eval()
        curve.append({'epoch': epoch+1, 'loss': loss_sum/count,
                      'last_batch_pre_clip_gradient_norm': float(norm),
                      'validation': score(model, validation)})
    return model, dict(initial_parameter_sha256=initial_hash,
        final_parameter_sha256=parameter_hash(model),
        unique_parameters=sum(p.numel() for p in model.parameters()),
        optimizer_steps=protocol.epochs * ((len(train.inputs)+protocol.batch_size-1)//protocol.batch_size),
        train_seconds=time.perf_counter()-start, curve=curve,
        checkpoint_selection='Fixed final epoch; validation is recorded but never selects an epoch/configuration.')


def controls(model, tokens):
    with torch.no_grad():
        full = model(tokens)
        uninterrupted, state = model.stream(tokens)
        split = tokens.shape[1] // 2
        first, carried = model.stream(tokens[:, :split])
        snapshot = [s.recurrent_state.clone() for s in carried]
        second, end = model.stream(tokens[:, split:], carried)
        chunks = torch.cat([first, second], dim=1)
        token_outputs = []
        step_state = None
        for token in tokens.split(1, dim=1):
            out, step_state = model.stream(token, step_state)
            token_outputs.append(out)
        token_outputs = torch.cat(token_outputs, dim=1)
        for value in (uninterrupted, chunks, token_outputs):
            torch.testing.assert_close(full, value, atol=3e-4, rtol=3e-4)
        for old, saved in zip(carried, snapshot):
            assert torch.equal(old.recurrent_state, saved)
        for a, b, c in zip(state, end, step_state):
            torch.testing.assert_close(a.recurrent_state, b.recurrent_state, atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(a.recurrent_state, c.recurrent_state, atol=3e-5, rtol=3e-5)
        reset, _ = model.stream(tokens)
        assert torch.equal(reset, uninterrupted)
        isolated, _ = model.stream(tokens.flip(0))
        torch.testing.assert_close(isolated.flip(0), uninterrupted, atol=3e-4, rtol=3e-4)
    return dict(full_vs_stream_max_abs=float((full-uninterrupted).abs().max()),
                split_max_abs=float((full-chunks).abs().max()),
                step_max_abs=float((full-token_outputs).abs().max()),
                reset_exact=True, batch_reorder_isolated=True,
                returned_state_bytes=sum(s.nbytes for s in state),
                state_offsets=[s.offset for s in state])


def benchmark(model, tokens, repetitions):
    rows = []
    process = psutil.Process()
    with torch.no_grad():
        model.stream(tokens)
        for repetition in range(repetitions):
            before = process.memory_info().rss
            start = time.perf_counter()
            _, state = model.stream(tokens)
            prefill = time.perf_counter()-start
            start = time.perf_counter()
            for _ in range(16):
                _, state = model.stream(tokens[:, -1:], state)
            one_token = (time.perf_counter()-start)/16
            after = process.memory_info().rss
            rows.append(dict(repetition=repetition+1, prefill_ms=prefill*1000,
                supplied_one_token_ms=one_token*1000, state_bytes=sum(s.nbytes for s in state),
                rss_before_bytes=before, rss_after_bytes=after))
    return rows


def run(output, protocol):
    protocol.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _run(output, protocol)
    except Exception as error:
        write_json(output/'failure.json', {'error_type': type(error).__name__, 'message': str(error),
            'status': 'Incomplete; intermediate results retained in this run-owned directory.'})
        raise


def _run(output, protocol):
    protocol_data = asdict(protocol)
    write_json(output / 'protocol.json', protocol_data)
    package_root = Path(__file__).resolve().parent
    runtime_sources = [{'path': p.relative_to(package_root).as_posix(),
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(package_root.rglob('*.py'))]
    manifests = [{'path': p.relative_to(package_root).as_posix(),
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                  'sources': json.loads(p.read_text(encoding='utf-8'))}
                 for p in sorted((package_root/'_vendor').rglob('manifest.json'))]
    write_json(output/'source-identity.json', {'runtime_files': runtime_sources,
                                              'upstream_manifests': manifests})
    torch.set_num_threads(protocol.threads)
    train, validation, heldout, data_manifest = datasets(protocol)
    write_json(output / 'datasets.json', data_manifest)
    results = []
    for seed in protocol.seeds:
        seed_hash = None
        for loops in protocol.loops:
            model, training = train_model(protocol, seed, loops, train, validation)
            if seed_hash is None:
                seed_hash = training['initial_parameter_sha256']
            assert seed_hash == training['initial_parameter_sha256']
            entry = dict(seed=seed, loops=loops, training=training, evaluations=[])
            for length, data in heldout.items():
                quality = score(model, data)
                labels = data.labels[data.labels != -100]
                entry['evaluations'].append(dict(length=length, quality=quality,
                    constant_value64_baseline=dict(correct=int((labels == 64).sum()),
                        queries=len(labels), accuracy=float((labels == 64).float().mean())),
                    controls=controls(model, data.inputs[:2]),
                    timing=benchmark(model, data.inputs[:1], protocol.repetitions)))
            checkpoint = output / f'model-seed{seed}-loops{loops}.pt'
            torch.save(dict(state_dict=model.state_dict(), loops=loops,
                            protocol=protocol_data), checkpoint)
            entry['checkpoint'] = {'file': checkpoint.name,
                                   'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            write_json(output / f'run-seed{seed}-loops{loops}.json', entry)
            results.append(entry)
    report = dict(schema_version=1, protocol=protocol_data, runs=results,
        protocol_sha256=hashlib.sha256((output/'protocol.json').read_bytes()).hexdigest(),
        data_manifest_sha256=hashlib.sha256((output/'datasets.json').read_bytes()).hexdigest(),
        source_identity_sha256=hashlib.sha256((output/'source-identity.json').read_bytes()).hexdigest(),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
            machine=platform.machine(), cpu=platform.processor(),
            logical_cpus=psutil.cpu_count(), total_memory_bytes=psutil.virtual_memory().total,
            versions={name: importlib.metadata.version(name)
                      for name in ('torch', 'numpy', 'einops', 'psutil')}),
        limits=['CPU float32 reference only; no fused GPU parity or pretrained language-model evaluation.',
                'Synthetic MQAR with128 vocabulary items,2 pairs and zero distractors; not the full Zoology or GDN paper benchmark.',
                'Shared-loop parameters match, but two loops use roughly twice the block work and twice the state; not compute-matched.',
                'Timing is teacher-forced supplied token processing, not generated language tokens; before/after RSS is not a peak.',
                'Fixed recurrent state excludes model weights, sequence activations, training graph, allocator and report data.',
                'These seeded local runs do not establish generalization or statistically reliable superiority.'])
    write_json(output / 'report.json', report)
    return report
