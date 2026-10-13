# CPU reference and state contract

The original GatedDeltaNet class body is retained. Its projections, gates, convolution guard, output normalization and projection are executed with CPU bindings. The change is the usable reference execution path and explicit returned state; variable-length input and inference are existing FLA capabilities.

`CPUReferenceLayer` supports dense CPU float32 tensors shaped `[batch, tokens, hidden]`. Both reference modes use the same parameters. It returns `(output, state)` and keeps no request state on the module. Pass `state=None` to reset. Preserve batch order between calls; a state belongs to its layer. Padding, packed sequences, GVA, autocast and GPU execution are refused.

The state contains the value-first recurrent tensor `[B,H,V,K]`, three convolution history tensors `[B,D,W]`, total token offset and layer identity. Incoming history is copied before the upstream update reference can mutate it. Returned tensor storage is compact; `state.nbytes` counts tensor storage, not a retained autograd graph or the process's full memory. `state.detach()` separates inference state from a training graph.

Two immutable class bindings select the actual chunk or recurrent Torch reference. This deliberately overrides FLA's original automatic operation selection, which otherwise depends on autograd, training and input length. Neither dispatch nor request state uses a mutable module-global selector.

The adapter reuses the original gate and normalization functions, preserves the q/k normalization epsilon inside the square root, and translates convolution and recurrence cache layouts. `ShortConvolution`, `RMSNorm` and `FusedRMSNormGated` class bodies are also retained. The source loader selects unchanged AST definitions to avoid importing Triton/CUDA or a remote trainer. It accepts only package-owned files; it is not an interface for arbitrary user code.

Zoology supplies the unchanged MQAR generator, token embedding, MLP, TransformerBlock, LMBackbone and LanguageModel bodies. CPU embeddings and zero drop-path bindings replace unnecessary GPU/torchvision imports. Shared loops reuse the same block object. A streaming extension preserves the original block operation order while returning one independent state per loop.

This is a local reference composition. It is not an upstream CPU backend and has not been compared with the fused GPU layer. Original files, version identities and licenses are retained under [`_vendor`](../src/recurrent_state_lab/_vendor/).
