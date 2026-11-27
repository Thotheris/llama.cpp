# Inkling 2 Mini on the Intel Arc Pro B70 with SYCL

This branch adds Inkling 2 to the ggml SYCL and Vulkan backends and tunes the SYCL backend for the Intel Arc Pro B70
(Xe2, 32 GB). It is upstream `074f1f3` plus the commits listed at the end of this page.

Only one setup is validated: Inkling-2-Mini Q4_K_M on one Arc Pro B70. The SYCL changes touch shared code (mul_mat,
flash attention, MoE). Other models and other Intel GPUs may work, but they are untested.

## Run the container image

1. Build the image from the repository root:

	```bash
	podman build -f .devops/intel-b70.Dockerfile --target server -t llama-inkling-b70 .
	```

2. Start the server. Mount a volume at `/root/.cache` so the compiled GPU kernels survive a restart:

	```bash
	podman run --rm --device /dev/dri -p 8080:8080 --shm-size 8g \
		-e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e GGML_SYCL_ENABLE_GRAPH=1 \
		-v llama-sycl-cache:/root/.cache \
		-v /path/to/Inkling-2-Mini-Q4_K_M.gguf:/models/Inkling-2-Mini-Q4_K_M.gguf:ro \
		llama-inkling-b70 -m /models/Inkling-2-Mini-Q4_K_M.gguf --host 0.0.0.0 --port 8080 \
		--ctx-size 131072 --n-gpu-layers 999 --parallel 1 --flash-attn on \
		--batch-size 2048 --ubatch-size 2048 --jinja --reasoning-format deepseek \
		--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 --no-mmproj
	```

The first start compiles the kernels, so the server takes about 60 s to report healthy. With the cache volume, later
starts take about 7 s.

`intel-b70.Dockerfile` is `intel.Dockerfile` with an older toolchain. It pins oneAPI 2026.0 and oneDNN 3.11. oneDNN
3.13 and 3.14 (oneAPI 2026.1) spend about 60 ms the first time each matmul shape runs. In a new container, that made
prefill 7 to 8% slower on every new prompt length.

## Build on the host

The host build needs oneAPI 2026.0, Level Zero, compute-runtime 26.40 and IGC 2.42:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
	-DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON \
	-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl -j
```

Run `build-sycl/bin/llama-server` with the same environment and arguments as the container.

## Environment

- Set `GGML_SYCL_ENABLE_GRAPH=1`. It records the decode step as a SYCL graph. Without it, each kernel launch costs 3
  to 4 µs on the host.
- Do not set `SYCL_CACHE_PERSISTENT=1`. With oneAPI 2026.0 and compute-runtime 26.40, it crashed or hung the server
  and left GPU page faults. The default cache in `/root/.cache` works.
- Keep the default context checkpoints. You do not need `--ctx-checkpoints 0`.

## Debug switches

These environment variables turn kernel paths off, for bisecting a wrong result. The defaults are the measured
configuration.

| Variable | Default | Effect |
|---|---|---|
| `GGML_SYCL_FA_XE` | `1` | `0` turns off the Xe2 flash-attention prefill kernel. |
| `GGML_SYCL_FA_XE_THR` | `0` | The lazy softmax rescale threshold. `0` rescales exactly. At `5.5`, the model missed a needle in a 110K prompt. |
| `GGML_SYCL_DENSE_DNN` | `1` | `0` runs the dense projections without oneDNN. |
| `GGML_SYCL_DENSE_DNN_MIN_COLS` | `48` | The smallest batch that uses oneDNN for the dense projections. |
| `GGML_SYCL_XMX_MAX_COLS` | `80` | The largest batch that uses the XMX GEMM on reordered weights. `0` turns it off. |
| `LLAMA_INKLING2_MIX_NORM_BATCH` | `1` | `0` uses unfused block-mix norms for batches of more than 8 tokens. |
| `LLAMA_SERVER_SAMPLE_THREADS` | `8` | The threads that sample the server's slots in parallel. |
| `LLAMA_SPEC_MAX_SLOTS` | `1` | The server stops drafting while more slots than this generate. `0` means no limit. |
| `LLAMA_SERVER_STEP_LOG` | unset | Any value prints one line per server step, with the ubatch sizes. |

## Measured performance

All numbers are for Inkling-2-Mini Q4_K_M on one Arc Pro B70. The last column names the commit that was measured.
`8a796214e` has the same code as the branch head. Only documentation changed after it.

| Metric | Value | How | Commit |
|---|---|---|---|
| Prefill, `llama-bench -p 2048 -ub 2048` | 6,387 tok/s | median of 6, between two control runs | `8a796214e` |
| Decode, `llama-bench -n 128` | 98.2 tok/s | median of 6, between two control runs | `8a796214e` |
| Prefill, `llama-bench -p 512` | 4,710 tok/s | median of 6 | `abdbb4b0b` |
| Chat prefill, 2K and 8K prompts | 5,300 and 5,860 tok/s | `llama-server`, `-ub 2048` | `abdbb4b0b` |
| Chat decode, empty and 8K context | 94 and 84 tok/s | `llama-server` | `abdbb4b0b` |
| Device memory at `--ctx-size 131072` | 24,143 MiB | the model 16,412, KV and state 4,579, compute 3,151 | `abdbb4b0b` |
| KL divergence against the Vulkan backend's logits | 0.0018 to 0.0028 | `llama-perplexity --kl-divergence`, ubatch 4, 12, 32, 128, 512 and 2048 | `8a796214e` |

The last fix only changes batches of 9 to 47 tokens whose input is a strided view, so the `abdbb4b0b` rows still
apply.

These accuracy checks passed on `8a796214e`, at the arguments above:

- A tool-call prompt returned a correct `get_weather` call 10 times out of 10.
- The model found a needle in a 91,890-token chat prompt at temperature 0.6 and at temperature 0.

## Known limits

- `ggml_sycl_op_mul_mat` had a bug that upstream also has. When src1 was not contiguous, the op copied it and then
  quantized it as `block_q8_1`. The kernels for reordered weights read a per-row layout instead, with all of a row's
  int8 values first and its scales after them. Inkling 2's chat prompts hit this through a 12-token ubatch, which
  wrote NaN into the KV cache. Commit `8a796214e` fixes the copy path. Other code paths also quantize into the
  reordered layout straight from src1: the MoE multi-column path and the fused GLU XMX path. They have not been
  checked.
- Only the SYCL path was measured on the final commits. The Vulkan kernels were last checked before the fix, and the
  fix does not touch them.

## What the branch changes

| Commit | What it adds |
|---|---|
| `e794a27cd` ggml: fused ops for the Inkling 2 graph, with CPU kernels | The fused ops the graph uses (REL_BIAS, BLOCK_MIX and BLOCK_MIX_NORM, GATE_MIX and GATE_MIX_MV, SSM_CONV_HIST, L2_CLIP, the SiTU GLU), with reference CPU kernels |
| `7803b4dd1` vulkan: kernels for the Inkling 2 fused ops and Intel Xe2 paths | Vulkan kernels for the same ops |
| `bd49c7639` sycl: Inkling 2 fused ops and the Xe2 MoE, dense and attention paths | SYCL kernels for the ops, a MoE schedule built on the device, ESIMD grouped GEMMs, dense projections through oneDNN on f16 weight copies, flash attention that computes the relative-position bias, decode attention, and graph recording |
| `3c0e6ca60` llama: graph pool, relative-position attention input, stream runs | Graph reuse across ubatch shapes, and the relative-position input of the attention |
| `6b0bf1233` model: Inkling 2 graph on the fused ops | The model graph in `src/models/inkling2.cpp` |
| `acbcf5e6a` dflash: Inkling 2 DSpark draft in the mtp.safetensors layout | The DSpark draft model in a second tensor layout |
| `a3972d6b0` speculative: Inkling 2 DSpark verify inside one cell block, adaptive pausing | A verify whose rows match one-token decode, and draft pausing when acceptance is low |
| `c0fb3d779` server: parallel slot sampling, draft slot limit, step log | `LLAMA_SERVER_SAMPLE_THREADS`, `LLAMA_SPEC_MAX_SLOTS` and `LLAMA_SERVER_STEP_LOG` |
| `abdbb4b0b` devops: Intel Arc Pro B70 image on the validated oneAPI 2026.0 toolchain | `.devops/intel-b70.Dockerfile` |
| `8a796214e` sycl: quantize a non-contiguous src1 in the layout the mul_mat op reads | The fix described in "Known limits" |
