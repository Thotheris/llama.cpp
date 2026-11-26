# Inkling 2 Mini on the Intel Arc Pro B70 (SYCL)

This branch adds Inkling 2 support to the ggml SYCL and Vulkan backends and tunes the SYCL backend for the Intel Arc
Pro B70 (Xe2, 32 GB). It is a fork branch, not an upstream proposal. Everything below was measured on one machine
with one model; other models and other Intel GPUs have not been validated on these code paths.

## What is in the branch

| Commit (subject) | What it does |
|---|---|
| ggml: fused ops for the Inkling 2 graph, with CPU kernels | REL_BIAS, BLOCK_MIX(_NORM), GATE_MIX(_MV), SSM_CONV_HIST, L2_CLIP, the SiTU GLU and the other ops the graph uses, with reference CPU kernels |
| vulkan: kernels for the Inkling 2 fused ops and Intel Xe2 paths | Vulkan kernels for the same ops |
| sycl: Inkling 2 fused ops and the Xe2 MoE, dense and attention paths | SYCL kernels for the ops, the device-side MoE schedule and ESIMD grouped GEMM, dense projections through oneDNN on persistent f16 weight copies, flash attention with the relative-position bias computed in the kernel, decode attention, graph recording |
| llama: graph pool, relative-position attention input, stream runs | graph reuse across ubatch shapes, the relative-position input of the attention |
| model: Inkling 2 graph on the fused ops | the model graph (src/models/inkling2.cpp) |
| dflash / speculative | the DSpark draft in the mtp.safetensors layout and its verify |
| server: parallel slot sampling, draft slot limit, step log | server options (`LLAMA_SERVER_SAMPLE_THREADS`, `LLAMA_SPEC_MAX_SLOTS`, `LLAMA_SERVER_STEP_LOG`) |
| devops: Intel Arc Pro B70 image | `.devops/intel-b70.Dockerfile` |
| sycl: quantize a non-contiguous src1 in the layout the mul_mat op reads | a fix for a bug that is also in upstream: see "Known issues" |

## Build

Host build (oneAPI 2026.0, Level Zero, compute-runtime 26.40, IGC 2.42):

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
      -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON \
      -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl -j
```

Container image (the build the measurements below were checked against):

```bash
podman build -f .devops/intel-b70.Dockerfile --target server -t llama-inkling-b70 .
```

`intel-b70.Dockerfile` differs from `intel.Dockerfile` only in the toolchain: it pins the oneAPI 2026.0 base image and
the oneDNN 2026.0 package (oneDNN 3.11). With oneDNN 3.13 (oneAPI 2026.1) every new matmul shape costs about 60 ms on
first use, which made prefill 7-8% slower in a fresh container.

## Run

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 \
llama-server -m Inkling-2-Mini-Q4_K_M.gguf --ctx-size 131072 --n-gpu-layers 999 --parallel 1 \
  --flash-attn on --batch-size 2048 --ubatch-size 2048 --jinja --reasoning-format deepseek \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0
```

- `GGML_SYCL_ENABLE_GRAPH=1` records the decode graph; without it every kernel launch costs 3-4 us on the host.
- In a container, mount a volume at `/root/.cache`. The first start JIT-compiles the kernels (about 50 s to load and
  a slow first request per new shape); with the cache kept the load takes about 7 s.
- Do not set `SYCL_CACHE_PERSISTENT=1`: on this stack it crashed or hung the server and left GPU page faults.

Switches that change kernels, for debugging (defaults are the measured configuration):

| Variable | Default | Effect |
|---|---|---|
| `GGML_SYCL_FA_XE=0` | 1 | turn off the Xe2 flash-attention prefill kernel |
| `GGML_SYCL_FA_XE_THR=t` | 0 | lazy softmax rescale threshold; 0 rescales exactly. With 5.5 a needle in a 110K prompt was lost |
| `GGML_SYCL_DENSE_DNN=0` | 1 | dense projections without oneDNN |
| `GGML_SYCL_DENSE_DNN_MIN_COLS=n` | 48 | smallest batch that takes oneDNN |
| `GGML_SYCL_XMX_MAX_COLS=n` | | largest batch that takes the XMX GEMM on reordered weights |
| `LLAMA_INKLING2_MIX_NORM_BATCH=0` | 1 | unfused block-mix norms for batches over 8 tokens |

## Measured (Inkling-2-Mini Q4_K_M, Arc Pro B70)

| Metric | Value | How |
|---|---|---|
| prefill, llama-bench pp2048 (-ub 2048) | 6,387 tok/s | median of 6, bracketed by controls |
| prefill, llama-bench pp512 | about 4,700 tok/s | median of 6 |
| decode, llama-bench tg128 | 98 tok/s | median of 6 |
| chat prefill, 2k / 8k prompt | about 5,300 / 5,860 tok/s | llama-server, -ub 2048 |
| chat decode, empty / 8k context | 94 / 84 tok/s | llama-server |
| VRAM at --ctx-size 131072 | about 24 GiB | driver residency |
| KL divergence vs Vulkan logits | 0.0018-0.0028 | llama-perplexity --kl-divergence, ub 4 to 2048 |

The chat and VRAM rows were measured on the commit before the last fix, which only changes batches of 9 to 47
tokens whose input is a strided view; the llama-bench and KLD rows were measured on the final commit.

Accuracy checks on the final commit: KLD against the Vulkan backend at ubatch sizes 4, 12, 32, 128, 512 and 2048; a
chat needle at 92K tokens (91,890 prompt tokens) found at temperature 0.6 and 0; tool calls 10/10 on a fixed prompt.

## Known issues and limits

- Only Inkling-2-Mini (Q4_K_M) on the Arc Pro B70 was validated. The SYCL changes touch shared paths (mul_mat,
  flash attention, MoE); other models may work but are untested.
- The non-contiguous src1 bug fixed in the last commit is in upstream too (ggml_sycl_op_mul_mat quantized a copied
  src1 as block_q8_1 even when the kernel reads the reordered SoA layout). Several other call sites that quantize
  into the reorder layout straight from src1 have not been audited.
- `--ctx-checkpoints 0` is not needed. The checkpoint split of chat prompts was what exposed the bug above.
