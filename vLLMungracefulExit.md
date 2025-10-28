server.py:
```
from vllm import LLM
import time
import signal
import sys
import os  # <-- You were missing this import

# Global flag for the signal handler
running = True

def handle_sigterm(signum, frame):
    """Signal handler to set the running flag to False."""
    global running
    print("Received SIGTERM, shutting down...")
    running = False

def main():
    """Main function to setup and run the server."""
    global running
    
    # Set up the signal handler
    signal.signal(signal.SIGTERM, handle_sigterm)

    # --- All logic now inside main() ---
    
    print("Loading vLLM model...")
    # Simulate loading a vLLM model
    llm = LLM(model="facebook/opt-125m")

    print(f"Model loaded. Waiting... PID: {os.getpid()}")
    try:
        while running:
            # In a real server, this is where you'd process requests
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
    finally:
        print("Cleaning up...")
        # Try deleting LLM instance
        del llm
        print("Exiting now.")
        sys.exit(0)

# This is the critical guard
if __name__ == "__main__":
    main()
```

client.py:
```
import subprocess
import time
import os
import signal

# Start the server
server = subprocess.Popen(["python3", "server.py"])
print(f"Started server with PID {server.pid}")

time.sleep(5)  # Let it load model

print("Terminating parent process abruptly...")
os.kill(server.pid, signal.SIGTERM)  # <- non-graceful shutdown
print("Parent terminated.")
```

Run:
```
> python client.py
Started server with PID 95636
[W1028 13:27:42.066241626 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: AutocastCPU
  previous kernel: registered at /pytorch/aten/src/ATen/autocast_mode.cpp:327
       new kernel: registered at /opt/workspace/ipex-cpu-dev/csrc/cpu/autocast/autocast_mode.cpp:112 (function operator())
INFO 10-28 13:27:43 [__init__.py:224] Automatically detected platform cpu.
Loading vLLM model...
INFO 10-28 13:27:43 [utils.py:239] non-default args: {'disable_log_stats': True, 'model': 'facebook/opt-125m'}
INFO 10-28 13:27:44 [model.py:653] Resolved architecture: OPTForCausalLM
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 10-28 13:27:44 [model.py:1714] Using max model len 2048
WARNING 10-28 13:27:44 [logger.py:75] Environment variable VLLM_CPU_KVCACHE_SPACE (GiB) for CPU backend is not set, using 4 by default.
INFO 10-28 13:27:44 [scheduler.py:225] Chunked prefill is enabled with max_num_batched_tokens=4096.
Terminating parent process abruptly...
Parent terminated.
Received SIGTERM, shutting down...
(rsaini) rsaini@rsaini-thinkpadt14gen3 ~/i/d/gracefulExit (main)> [W1028 13:27:45.684694044 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: AutocastCPU
  previous kernel: registered at /pytorch/aten/src/ATen/autocast_mode.cpp:327
       new kernel: registered at /opt/workspace/ipex-cpu-dev/csrc/cpu/autocast/autocast_mode.cpp:112 (function operator())
INFO 10-28 13:27:46 [__init__.py:224] Automatically detected platform cpu.
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [core.py:727] Waiting for init message from front-end.
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [core.py:94] Initializing a V1 LLM engine (v0.11.0rc2.dev399+ge94cfd51d) with config: model='facebook/opt-125m', speculative_config=None, tokenizer='facebook/opt-125m', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cpu, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=facebook/opt-125m, enable_prefix_caching=True, chunked_prefill_enabled=True, pooler_config=None, compilation_config={'level': 2, 'debug_dump_path': None, 'cache_dir': '', 'backend': 'inductor', 'custom_ops': ['none'], 'splitting_ops': None, 'use_inductor': True, 'compile_sizes': None, 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'dce': True, 'size_asserts': False, 'nan_asserts': False, 'epilogue_fusion': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'use_cudagraph': True, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'full_cuda_graph': False, 'use_inductor_graph_partition': False, 'pass_config': {}, 'max_capture_size': None, 'local_cache_dir': None}
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [importing.py:44] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [importing.py:68] Triton not installed or not compatible; certain GPU-related functions will not be available.
(EngineCore_DP0 pid=95672) WARNING 10-28 13:27:47 [_logger.py:72] Pin memory is not supported on CPU.
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:164] auto thread-binding list (id, physical core): [(1, 0), (3, 1), (5, 2), (7, 3), (8, 4), (9, 5), (10, 6), (11, 7), (12, 8), (13, 9), (14, 10), (15, 11)]
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70] OMP threads binding of Process 95672:
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]       OMP tid: 95672, core 1
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]       OMP tid: 95715, core 3
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]       OMP tid: 95716, core 5
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]       OMP tid: 95717, core 7
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]       OMP tid: 95718, core 8
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]       OMP tid: 95719, core 9
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]       OMP tid: 95720, core 10
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]       OMP tid: 95721, core 11
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]       OMP tid: 95722, core 12
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]       OMP tid: 95723, core 13
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]       OMP tid: 95724, core 14
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]       OMP tid: 95725, core 15
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_worker.py:70]
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [parallel_state.py:1231] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu_model_runner.py:67] Starting to load model facebook/opt-125m...
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [cpu.py:145] Using Torch SDPA backend.
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:47 [weight_utils.py:419] Using model weights format ['*.safetensors', '*.bin', '*.pt']
Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  9.13it/s]
Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  9.12it/s]
(EngineCore_DP0 pid=95672)
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:48 [default_loader.py:314] Loading weights took 0.11 seconds
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:48 [kv_cache_utils.py:1199] GPU KV cache size: 116,480 tokens
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:48 [kv_cache_utils.py:1204] Maximum concurrency for 2,048 tokens per request: 56.88x
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:48 [cpu_model_runner.py:77] Warming up model for the compilation...
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:50 [cpu_model_runner.py:87] Warming up done.
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:50 [core.py:240] init engine (profile, create kv cache, warmup model) took 2.67 seconds
(EngineCore_DP0 pid=95672) WARNING 10-28 13:27:50 [logger.py:75] Environment variable VLLM_CPU_KVCACHE_SPACE (GiB) for CPU backend is not set, using 4 by default.
(EngineCore_DP0 pid=95672) INFO 10-28 13:27:51 [gc_utils.py:40] GC Debug Config. enabled:False,top_objects:-1
INFO 10-28 13:27:51 [llm.py:337] Supported tasks: ['generate']
Model loaded. Waiting... PID: 95636
Cleaning up...
Exiting now.
```
