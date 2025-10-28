server.py:
```
from vllm import LLM
import time
import signal
import sys

running = True

def handle_sigterm(signum, frame):
    global running
    print("Received SIGTERM, shutting down...")
    running = False

signal.signal(signal.SIGTERM, handle_sigterm)

# Simulate loading a vLLM model
llm = LLM(model="facebook/opt-125m")

print("Model loaded. Waiting... PID:", os.getpid())
try:
    while running:
        time.sleep(1)
except KeyboardInterrupt:
    print("Keyboard interrupt received.")
finally:
    print("Cleaning up...")
    # Try deleting LLM instance
    del llm
    print("Exiting now.")
    sys.exit(0)

if __name__ == "__main__":   # <- **critical** for multiprocessing
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
Started server with PID 93617
[W1028 13:03:01.717188052 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: AutocastCPU
  previous kernel: registered at /pytorch/aten/src/ATen/autocast_mode.cpp:327
       new kernel: registered at /opt/workspace/ipex-cpu-dev/csrc/cpu/autocast/autocast_mode.cpp:112 (function operator())
INFO 10-28 13:03:02 [__init__.py:224] Automatically detected platform cpu.
INFO 10-28 13:03:03 [utils.py:239] non-default args: {'disable_log_stats': True, 'model': 'facebook/opt-125m'}
INFO 10-28 13:03:03 [model.py:653] Resolved architecture: OPTForCausalLM
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 10-28 13:03:03 [model.py:1714] Using max model len 2048
WARNING 10-28 13:03:03 [logger.py:75] Environment variable VLLM_CPU_KVCACHE_SPACE (GiB) for CPU backend is not set, using 4 by default.
INFO 10-28 13:03:03 [scheduler.py:225] Chunked prefill is enabled with max_num_batched_tokens=4096.
Terminating parent process abruptly...
Parent terminated.
Received SIGTERM, shutting down...
(rsaini) rsaini@rsaini-thinkpadt14gen3 ~/i/debug> [W1028 13:03:05.288541981 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
    registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: AutocastCPU
  previous kernel: registered at /pytorch/aten/src/ATen/autocast_mode.cpp:327
       new kernel: registered at /opt/workspace/ipex-cpu-dev/csrc/cpu/autocast/autocast_mode.cpp:112 (function operator())
INFO 10-28 13:03:06 [__init__.py:224] Automatically detected platform cpu.
INFO 10-28 13:03:07 [utils.py:239] non-default args: {'disable_log_stats': True, 'model': 'facebook/opt-125m'}
INFO 10-28 13:03:07 [model.py:653] Resolved architecture: OPTForCausalLM
`torch_dtype` is deprecated! Use `dtype` instead!
INFO 10-28 13:03:07 [model.py:1714] Using max model len 2048
WARNING 10-28 13:03:07 [logger.py:75] Environment variable VLLM_CPU_KVCACHE_SPACE (GiB) for CPU backend is not set, using 4 by default.
INFO 10-28 13:03:07 [scheduler.py:225] Chunked prefill is enabled with max_num_batched_tokens=4096.
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/spawn.py", line 131, in _main
    prepare(preparation_data)
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/spawn.py", line 246, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen runpy>", line 287, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "/home/rsaini/inference/debug/server.py", line 16, in <module>
    llm = LLM(model="facebook/opt-125m")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/inference/vllm_source/vllm/entrypoints/llm.py", line 328, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/llm_engine.py", line 186, in from_engine_args
    return cls(
           ^^^^
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/llm_engine.py", line 121, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/core_client.py", line 93, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/core_client.py", line 641, in __init__
    super().__init__(
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/core_client.py", line 470, in __init__
    with launch_core_engines(vllm_config, executor_class, log_stats) as (
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 137, in __enter__
    return next(self.gen)
           ^^^^^^^^^^^^^^
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/utils.py", line 797, in launch_core_engines
    local_engine_manager = CoreEngineProcManager(
                           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/utils.py", line 141, in __init__
    proc.start()
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/context.py", line 289, in _Popen
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/popen_spawn_posix.py", line 42, in _launch
    prep_data = spawn.get_preparation_data(process_obj._name)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/spawn.py", line 164, in get_preparation_data
    _check_not_importing_main()
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/multiprocessing/spawn.py", line 140, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html

Traceback (most recent call last):
  File "/home/rsaini/inference/debug/server.py", line 16, in <module>
    llm = LLM(model="facebook/opt-125m")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/inference/vllm_source/vllm/entrypoints/llm.py", line 328, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/llm_engine.py", line 186, in from_engine_args
    return cls(
           ^^^^
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/llm_engine.py", line 121, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/core_client.py", line 93, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/core_client.py", line 641, in __init__
    super().__init__(
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/core_client.py", line 470, in __init__
    with launch_core_engines(vllm_config, executor_class, log_stats) as (
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/rsaini/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/utils.py", line 815, in launch_core_engines
    wait_for_engine_startup(
  File "/home/rsaini/inference/vllm_source/vllm/v1/engine/utils.py", line 872, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore_DP0': 1}
```
