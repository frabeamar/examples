import os
from typing import Dict, Optional

import numpy as np
import torch
import torch_tensorrt as torch_trt
import torchvision.models as models
from torch_tensorrt.dynamo._defaults import TIMING_CACHE_PATH
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache

np.random.seed(0)
torch.manual_seed(0)

model = models.resnet18(pretrained=True).to("cuda").eval()
enabled_precisions = {torch.float}
min_block_size = 1
use_python_runtime = False


def remove_timing_cache(path=TIMING_CACHE_PATH):
    if os.path.exists(path):
        os.remove(path)


def torch_compile(iterations=3):
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # The 1st iteration is to measure the compilation time without engine caching
    # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
    # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
    # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
    for i in range(iterations):
        inputs = [torch.rand((100, 3, 224, 224)).to("cuda")]
        # remove timing cache and reset dynamo just for engine caching messurement
        remove_timing_cache()
        torch._dynamo.reset()

        if i == 0:
            cache_built_engines = False
            reuse_cached_engines = False
        else:
            cache_built_engines = True
            reuse_cached_engines = True

        start.record()
        compiled_model = torch.compile(
            model,
            backend="tensorrt",
            options={
                "use_python_runtime": True,
                "enabled_precisions": enabled_precisions,
                "min_block_size": min_block_size,
                "immutable_weights": False,
                "cache_built_engines": cache_built_engines,
                "reuse_cached_engines": reuse_cached_engines,
            },
        )
        with torch.no_grad():
            compiled_model(*inputs)  # trigger the compilation
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    print("----------------torch_compile----------------")
    print("disable engine caching, used:", times[0], "ms")
    print("enable engine caching to cache engines, used:", times[1], "ms")
    print("enable engine caching to reuse engines, used:", times[2], "ms")


def dynamo_compile(iterations=3):
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
    # Mark the dim0 of inputs as dynamic
    batch = torch.export.Dim("batch", min=1, max=200)
    exp_program = torch.export.export(
        model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
    )

    # The 1st iteration is to measure the compilation time without engine caching
    # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
    # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
    # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
    for i in range(iterations):
        inputs = [torch.rand((100 + i, 3, 224, 224)).to("cuda")]
        remove_timing_cache()  # remove timing cache just for engine caching messurement
        if i == 0:
            cache_built_engines = False
            reuse_cached_engines = False
        else:
            cache_built_engines = True
            reuse_cached_engines = True

        start.record()
        trt_gm = torch_trt.dynamo.compile(
            exp_program,
            tuple(inputs),
            use_python_runtime=use_python_runtime,
            enabled_precisions=enabled_precisions,
            min_block_size=min_block_size,
            immutable_weights=False,
            cache_built_engines=cache_built_engines,
            reuse_cached_engines=reuse_cached_engines,
            engine_cache_size=1 << 30,  # 1GB
        )
        # output = trt_gm(*inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    print("----------------dynamo_compile----------------")
    print("disable engine caching, used:", times[0], "ms")
    print("enable engine caching to cache engines, used:", times[1], "ms")
    print("enable engine caching to reuse engines, used:", times[2], "ms")


class RAMEngineCache(BaseEngineCache):
    def __init__(
        self,
    ) -> None:
        """
        Constructs a user held engine cache in memory.
        """
        self.engine_cache: Dict[str, bytes] = {}

    def save(
        self,
        hash: str,
        blob: bytes,
    ):
        """
        Insert the engine blob to the cache.

        Args:
            hash (str): The hash key to associate with the engine blob.
            blob (bytes): The engine blob to be saved.

        Returns:
            None
        """
        self.engine_cache[hash] = blob

    def load(self, hash: str) -> Optional[bytes]:
        """
        Load the engine blob from the cache.

        Args:
            hash (str): The hash key of the engine to load.

        Returns:
            Optional[bytes]: The engine blob if found, None otherwise.
        """
        if hash in self.engine_cache:
            return self.engine_cache[hash]
        else:
            return None


def torch_compile_my_cache(iterations=3):
    times = []
    engine_cache = RAMEngineCache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # The 1st iteration is to measure the compilation time without engine caching
    # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
    # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
    # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
    for i in range(iterations):
        inputs = [torch.rand((100, 3, 224, 224)).to("cuda")]
        # remove timing cache and reset dynamo just for engine caching messurement
        remove_timing_cache()
        torch._dynamo.reset()

        if i == 0:
            cache_built_engines = False
            reuse_cached_engines = False
        else:
            cache_built_engines = True
            reuse_cached_engines = True

        start.record()
        compiled_model = torch.compile(
            model,
            backend="tensorrt",
            options={
                "use_python_runtime": True,
                "enabled_precisions": enabled_precisions,
                "min_block_size": min_block_size,
                "immutable_weights": False,
                "cache_built_engines": cache_built_engines,
                "reuse_cached_engines": reuse_cached_engines,
                "custom_engine_cache": engine_cache,
            },
        )
        with torch.no_grad():
            compiled_model(*inputs)  # trigger the compilation
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    print("----------------torch_compile----------------")
    print("disable engine caching, used:", times[0], "ms")
    print("enable engine caching to cache engines, used:", times[1], "ms")
    print("enable engine caching to reuse engines, used:", times[2], "ms")


torch_compile_my_cache()
dynamo_compile()
torch_compile()
