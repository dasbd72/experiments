"""An experiment of using multiple CUDA streams in PyTorch."""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import profiler
from torch.nn import functional as F


class Experiment:
    """An experiment with multiple CUDA streams."""

    def __init__(self, thread_pool=False, pin_memory=False):
        self.thread_pool = thread_pool
        self.pin_memory = pin_memory

        self.compute_size = 8000
        self.cpy_size = 10000

        self.com_stream = torch.cuda.Stream()
        self.com_buf_a = torch.randn(
            (self.compute_size, self.compute_size), device="cuda"
        )
        self.com_buf_b = torch.randn(
            (self.compute_size, self.compute_size), device="cuda"
        )
        self.com_buf_c = torch.zeros(
            (self.compute_size, self.compute_size), device="cuda"
        )

        self.dtoh_stream = torch.cuda.Stream()
        self.dtoh_gpu_buf = torch.randn(
            (self.cpy_size, self.cpy_size), device="cuda"
        )
        self.dtoh_cpu_buf = torch.empty(
            (self.cpy_size, self.cpy_size),
            device="cpu",
            pin_memory=self.pin_memory,
        )

        self.htod_stream = torch.cuda.Stream()
        self.htod_cpu_buf = torch.randn(
            (self.cpy_size, self.cpy_size),
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.htod_gpu_buf = torch.empty(
            (self.cpy_size, self.cpy_size), device="cuda"
        )

    def _compute(self):
        # pylint: disable=not-callable
        with torch.cuda.stream(self.com_stream):
            self.com_buf_c = F.linear(self.com_buf_a, self.com_buf_b)

    def _dtoh(self):
        with torch.cuda.stream(self.dtoh_stream):
            self.dtoh_cpu_buf.copy_(self.dtoh_gpu_buf, non_blocking=True)

    def _htod(self):
        with torch.cuda.stream(self.htod_stream):
            self.htod_gpu_buf.copy_(self.htod_cpu_buf, non_blocking=True)

    def _launch(self):
        if self.thread_pool:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(self._compute),
                    executor.submit(self._dtoh),
                    executor.submit(self._htod),
                ]
                for future in futures:
                    future.result()
        else:
            self._compute()
            self._dtoh()
            self._htod()
        torch.cuda.synchronize()

    def run(self):
        """Run the experiment and profile it."""
        self._launch()

        start = time.perf_counter()
        self._compute()
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"Computation time: {end - start:.6f} seconds.")

        start = time.perf_counter()
        self._dtoh()
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"DtoH time: {end - start:.6f} seconds.")

        start = time.perf_counter()
        self._htod()
        torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"HtoD time: {end - start:.6f} seconds.")

        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            self._launch()
        file_name = "trace"
        if self.thread_pool:
            file_name += "_threadpool"
        if self.pin_memory:
            file_name += "_pinmemory"
        file_name += ".json"
        file_path = os.path.join(
            os.path.dirname(__file__), "traces", file_name
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        prof.export_chrome_trace(file_path)
        print(f"Trace exported to {file_path}.")


def main():
    "Main function to run the experiment."
    exp = Experiment(thread_pool=False, pin_memory=False)
    exp.run()
    exp = Experiment(thread_pool=True, pin_memory=False)
    exp.run()
    exp = Experiment(thread_pool=False, pin_memory=True)
    exp.run()
    exp = Experiment(thread_pool=True, pin_memory=True)
    exp.run()


if __name__ == "__main__":
    main()
