from time import perf_counter


class Profiling(object):
    def __init__(self, phase):
        self.phase = phase

    def __enter__(self):
        self.start = perf_counter()
        print(f"- Phase {self.phase}", flush=True)
        return self

    def __exit__(self, *args):
        t = perf_counter() - self.start
        print(f"  {t:.3f} seconds", flush=True)