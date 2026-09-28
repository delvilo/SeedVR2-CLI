"""Append-only phase measurements, with bounded host-side bookkeeping."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time


def process_rss_bytes():
    """Linux self RSS, including containers whose visible PID namespace differs."""
    with open('/proc/self/statm') as status:
        return int(status.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')


class MemoryRecorder:
    def __init__(self, path=None, device=None):
        self.path = Path(path) if path else None
        self.device = device

    def record(self, event, **values):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as output:
            output.write(json.dumps(dict(event=event, pid=os.getpid(), **values)) + '\n')

    @contextmanager
    def phase(self, name):
        if self.path is None:
            yield
            return
        import torch
        start = time.monotonic()
        peak = [process_rss_bytes()]
        before = peak[0]
        stop = threading.Event()

        def sample():
            while not stop.wait(0.05):
                peak[0] = max(peak[0], process_rss_bytes())

        cuda = torch.cuda.is_available() and (self.device is None or torch.device(self.device).type == 'cuda')
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        ok = False
        try:
            yield
            ok = True
        finally:
            stop.set()
            monitor.join()
            after = process_rss_bytes()
            values = dict(phase=name, ok=ok, seconds=time.monotonic() - start,
                          rss_before=before, rss_after=after, rss_peak=max(peak[0], after))
            if cuda:
                values.update(cuda_allocated=torch.cuda.memory_allocated(self.device),
                              cuda_reserved=torch.cuda.memory_reserved(self.device),
                              cuda_peak_allocated=torch.cuda.max_memory_allocated(self.device),
                              cuda_peak_reserved=torch.cuda.max_memory_reserved(self.device))
            self.record('phase', **values)
