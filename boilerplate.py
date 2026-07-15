"""
SimPy simulation + real-time Tkinter GUI + live PC hardware stats.

ARCHITECTURE
------------
Thread 1 (main/GUI thread): runs Tkinter's mainloop(). Uses self.after() to
    poll a queue.Queue for simulation updates, and to sample hardware stats
    directly (psutil calls are fast/non-blocking once "warmed up").
Thread 2 (worker thread): runs simpy.Environment().run() step by step. The
    simulation pushes state snapshots onto a thread-safe queue.Queue instead
    of touching Tkinter widgets directly -- Tkinter is NOT thread-safe, so
    the worker thread must never call any tk widget method.

Install:
    pip install simpy psutil
    pip install pynvml   # optional, only needed for NVIDIA GPU stats
"""

import queue
import random
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk

import psutil
import simpy

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Simulation -- runs entirely on the worker thread
# ---------------------------------------------------------------------------

@dataclass
class SimEvent:
    sim_time: float
    kind: str  # "job_start" | "job_end"
    payload: dict = field(default_factory=dict)


class JobShopSimulation:
    """Toy example: jobs arrive and compete for N GPU 'slots'.
    Swap this class out for your actual GPU scheduling model -- the
    threading/queue wiring around it stays the same."""

    def __init__(self, event_q: "queue.Queue[SimEvent]", stop_flag: threading.Event,
                 num_gpus: int = 4, real_seconds_per_sim_unit: float = 0.4):
        self.event_q = event_q
        self.stop_flag = stop_flag
        self.pace = real_seconds_per_sim_unit
        self.env = simpy.Environment()
        self.gpus = simpy.Resource(self.env, capacity=num_gpus)
        self.job_id = 0

    def job(self, duration):
        jid = self.job_id
        self.job_id += 1
        with self.gpus.request() as req:
            yield req
            self.event_q.put(SimEvent(self.env.now, "job_start",
                                       {"id": jid, "queue_len": len(self.gpus.queue)}))
            yield self.env.timeout(duration)
            self.event_q.put(SimEvent(self.env.now, "job_end", {"id": jid}))

    def arrivals(self):
        while True:
            yield self.env.timeout(random.expovariate(1 / 2.0))
            self.env.process(self.job(random.uniform(1, 5)))

    def run(self):
        self.env.process(self.arrivals())
        # Step the clock manually (instead of env.run()) so we can pace it
        # to wall-clock time and check the stop flag between events.
        last_time = 0.0
        while not self.stop_flag.is_set():
            self.env.step()
            dt = self.env.now - last_time
            last_time = self.env.now
            time.sleep(min(dt, 1.0) * self.pace)


def simulation_thread_main(event_q, stop_flag):
    sim = JobShopSimulation(event_q, stop_flag)
    try:
        sim.run()
    except simpy.core.EmptySchedule:
        pass


# ---------------------------------------------------------------------------
# 2. Hardware sampling -- cheap enough to call directly on the GUI thread
# ---------------------------------------------------------------------------

def sample_hardware() -> dict:
    stats = {
        "cpu_total": psutil.cpu_percent(interval=None),
        "mem": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
    }
    if GPU_AVAILABLE:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            stats["gpu_util"] = util.gpu
            stats["gpu_mem"] = 100 * mem.used / mem.total
        except Exception:
            pass
    return stats


# ---------------------------------------------------------------------------
# 3. GUI -- only this thread ever touches Tkinter widgets
# ---------------------------------------------------------------------------

class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SimPy Simulation + Hardware Monitor")
        self.geometry("640x520")

        self.event_q: "queue.Queue[SimEvent]" = queue.Queue()
        self.stop_flag = threading.Event()
        self.active_jobs = {}

        self._build_sim_panel()
        self._build_hw_panel()

        self.worker = threading.Thread(
            target=simulation_thread_main, args=(self.event_q, self.stop_flag), daemon=True
        )
        self.worker.start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_sim_queue)   # fast: sim feels "real time"
        self.after(500, self._poll_hardware)    # slower: hardware sampling has overhead

    # ---- simulation panel ----
    def _build_sim_panel(self):
        frame = ttk.LabelFrame(self, text="Simulation")
        frame.pack(fill="x", padx=10, pady=10)

        self.sim_time_var = tk.StringVar(value="t = 0.00")
        self.queue_len_var = tk.StringVar(value="Queue length: 0")
        ttk.Label(frame, textvariable=self.sim_time_var).pack(anchor="w")
        ttk.Label(frame, textvariable=self.queue_len_var).pack(anchor="w")

        self.canvas = tk.Canvas(frame, height=80, bg="white")
        self.canvas.pack(fill="x", padx=5, pady=5)

    # ---- hardware panel ----
    def _build_hw_panel(self):
        frame = ttk.LabelFrame(self, text="Hardware")
        frame.pack(fill="x", padx=10, pady=10)

        self.bars = {}
        keys = ["CPU", "RAM", "Disk"] + (["GPU"] if GPU_AVAILABLE else [])
        for key in keys:
            row = ttk.Frame(frame)
            row.pack(fill="x", padx=5, pady=2)
            ttk.Label(row, text=key, width=6).pack(side="left")
            bar = ttk.Progressbar(row, length=400, maximum=100)
            bar.pack(side="left", padx=5)
            pct = ttk.Label(row, text="0%")
            pct.pack(side="left")
            self.bars[key] = (bar, pct)

    # ---- pollers (both driven by self.after, both on the main thread) ----
    def _poll_sim_queue(self):
        try:
            while True:  # drain everything queued since the last poll
                evt = self.event_q.get_nowait()
                self._handle_event(evt)
        except queue.Empty:
            pass
        self.after(100, self._poll_sim_queue)

    def _handle_event(self, evt: SimEvent):
        self.sim_time_var.set(f"t = {evt.sim_time:.2f}")
        if evt.kind == "job_start":
            self.queue_len_var.set(f"Queue length: {evt.payload['queue_len']}")
            self._draw_job(evt.payload["id"], active=True)
        elif evt.kind == "job_end":
            self._draw_job(evt.payload["id"], active=False)

    def _draw_job(self, jid, active):
        x = 10 + (jid % 30) * 20
        if active:
            self.active_jobs[jid] = self.canvas.create_oval(x, 20, x + 15, 35, fill="#4caf50")
        else:
            item = self.active_jobs.pop(jid, None)
            if item:
                self.canvas.delete(item)

    def _poll_hardware(self):
        stats = sample_hardware()
        self._set_bar("CPU", stats["cpu_total"])
        self._set_bar("RAM", stats["mem"])
        self._set_bar("Disk", stats["disk"])
        if GPU_AVAILABLE and "gpu_util" in stats:
            self._set_bar("GPU", stats["gpu_util"])
        self.after(500, self._poll_hardware)

    def _set_bar(self, key, value):
        bar, pct = self.bars[key]
        bar["value"] = value
        pct.config(text=f"{value:.0f}%")

    def _on_close(self):
        self.stop_flag.set()
        self.worker.join(timeout=1)
        self.destroy()


if __name__ == "__main__":
    Dashboard().mainloop()