"""
Multi-tenant GPU scheduling simulation (SimPy) + Tkinter GUI + live PC
hardware stats. GPU count is fixed by the user BEFORE clicking Start; it
cannot change while a simulation is running.

Install:
    pip install simpy psutil
    pip install pynvml   # optional, only for NVIDIA GPU stats
"""

import queue
import random
import threading
import tkinter as tk
from tkinter import ttk

import psutil
import simpy

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

TENANTS = {"TenantA": 1 / 2.0, "TenantB": 1 / 3.0, "TenantC": 1 / 5.0}
TENANT_COLORS = {"TenantA": "#4caf50", "TenantB": "#2196f3", "TenantC": "#ff9800"}


# ---------------------------------------------------------------------------
# 1. Simulation -- runs entirely on the worker thread, capacity is fixed
#    for the lifetime of one run (set once, at construction time).
# ---------------------------------------------------------------------------

class GPUPool:
    """Fixed-capacity pool that also tracks WHICH slot id a tenant holds,
    so the GUI can draw a stable grid of boxes (box #3 always means the
    same box for the whole run)."""

    def __init__(self, env, capacity):
        self.resource = simpy.Resource(env, capacity=capacity)
        self.free_ids = list(range(capacity))
        self.owner = {}  # gpu_id -> tenant name

    def request(self, tenant):
        req = self.resource.request()
        yield req
        gpu_id = self.free_ids.pop()
        self.owner[gpu_id] = tenant
        return req, gpu_id

    def release(self, req, gpu_id):
        del self.owner[gpu_id]
        self.free_ids.append(gpu_id)
        self.resource.release(req)


class MultiTenantSimulation:
    def __init__(self, event_q, stop_flag, capacity, tenants):
        self.event_q, self.stop_flag = event_q, stop_flag
        self.env = simpy.Environment()
        self.pool = GPUPool(self.env, capacity)
        self.capacity = capacity
        self.tenants = tenants
        self.stats = {t: {"queued": 0, "running": 0, "completed": 0} for t in tenants}

    def job(self, tenant, duration):
        self.stats[tenant]["queued"] += 1
        req, gpu_id = yield from self.pool.request(tenant)
        self.stats[tenant]["queued"] -= 1
        self.stats[tenant]["running"] += 1
        self._snapshot()
        yield self.env.timeout(duration)
        self.pool.release(req, gpu_id)
        self.stats[tenant]["running"] -= 1
        self.stats[tenant]["completed"] += 1
        self._snapshot()

    def arrivals(self, tenant, rate):
        while True:
            yield self.env.timeout(random.expovariate(rate))
            self.env.process(self.job(tenant, random.uniform(1, 5)))

    def _snapshot(self):
        self.event_q.put({
            "sim_time": self.env.now,
            "capacity": self.capacity,
            "gpu_owner": dict(self.pool.owner),
            "tenant_stats": {t: dict(s) for t, s in self.stats.items()},
        })

    def run(self):
        for tenant, rate in self.tenants.items():
            self.env.process(self.arrivals(tenant, rate))
        while not self.stop_flag.is_set():
            self.env.step()


def simulation_thread_main(event_q, stop_flag, capacity, tenants):
    sim = MultiTenantSimulation(event_q, stop_flag, capacity, tenants)
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
            stats["gpu_util"] = util.gpu
        except Exception:
            pass
    return stats


# ---------------------------------------------------------------------------
# 3. GUI -- only this thread ever touches Tkinter widgets
# ---------------------------------------------------------------------------

class Dashboard(tk.Tk):
    GRID_COLS = 8
    BOX = 50  # px per GPU box

    def __init__(self):
        super().__init__()
        self.title("Multi-Tenant GPU Scheduler")
        self.geometry("640x640")

        self.event_q = None
        self.stop_flag = None
        self.worker = None

        self._build_config_panel()
        self._build_gpu_grid_panel()
        self._build_tenant_panel()
        self._build_hw_panel()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(500, self._poll_hardware)  # hardware panel runs regardless of sim state

    # ---- setup / control panel ----
    def _build_config_panel(self):
        frame = ttk.LabelFrame(self, text="Configuration")
        frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame, text="Number of GPUs:").pack(side="left", padx=5)
        self.gpu_count_var = tk.IntVar(value=4)
        self.gpu_spinbox = ttk.Spinbox(frame, from_=1, to=32, width=5,
                                        textvariable=self.gpu_count_var)
        self.gpu_spinbox.pack(side="left", padx=5)

        self.start_btn = ttk.Button(frame, text="Start", command=self._start_simulation)
        self.start_btn.pack(side="left", padx=10)
        self.stop_btn = ttk.Button(frame, text="Stop", command=self._stop_simulation,
                                    state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        self.status_var = tk.StringVar(value="Idle -- choose GPU count and click Start")
        ttk.Label(frame, textvariable=self.status_var).pack(side="left", padx=15)

    # ---- GPU grid panel ----
    def _build_gpu_grid_panel(self):
        frame = ttk.LabelFrame(self, text="GPU Grid")
        frame.pack(fill="x", padx=10, pady=10)
        self.sim_time_var = tk.StringVar(value="t = 0.00")
        ttk.Label(frame, textvariable=self.sim_time_var).pack(anchor="w")
        self.grid_canvas = tk.Canvas(frame, height=220, bg="white")
        self.grid_canvas.pack(fill="x", padx=5, pady=5)

    # ---- tenant stats panel ----
    def _build_tenant_panel(self):
        frame = ttk.LabelFrame(self, text="Tenants")
        frame.pack(fill="x", padx=10, pady=10)
        self.tenant_vars = {}
        for tenant in TENANTS:
            row = ttk.Frame(frame)
            row.pack(fill="x", padx=5, pady=2)
            swatch = tk.Canvas(row, width=14, height=14, highlightthickness=0)
            swatch.create_rectangle(0, 0, 14, 14, fill=TENANT_COLORS[tenant], outline="")
            swatch.pack(side="left", padx=(0, 5))
            ttk.Label(row, text=tenant, width=10).pack(side="left")
            var = tk.StringVar(value="queued: 0   running: 0   completed: 0")
            ttk.Label(row, textvariable=var).pack(side="left")
            self.tenant_vars[tenant] = var

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

    # ---- lifecycle: start / stop ----
    def _set_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.gpu_spinbox.configure(state=state)
        self.start_btn.configure(state=state)
        self.stop_btn.configure(state="disabled" if enabled else "normal")

    def _start_simulation(self):
        try:
            capacity = int(self.gpu_count_var.get())
        except (tk.TclError, ValueError):
            self.status_var.set("Enter a valid integer GPU count")
            return
        if capacity < 1:
            self.status_var.set("GPU count must be at least 1")
            return

        self._set_controls_enabled(False)
        self.status_var.set(f"Running with {capacity} GPU(s)...")

        self.event_q = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = threading.Thread(
            target=simulation_thread_main,
            args=(self.event_q, self.stop_flag, capacity, TENANTS),
            daemon=True,
        )
        self.worker.start()
        self.after(100, self._poll_sim_queue)

    def _stop_simulation(self):
        if self.stop_flag:
            self.stop_flag.set()
        self.status_var.set("Stopping...")
        self.after(50, self._check_worker_stopped)

    def _check_worker_stopped(self):
        if self.worker and self.worker.is_alive():
            self.after(50, self._check_worker_stopped)
            return
        self.status_var.set("Idle -- choose GPU count and click Start")
        self._set_controls_enabled(True)

    # ---- pollers ----
    def _poll_sim_queue(self):
        if self.event_q is None:
            return
        snapshot = None
        try:
            while True:  # only the latest snapshot matters for drawing
                snapshot = self.event_q.get_nowait()
        except queue.Empty:
            pass
        if snapshot:
            self._render_snapshot(snapshot)
        if self.worker and self.worker.is_alive():
            self.after(100, self._poll_sim_queue)

    def _render_snapshot(self, snap):
        self.sim_time_var.set(f"t = {snap['sim_time']:.2f}")
        self._draw_grid(snap["capacity"], snap["gpu_owner"])
        for tenant, s in snap["tenant_stats"].items():
            self.tenant_vars[tenant].set(
                f"queued: {s['queued']}   running: {s['running']}   completed: {s['completed']}"
            )

    def _draw_grid(self, capacity, gpu_owner):
        self.grid_canvas.delete("all")
        pad = 6
        for gpu_id in range(capacity):
            col = gpu_id % self.GRID_COLS
            row = gpu_id // self.GRID_COLS
            x0 = pad + col * (self.BOX + pad)
            y0 = pad + row * (self.BOX + pad)
            x1, y1 = x0 + self.BOX, y0 + self.BOX
            tenant = gpu_owner.get(gpu_id)
            color = TENANT_COLORS.get(tenant, "#e0e0e0")  # gray = idle
            self.grid_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#999999")
            self.grid_canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=str(gpu_id),
                                          fill="white" if tenant else "black")
        rows_needed = (capacity + self.GRID_COLS - 1) // self.GRID_COLS
        self.grid_canvas.configure(height=max(80, rows_needed * (self.BOX + pad) + pad))

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
        if self.stop_flag:
            self.stop_flag.set()
        if self.worker:
            self.worker.join(timeout=1)
        self.destroy()


if __name__ == "__main__":
    Dashboard().mainloop()