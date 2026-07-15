import tkinter as tk
from tkinter import ttk
import queue
import threading

from GPUMonitoring.monitoring import GPU_COUNT, feed_hardware, GPU_AVAILABLE
from simulation.simulator import SimEvent, simulation_thread_main
from Workload.workload import DEFAULT_TENANTS, WORKLOAD_TYPES
from Scheduling_control.loader import load_schedulers

class Dashboard(tk.Tk):
    GRID_COLS = 8
    BOX = 50  # px per GPU box in the grid

    def __init__(self):
        super().__init__()
        self.title("SimPy Simulation + Hardware Monitor")
        self.geometry("1280x720")

        self.event_q = None
        self.stop_flag = None
        self.worker = None

        # Scheduling algorithms the user can pick; each decides how tenants map
        # to GPUs/servers. Discovered from the schedulers/ plugin directory and
        # refreshable at runtime via the "Reload schedulers" button. A fresh
        # instance is created per run (some are stateful).
        self.schedulers = load_schedulers()

        # Tenants that generate the simulated workload, plus a color lookup
        # used to shade the GPU grid by owning tenant.
        self.tenants = DEFAULT_TENANTS
        self.tenant_colors = {t.tenant_id: t.color for t in self.tenants}

        self._build_config_panel()
        self._build_gpu_grid_panel()
        self._build_queue_panel()
        self._build_tenant_panel()
        self._build_hw_panel()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_hardware)


    def _build_config_panel(self):
        frame = ttk.LabelFrame(self, text="Configuration")
        frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(frame, text="Number of GPUs:").pack(side="left", padx=5)
        self.gpu_count_var = tk.IntVar(value=4)
        self.gpu_spinbox = ttk.Spinbox(frame, from_=1, to=16, textvariable=self.gpu_count_var, width=5)
        self.gpu_spinbox.pack(side="left", padx=5)

        ttk.Label(frame, text="Scheduler:").pack(side="left", padx=(15, 5))
        self.scheduler_var = tk.StringVar()
        self.scheduler_combo = ttk.Combobox(
            frame, textvariable=self.scheduler_var, state="readonly", width=28,
        )
        self.scheduler_combo.pack(side="left", padx=5)
        self._refresh_scheduler_combo()

        self.reload_btn = ttk.Button(frame, text="Reload schedulers",
                                     command=self._reload_schedulers)
        self.reload_btn.pack(side="left", padx=5)

        self.start_btn = ttk.Button(frame, text="Start Simulation", command=self._start_simulation)
        self.start_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(frame, text="Stop Simulation", command=self._stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(side="left", padx=5)
        
        self.status_var = tk.StringVar(value="Simulation not started. -- Choose GPU count and click 'Start'")
        ttk.Label(frame, textvariable=self.status_var).pack(side="left", padx=10)

    def _refresh_scheduler_combo(self):
        """Point the dropdown at the current self.schedulers, preserving the
        user's selection when that scheduler still exists."""
        names = list(self.schedulers)
        self.scheduler_combo["values"] = names
        if self.scheduler_var.get() not in self.schedulers:
            self.scheduler_var.set(names[0] if names else "")

    def _reload_schedulers(self):
        """Rescan the schedulers/ directory so newly written or edited plugins
        appear without restarting the app."""
        self.schedulers = load_schedulers()
        self._refresh_scheduler_combo()
        self.status_var.set(f"Loaded {len(self.schedulers)} scheduler(s) from schedulers/")

    def _build_gpu_grid_panel(self):
        frame = ttk.LabelFrame(self, text="GPU Grid")
        frame.pack(fill="x", padx=10, pady=(0, 10))
        self.sim_time_var = tk.StringVar(value="t = 0.00")
        ttk.Label(frame, textvariable=self.sim_time_var).pack(anchor="w")
        self.grid_canvas = tk.Canvas(frame, height=120, bg="white")
        self.grid_canvas.pack(fill="x", padx=5, pady=5)

    def _build_queue_panel(self):
        frame = ttk.LabelFrame(self, text="Job Queue  (generated jobs and their assigned GPU)")
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        cols = ("job", "tenant", "type", "size", "iters", "state", "gpu")
        headings = {"job": "Job #", "tenant": "Tenant", "type": "Type",
                    "size": "Matrix", "iters": "Iters", "state": "State",
                    "gpu": "GPU"}
        widths = {"job": 60, "tenant": 90, "type": 70, "size": 80,
                  "iters": 60, "state": 90, "gpu": 60}

        self.queue_tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        for c in cols:
            self.queue_tree.heading(c, text=headings[c])
            self.queue_tree.column(c, width=widths[c], anchor="center")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.queue_tree.yview)
        self.queue_tree.configure(yscrollcommand=vsb.set)
        self.queue_tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        vsb.pack(side="right", fill="y")

        # Shade each row by its owning tenant so the queue is scannable at a glance.
        for tenant in self.tenants:
            self.queue_tree.tag_configure(tenant.tenant_id, background=tenant.color)

    def _build_tenant_panel(self):
        frame = ttk.LabelFrame(self, text="Tenants")
        frame.pack(fill="x", padx=10, pady=(0, 10))
        self.tenant_vars = {}
        for tenant in self.tenants:
            row = ttk.Frame(frame)
            row.pack(fill="x", padx=5, pady=2)
            swatch = tk.Canvas(row, width=14, height=14, highlightthickness=0)
            swatch.create_rectangle(0, 0, 14, 14, fill=tenant.color, outline="")
            swatch.pack(side="left", padx=(0, 5))
            ttk.Label(row, text=tenant.tenant_id, width=10).pack(side="left")
            var = tk.StringVar(value="queued: 0   running: 0   completed: 0")
            ttk.Label(row, textvariable=var).pack(side="left")
            self.tenant_vars[tenant.tenant_id] = var

    def _build_hw_panel(self):
        frame = ttk.LabelFrame(self, text="Hardware")
        frame.pack(fill="x", padx=10, pady=10)
        self.hardware_frame = frame
        self.bars = {}

        # System percentage bars.
        for key in ["CPU", "RAM", "Disk"]:
            row = ttk.Frame(frame)
            row.pack(fill="x", padx=5, pady=2)
            ttk.Label(row, text=key, width=10).pack(side="left")
            bar = ttk.Progressbar(row, length=400, maximum=100)
            bar.pack(side="left", padx=5)
            pct = ttk.Label(row, text="0%")
            pct.pack(side="left")
            self.bars[key] = (bar, pct)

        # GPU panels are built lazily as GPUs report in, keyed by index.
        self.gpu_container = ttk.Frame(frame)
        self.gpu_container.pack(fill="x", padx=5, pady=(6, 2))
        self.gpu_panels = {}
        
    def _set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.gpu_spinbox.config(state=state)
        # readonly combobox re-enables to "readonly", not plain NORMAL
        self.scheduler_combo.config(state="readonly" if enabled else tk.DISABLED)
        self.reload_btn.config(state=state)
        self.start_btn.config(state=state)
        self.stop_btn.config(state=tk.NORMAL if not enabled else tk.DISABLED)
        
    def _start_simulation(self):
        try:
            capacity = int(self.gpu_count_var.get())
        except (tk.TclError, ValueError):
            self.status_var.set("Enter a valid integer GPU count")
            return
        if capacity < 1:
            self.status_var.set("GPU count must be at least 1")
            return

        name = self.scheduler_var.get()
        if name not in self.schedulers:
            self.status_var.set("No scheduler selected -- add one to schedulers/ and click 'Reload schedulers'")
            return
        scheduler = self.schedulers[name]()

        self._set_controls_enabled(False)
        self.status_var.set(f"Running with {capacity} GPU(s) -- {self.scheduler_var.get()}")

        self.event_q = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = threading.Thread(
            target=simulation_thread_main,
            args=(self.event_q, self.stop_flag, capacity, self.tenants, scheduler),
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
        self.status_var.set("Simulation not started. -- Choose GPU count and click 'Start'")
        self._set_controls_enabled(True)

    # ---- simulation snapshot polling / rendering ----
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
        # Map each busy GPU to the one-letter tag of the workload running on it,
        # so the grid box shows both who owns the GPU (color) and what it's
        # running (letter).
        gpu_types = {}
        for j in snap.get("jobs", []):
            if j.get("gpu") is not None and j.get("state") == "running":
                wt = WORKLOAD_TYPES.get(j.get("wtype"))
                gpu_types[j["gpu"]] = wt.label if wt else "?"
        self._draw_grid(snap["capacity"], snap["gpu_owner"], gpu_types)
        for tenant_id, s in snap["tenant_stats"].items():
            if tenant_id in self.tenant_vars:
                self.tenant_vars[tenant_id].set(
                    f"queued: {s['queued']}   running: {s['running']}   completed: {s['completed']}"
                )
        self._render_queue(snap.get("jobs", []))

    def _render_queue(self, jobs):
        """Rebuild the job-queue table from the latest snapshot. Active jobs
        (queued + running) are shown; completed jobs leave the queue."""
        tree = self.queue_tree
        tree.delete(*tree.get_children())
        for j in sorted(jobs, key=lambda r: r["job_id"]):
            gpu = j["gpu"]
            tree.insert(
                "", "end",
                values=(j["job_id"], j["tenant"], (j.get("wtype") or "—").capitalize(),
                        f"{j['size']}²", j["iters"], j["state"],
                        "—" if gpu is None else gpu),
                tags=(j["tenant"],),
            )

    def _draw_grid(self, capacity, gpu_owner, gpu_types=None):
        gpu_types = gpu_types or {}
        self.grid_canvas.delete("all")
        pad = 6
        for gpu_id in range(capacity):
            col = gpu_id % self.GRID_COLS
            row = gpu_id // self.GRID_COLS
            x0 = pad + col * (self.BOX + pad)
            y0 = pad + row * (self.BOX + pad)
            x1, y1 = x0 + self.BOX, y0 + self.BOX
            tenant_id = gpu_owner.get(gpu_id)
            color = self.tenant_colors.get(tenant_id, "#e0e0e0")  # gray = idle
            fg = "white" if tenant_id else "black"
            self.grid_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#999999")
            # GPU id near the top, workload-type letter (L/M/H) below it.
            self.grid_canvas.create_text((x0 + x1) / 2, y0 + self.BOX * 0.33,
                                         text=str(gpu_id), fill=fg)
            letter = gpu_types.get(gpu_id)
            if letter:
                self.grid_canvas.create_text((x0 + x1) / 2, y0 + self.BOX * 0.68,
                                             text=letter, fill=fg,
                                             font=("TkDefaultFont", 12, "bold"))
        rows_needed = (capacity + self.GRID_COLS - 1) // self.GRID_COLS
        self.grid_canvas.configure(height=max(80, rows_needed * (self.BOX + pad) + pad))


    def _ensure_gpu_panel(self, index, name):
        """Create (once) and return the widget group for a single GPU."""
        if index in self.gpu_panels:
            return self.gpu_panels[index]

        title = f"GPU {index}" + (f": {name}" if name else "")
        frame = ttk.LabelFrame(self.gpu_container, text=title)
        frame.pack(fill="x", pady=4)

        def _bar_row(label_text):
            row = ttk.Frame(frame)
            row.pack(fill="x", padx=5, pady=1)
            ttk.Label(row, text=label_text, width=6).pack(side="left")
            bar = ttk.Progressbar(row, length=320, maximum=100)
            bar.pack(side="left", padx=5)
            value = ttk.Label(row, text="—", width=24, anchor="w")
            value.pack(side="left")
            return bar, value

        util_bar, util_val = _bar_row("Util")
        mem_bar, mem_val = _bar_row("Mem")
        fan_bar, fan_val = _bar_row("Fan")

        stat_row = ttk.Frame(frame)
        stat_row.pack(fill="x", padx=5, pady=(1, 4))
        stat_lbl = ttk.Label(stat_row, text="—", anchor="w")
        stat_lbl.pack(side="left")

        panel = {
            "util_bar": util_bar, "util_val": util_val,
            "mem_bar": mem_bar, "mem_val": mem_val,
            "fan_bar": fan_bar, "fan_val": fan_val,
            "stat_lbl": stat_lbl,
        }
        self.gpu_panels[index] = panel
        return panel

    def _update_gpu_panel(self, gpu):
        """Push one GPU's stats dict into its panel, creating it if needed."""
        panel = self._ensure_gpu_panel(gpu["index"], gpu.get("name"))

        util = gpu.get("gpu_util")
        if util is not None:
            panel["util_bar"]["value"] = util
            panel["util_val"].config(text=f"{util:.0f}%")

        mem_pct = gpu.get("mem_percent")
        if mem_pct is not None:
            panel["mem_bar"]["value"] = mem_pct
            used, total = gpu.get("mem_used_mb"), gpu.get("mem_total_mb")
            if used is not None and total is not None:
                panel["mem_val"].config(text=f"{mem_pct:.0f}%  ({used} / {total} MB)")
            else:
                panel["mem_val"].config(text=f"{mem_pct:.0f}%")

        fan = gpu.get("fan_percent")
        if fan is not None:
            panel["fan_bar"]["value"] = fan
            panel["fan_val"].config(text=f"{fan}%")
        else:
            panel["fan_val"].config(text="n/a")

        parts = []
        if gpu.get("temp_c") is not None:
            parts.append(f"Temp {gpu['temp_c']}°C")
        if gpu.get("power_w") is not None:
            if gpu.get("power_limit_w") is not None:
                parts.append(f"Power {gpu['power_w']} / {gpu['power_limit_w']} W")
            else:
                parts.append(f"Power {gpu['power_w']} W")
        if gpu.get("clock_graphics_mhz") is not None:
            parts.append(f"Gfx {gpu['clock_graphics_mhz']} MHz")
        if gpu.get("clock_mem_mhz") is not None:
            parts.append(f"MemClk {gpu['clock_mem_mhz']} MHz")
        panel["stat_lbl"].config(text="    ".join(parts) if parts else "—")

    def _set_bar(self, key, value):
        bar, pct = self.bars[key]
        bar["value"] = value
        pct.config(text=f"{value:.0f}%")


    def _poll_hardware(self):
        stats = feed_hardware()
        self._set_bar("CPU", stats["cpu_total"])
        self._set_bar("RAM", stats["mem"])
        self._set_bar("Disk", stats["disk"])

        if GPU_AVAILABLE:
            for gpu in stats.get("gpus", []):
                self._update_gpu_panel(gpu)

        self.after(500, self._poll_hardware)
        
    def _on_close(self):
        if self.stop_flag:
            self.stop_flag.set()
        if self.worker:
            self.worker.join(timeout=1)
        self.destroy()