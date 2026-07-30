import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import queue
import threading

from GPUMonitoring.monitoring import GPU_COUNT, feed_hardware, GPU_AVAILABLE
from simulation.simulator import SimEvent, simulation_thread_main
from simulation.results import export_run
from Workload.workload import (DEFAULT_ARRIVAL_RATE, DEFAULT_ITER_MULTIPLIER,
                               WORKLOAD_TYPES, make_tenants, generate_trace)
from Scheduling_control.loader import load_schedulers

class Dashboard(tk.Tk):
    GRID_COLS = 8
    BOX = 50  # px per GPU box in the grid

    def __init__(self):
        super().__init__()
        self.title("SimPy Simulation + Hardware Monitor")
        # Wide enough for the Configuration + Hardware columns to sit side by
        # side (config needs ~800px, Hardware ~510px) without clipping.
        self.geometry("1440x820")

        self.event_q = None
        self.stop_flag = None
        self.worker = None
        self._user_stopped = False

        # Artifacts of the most recent completed run: the raw per-job rows,
        # the admission/throttle logs, the measured telemetry timeline (all
        # for CSV export) and the folded summary (for the results view).
        # Filled from the engine's final done=True snapshot; a label
        # identifies the run.
        self.last_results = []
        self.last_summary = None
        self.last_admission_log = []
        self.last_throttle_log = []
        self.last_telemetry = []
        self.last_capacity = 1
        self._last_run_label = "run"

        # Scheduling algorithms the user can pick; each decides how tenants map
        # to GPUs/servers. Discovered from the schedulers/ plugin directory and
        # refreshable at runtime via the "Reload schedulers" button. A fresh
        # instance is created per run (some are stateful).
        self.schedulers = load_schedulers()

        # Everything lives inside a scrollable container so the full UI is
        # reachable even when it's taller than the window (each panel is packed
        # into self.content instead of directly into the root).
        self._build_scrollable_container()
        # Two-column top band: config / GPU grid / job flow stack on the left,
        # Hardware sits beside them on the right.
        self._build_columns()

        self._build_config_panel()

        # Tenants that generate the simulated workload, plus a color lookup used
        # to shade the GPU grid by owning tenant. The set is derived from the
        # experiment config and rebuilt on every run (see _apply_tenant_config).
        self.tenants = make_tenants(self.num_tenants_var.get())
        self.tenant_colors = {t.tenant_id: t.color for t in self.tenants}

        self._build_gpu_grid_panel()
        self._build_flow_panel()
        self._build_hw_panel()  # right column, beside config / grid / flow
        # Wider panels pack full-width below the two-column band.
        self._build_admission_panel()
        self._build_queue_panel()
        self._build_tenant_panel()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_hardware)


    def _build_scrollable_container(self):
        """Host all panels inside a scrollable Canvas so the whole UI stays
        reachable when it's taller than the window. Panels pack into
        self.content (the inner frame) rather than the root window."""
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.content = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=self.content, anchor="nw")

        # Keep the scrollregion in sync with the content's real size, and stretch
        # the inner frame to the canvas width so horizontal fill/pack still works.
        self.content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(window, width=e.width))

        # Mouse-wheel scrolling while the pointer is over the canvas area.
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")

        def _bind_wheel(_):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_wheel(_):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)
        self._scroll_canvas = canvas

    def _build_columns(self):
        """Split the top of the UI into two columns. The left column stacks the
        Configuration, GPU Grid and Job Flow panels; the right column holds the
        Hardware panel beside them. The wider panels (Admission, Job Queue,
        Tenants) pack full-width into self.content below this band."""
        top = ttk.Frame(self.content)
        top.pack(fill="x")
        self.left_col = ttk.Frame(top)
        self.left_col.pack(side="left", fill="both", expand=True)
        self.right_col = ttk.Frame(top)
        self.right_col.pack(side="left", fill="y", anchor="n")

    def _build_config_panel(self):
        frame = ttk.LabelFrame(self.left_col, text="Configuration")
        frame.pack(fill=tk.X, padx=10, pady=10)

        # --- row 1: hardware + scheduler + run controls ---
        row1 = ttk.Frame(frame)
        row1.pack(fill="x", padx=5, pady=(4, 2))

        # Each "GPU" slot is one simulated MIG partition (§3.8 iii): an equal
        # slice of the physical VRAM, served by its own worker thread.
        ttk.Label(row1, text="Partitions (GPUs):").pack(side="left", padx=5)
        self.gpu_count_var = tk.IntVar(value=4)
        self.gpu_spinbox = ttk.Spinbox(row1, from_=1, to=16, textvariable=self.gpu_count_var, width=5)
        self.gpu_spinbox.pack(side="left", padx=5)

        ttk.Label(row1, text="Scheduler:").pack(side="left", padx=(15, 5))
        self.scheduler_var = tk.StringVar()
        self.scheduler_combo = ttk.Combobox(
            row1, textvariable=self.scheduler_var, state="readonly", width=28,
        )
        self.scheduler_combo.pack(side="left", padx=5)
        self._refresh_scheduler_combo()

        self.reload_btn = ttk.Button(row1, text="Reload schedulers",
                                     command=self._reload_schedulers)
        self.reload_btn.pack(side="left", padx=5)

        self.start_btn = ttk.Button(row1, text="Start Simulation", command=self._start_simulation)
        self.start_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(row1, text="Stop Simulation", command=self._stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(side="left", padx=5)

        # --- row 2: experiment config (defines the finite, seeded job trace) ---
        row2 = ttk.Frame(frame)
        row2.pack(fill="x", padx=5, pady=(2, 4))

        # A run is fully determined by (number of tenants, per-tenant counts of
        # each workload type, seed). These are the knobs for that.
        ttk.Label(row2, text="Tenants:").pack(side="left", padx=(5, 2))
        self.num_tenants_var = tk.IntVar(value=3)
        self.tenants_spinbox = ttk.Spinbox(row2, from_=1, to=12, width=4,
                                           textvariable=self.num_tenants_var)
        self.tenants_spinbox.pack(side="left", padx=(0, 10))

        # Per-tenant counts for each workload type; each tenant contributes this
        # many light / medium / heavy jobs to the trace.
        self.count_vars = {}
        self.count_spinboxes = []
        for name, default in (("light", 4), ("medium", 3), ("heavy", 1)):
            ttk.Label(row2, text=f"{name.capitalize()}/tenant:").pack(side="left", padx=(5, 2))
            var = tk.IntVar(value=default)
            sb = ttk.Spinbox(row2, from_=0, to=99, width=4, textvariable=var)
            sb.pack(side="left", padx=(0, 8))
            self.count_vars[name] = var
            self.count_spinboxes.append(sb)

        ttk.Label(row2, text="Seed:").pack(side="left", padx=(10, 2))
        self.seed_var = tk.IntVar(value=0)
        self.seed_spinbox = ttk.Spinbox(row2, from_=0, to=999999, width=8,
                                        textvariable=self.seed_var)
        self.seed_spinbox.pack(side="left", padx=(0, 10))

        # Whole-experiment compute-intensity scale: every job runs this many
        # times its drawn matmul iterations (e.g. 1000 => 1000x the iterations).
        # Scales wall-clock work linearly without touching memory or arrivals.
        ttk.Label(row2, text="Iterations x:").pack(side="left", padx=(5, 2))
        self.iter_mult_var = tk.IntVar(value=DEFAULT_ITER_MULTIPLIER)
        self.iter_mult_spinbox = ttk.Spinbox(row2, from_=1, to=100000, increment=1,
                                             width=8, textvariable=self.iter_mult_var)
        self.iter_mult_spinbox.pack(side="left", padx=(0, 5))

        # --- row 3: §3.8 knobs -- arrival intensity + admission control ---
        row3 = ttk.Frame(frame)
        row3.pack(fill="x", padx=5, pady=(2, 4))

        ttk.Label(row3, text="Arrival rate (jobs/s/tenant):").pack(side="left", padx=(5, 2))
        self.arrival_rate_var = tk.DoubleVar(value=DEFAULT_ARRIVAL_RATE)
        self.arrival_spinbox = ttk.Spinbox(row3, from_=0.05, to=20.0, increment=0.05,
                                           width=6, textvariable=self.arrival_rate_var)
        self.arrival_spinbox.pack(side="left", padx=(0, 10))

        ttk.Label(row3, text="Adm. threshold %:").pack(side="left", padx=(5, 2))
        self.threshold_var = tk.IntVar(value=90)
        self.threshold_spinbox = ttk.Spinbox(row3, from_=10, to=100, increment=5,
                                             width=5, textvariable=self.threshold_var)
        self.threshold_spinbox.pack(side="left", padx=(0, 10))

        ttk.Label(row3, text="Safety margin (MB):").pack(side="left", padx=(5, 2))
        self.margin_var = tk.IntVar(value=64)
        self.margin_spinbox = ttk.Spinbox(row3, from_=0, to=2048, increment=16,
                                          width=6, textvariable=self.margin_var)
        self.margin_spinbox.pack(side="left", padx=(0, 10))

        self.status_var = tk.StringVar(value="Simulation not started. -- Configure the experiment and click 'Start'")
        ttk.Label(frame, textvariable=self.status_var).pack(anchor="w", padx=10, pady=(0, 4))

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
        frame = ttk.LabelFrame(self.left_col, text="GPU Grid")
        frame.pack(fill="x", padx=10, pady=(0, 10))
        self.sim_time_var = tk.StringVar(value="t = 0.00")
        ttk.Label(frame, textvariable=self.sim_time_var).pack(anchor="w")
        self.grid_canvas = tk.Canvas(frame, height=120, bg="white")
        self.grid_canvas.pack(fill="x", padx=5, pady=5)

    def _build_flow_panel(self):
        """Live job-flow view: jobs arrive into the global queue on the left and
        fan out to the per-GPU server lanes on the right, so distribution across
        GPUs is visible as it happens. Driven entirely by the same snapshot the
        poll loop already delivers -- no extra engine data or threading."""
        frame = ttk.LabelFrame(
            self.left_col,
            text="Job Flow  (arrivals -> global queue -> GPU distribution)")
        frame.pack(fill="x", padx=10, pady=(0, 10))
        self.flow_canvas = tk.Canvas(frame, height=140, bg="white",
                                     highlightthickness=0)
        self.flow_canvas.pack(fill="x", padx=5, pady=5)
        # Job-flow animation state. `_flow_prev` maps job_id -> the GPU it was
        # placed on last frame (None if unplaced), so the next frame can detect
        # a fresh assignment and glide a token along the connector. `_flow_seq`
        # hands out a unique canvas tag per in-flight animation.
        self._flow_prev = {}
        self._flow_seq = 0

    # px geometry for the flow panel
    FLOW_TOK_W = 34
    FLOW_TOK_H = 22

    def _draw_flow_token(self, c, x, y, job, running=False, tags="static"):
        """Draw one job as a tenant-colored box stamped with its id + workload
        letter (L/M/H), matching the grid's color/letter scheme. The running job
        gets a bold dark outline so it stands out from the queued ones behind it."""
        w, h = self.FLOW_TOK_W, self.FLOW_TOK_H
        tenant = job.get("tenant")
        color = self.tenant_colors.get(tenant, "#e0e0e0")
        wt = WORKLOAD_TYPES.get(job.get("wtype"))
        letter = wt.label if wt else "?"
        outline, width = ("#333333", 2) if running else ("#aaaaaa", 1)
        c.create_rectangle(x, y, x + w, y + h, fill=color, outline=outline,
                           width=width, tags=tags)
        c.create_text(x + w / 2, y + h / 2, text=f"{job['job_id']}{letter}",
                      fill="white" if tenant else "black",
                      font=("TkDefaultFont", 8, "bold"), tags=tags)

    def _draw_flow(self, snap):
        """Render the job-flow panel from a snapshot: queued jobs stacked in the
        global-queue column, and each GPU's lane showing its running job plus the
        jobs waiting in that server's queue behind it (FIFO order). Arrows fan
        from the queue to the lanes, and a token is animated along the arrow
        whenever a job is newly assigned to a GPU.

        Only the "static" layer is redrawn each snapshot; in-flight assignment
        animations carry the "anim" tag and are left untouched, so a redraw
        landing mid-glide doesn't erase them."""
        c = self.flow_canvas
        c.delete("static")
        capacity = snap["capacity"]
        jobs = snap.get("jobs", [])

        queued = sorted((j for j in jobs if j.get("state") == "queued"),
                        key=lambda j: j["job_id"])
        running = {j["gpu"]: j for j in jobs
                   if j.get("state") == "running" and j.get("gpu") is not None}
        # Jobs waiting in each server's queue, ordered by when they were assigned
        # there (true FIFO order behind the running job).
        queued_by_gpu = {i: [] for i in range(capacity)}
        for j in jobs:
            g = j.get("gpu")
            if j.get("state") == "server-queue" and g is not None and 0 <= g < capacity:
                queued_by_gpu[g].append(j)
        for waiting in queued_by_gpu.values():
            waiting.sort(key=lambda j: (j.get("assign_time") or 0.0, j["job_id"]))

        pad = 10
        tok_w, tok_h = self.FLOW_TOK_W, self.FLOW_TOK_H
        top = pad + 14
        qx = pad
        max_show = 4
        lane_x = qx + tok_w + 100
        lane_w = 340
        lane_h = tok_h + 8
        src_x = qx + tok_w + 6
        src_y = top + tok_h / 2

        # Detect jobs assigned to a GPU since the last frame and glide a token
        # from the queue along the connector to the target lane. `_flow_prev`
        # holds None for a job that was still unplaced, so an int-now / None-before
        # transition is exactly a fresh placement.
        new_prev = {}
        for j in jobs:
            on_gpu = (j["gpu"] if j.get("state") in ("server-queue", "running")
                      and j.get("gpu") is not None else None)
            jid = j["job_id"]
            new_prev[jid] = on_gpu
            if (on_gpu is not None and 0 <= on_gpu < capacity
                    and self._flow_prev.get(jid) is None):
                self._start_flow_anim(j, (src_x, src_y),
                                      (lane_x + 4, top + on_gpu * lane_h))
        self._flow_prev = new_prev

        c.create_text(qx, pad - 2, text="GLOBAL QUEUE", anchor="nw",
                      font=("TkDefaultFont", 8, "bold"), fill="#333333",
                      tags="static")
        if queued:
            for k, j in enumerate(queued[:max_show]):
                self._draw_flow_token(c, qx, top + k * (tok_h + 4), j)
            if len(queued) > max_show:
                c.create_text(qx, top + max_show * (tok_h + 4),
                              text=f"+{len(queued) - max_show} more", anchor="nw",
                              font=("TkDefaultFont", 8), fill="#777777",
                              tags="static")
        else:
            c.create_text(qx, top, text="(empty)", anchor="nw",
                          font=("TkDefaultFont", 8), fill="#aaaaaa", tags="static")

        # Lanes: one per GPU, stacked on the right. Each shows the running job
        # (bold outline) then the jobs waiting in that server's queue.
        for gpu_id in range(capacity):
            ly = top + gpu_id * lane_h
            mid = ly + tok_h / 2
            c.create_line(src_x, src_y, lane_x - 4, mid, fill="#d0d0d0",
                          arrow="last", tags="static")
            c.create_rectangle(lane_x, ly, lane_x + lane_w, ly + tok_h,
                               fill="#f7f7f7", outline="#dddddd", tags="static")
            c.create_text(lane_x - 8, mid, text=f"GPU {gpu_id}", anchor="e",
                          font=("TkDefaultFont", 8, "bold"), tags="static")
            run = running.get(gpu_id)
            waiting = queued_by_gpu[gpu_id]
            x = lane_x + 4
            if run:
                self._draw_flow_token(c, x, ly, run, running=True)
                x += tok_w + 12
            elif not waiting:
                c.create_text(x + 4, mid, text="idle", anchor="w",
                              font=("TkDefaultFont", 8), fill="#bbbbbb",
                              tags="static")
            shown = 0
            for wj in waiting:
                if x + tok_w > lane_x + lane_w - 26:
                    break
                self._draw_flow_token(c, x, ly, wj, running=False)
                x += tok_w + 3
                shown += 1
            if shown < len(waiting):
                c.create_text(lane_x + lane_w - 4, mid,
                              text=f"+{len(waiting) - shown}", anchor="e",
                              font=("TkDefaultFont", 8, "bold"), fill="#ff8f00",
                              tags="static")

        # Keep any gliding assignment tokens above the freshly drawn static layer.
        c.tag_raise("anim")
        rows = max(capacity, max_show + 1)
        c.configure(height=max(120, top + rows * lane_h + pad))

    def _start_flow_anim(self, job, src, dst, steps=12):
        """Create a token at `src` and glide it to `dst` over `steps` frames to
        visualize a job being assigned to a GPU. The token carries the "anim"
        tag (so snapshot redraws leave it alone) plus a unique per-animation tag
        used to move and finally delete it."""
        c = self.flow_canvas
        self._flow_seq += 1
        tag = f"anim{self._flow_seq}"
        x0, y0 = src
        w, h = self.FLOW_TOK_W, self.FLOW_TOK_H
        tenant = job.get("tenant")
        color = self.tenant_colors.get(tenant, "#e0e0e0")
        wt = WORKLOAD_TYPES.get(job.get("wtype"))
        letter = wt.label if wt else "?"
        # Orange outline marks the token as in-transit (being placed).
        c.create_rectangle(x0, y0, x0 + w, y0 + h, fill=color, outline="#ff8f00",
                           width=2, tags=("anim", tag))
        c.create_text(x0 + w / 2, y0 + h / 2, text=f"{job['job_id']}{letter}",
                      fill="white" if tenant else "black",
                      font=("TkDefaultFont", 8, "bold"), tags=("anim", tag))
        dx = (dst[0] - x0) / steps
        dy = (dst[1] - y0) / steps
        self._step_flow_anim(tag, dx, dy, steps)

    def _step_flow_anim(self, tag, dx, dy, remaining):
        """Advance one in-flight assignment animation by a single frame, or
        remove its token once it reaches the lane."""
        c = self.flow_canvas
        try:
            if remaining <= 0:
                c.delete(tag)
                return
            c.move(tag, dx, dy)
        except tk.TclError:
            return  # canvas gone (window closing) -- stop quietly
        self.after(30, lambda: self._step_flow_anim(tag, dx, dy, remaining - 1))

    def _build_admission_panel(self):
        """Live admission-control view: outcome counters plus a rolling feed
        of the most recent scheduling decisions and throttle events."""
        frame = ttk.LabelFrame(self.content, text="Admission control  (scheduling decisions / throttling events)")
        frame.pack(fill="x", padx=10, pady=(0, 10))
        self.admission_counters_var = tk.StringVar(
            value="decisions: 0   admit: 0   queue: 0   reject: 0   throttle: 0   throttled now: —")
        ttk.Label(frame, textvariable=self.admission_counters_var).pack(
            anchor="w", padx=5, pady=(2, 0))
        self.admission_list = tk.Listbox(frame, height=5, font=("Consolas", 9),
                                         activestyle="none")
        self.admission_list.pack(fill="x", padx=5, pady=(2, 4))

    def _build_queue_panel(self):
        frame = ttk.LabelFrame(self.content, text="Job Queue  (generated jobs and their assigned GPU)")
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        cols = ("job", "tenant", "type", "size", "iters", "state", "gpu",
                "gwait", "swait")
        headings = {"job": "Job #", "tenant": "Tenant", "type": "Type",
                    "size": "Matrix", "iters": "Iters", "state": "State",
                    "gpu": "GPU", "gwait": "Global wait", "swait": "Server wait"}
        widths = {"job": 60, "tenant": 90, "type": 70, "size": 80,
                  "iters": 60, "state": 100, "gpu": 60, "gwait": 90, "swait": 90}

        self.queue_tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        for c in cols:
            self.queue_tree.heading(c, text=headings[c])
            self.queue_tree.column(c, width=widths[c], anchor="center")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.queue_tree.yview)
        self.queue_tree.configure(yscrollcommand=vsb.set)
        self.queue_tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        vsb.pack(side="right", fill="y")

        self._apply_queue_tenant_tags()

    def _apply_queue_tenant_tags(self):
        """Shade each queue row by its owning tenant so the table is scannable
        at a glance. Re-run whenever the tenant set changes."""
        for tenant in self.tenants:
            self.queue_tree.tag_configure(tenant.tenant_id, background=tenant.color)
        # Completed jobs stay in the table but grayed out, so the full trace
        # remains visible as the run progresses.
        self.queue_tree.tag_configure(
            "completed", background="#f0f0f0", foreground="#a0a0a0")

    def _build_tenant_panel(self):
        frame = ttk.LabelFrame(self.content, text="Tenants")
        frame.pack(fill="x", padx=10, pady=(0, 10))
        self.tenant_frame = frame
        self.tenant_vars = {}
        self._populate_tenant_rows()

    def _populate_tenant_rows(self):
        """(Re)build one stats row per tenant. Called at init and again on each
        run, since the tenant set is regenerated from the experiment config."""
        for child in self.tenant_frame.winfo_children():
            child.destroy()
        self.tenant_vars = {}
        self.tenant_wait_vars = {}
        for tenant in self.tenants:
            row = ttk.Frame(self.tenant_frame)
            row.pack(fill="x", padx=5, pady=2)
            swatch = tk.Canvas(row, width=14, height=14, highlightthickness=0)
            swatch.create_rectangle(0, 0, 14, 14, fill=tenant.color, outline="")
            swatch.pack(side="left", padx=(0, 5))
            ttk.Label(row, text=tenant.tenant_id, width=10).pack(side="left")
            var = tk.StringVar(value="queued: 0   waiting: 0   running: 0   completed: 0")
            ttk.Label(row, textvariable=var).pack(side="left")
            # Average per-job wait, split by queue type, over completed jobs.
            wvar = tk.StringVar(value="avg wait — global: 0.00s   server: 0.00s")
            ttk.Label(row, textvariable=wvar, foreground="#555555").pack(side="left", padx=(15, 0))
            self.tenant_vars[tenant.tenant_id] = var
            self.tenant_wait_vars[tenant.tenant_id] = wvar

    def _build_hw_panel(self):
        frame = ttk.LabelFrame(self.right_col, text="Hardware")
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
        self.tenants_spinbox.config(state=state)
        for sb in self.count_spinboxes:
            sb.config(state=state)
        self.seed_spinbox.config(state=state)
        self.iter_mult_spinbox.config(state=state)
        self.arrival_spinbox.config(state=state)
        self.threshold_spinbox.config(state=state)
        self.margin_spinbox.config(state=state)
        # readonly combobox re-enables to "readonly", not plain NORMAL
        self.scheduler_combo.config(state="readonly" if enabled else tk.DISABLED)
        self.reload_btn.config(state=state)
        self.start_btn.config(state=state)
        self.stop_btn.config(state=tk.NORMAL if not enabled else tk.DISABLED)

    def _read_experiment_config(self):
        """Validate and return the experiment config dict from the widgets, or
        None (after setting a status message) if invalid."""
        try:
            capacity = int(self.gpu_count_var.get())
            num_tenants = int(self.num_tenants_var.get())
            counts = {name: int(var.get()) for name, var in self.count_vars.items()}
            seed = int(self.seed_var.get())
            iter_multiplier = int(self.iter_mult_var.get())
            arrival_rate = float(self.arrival_rate_var.get())
            threshold_pct = float(self.threshold_var.get())
            safety_margin_mb = int(self.margin_var.get())
        except (tk.TclError, ValueError):
            self.status_var.set("Enter valid numbers for partitions, tenants, counts, "
                                "seed, iterations x, arrival rate, threshold, and margin")
            return None
        if capacity < 1:
            self.status_var.set("Partition count must be at least 1")
            return None
        if num_tenants < 1:
            self.status_var.set("Number of tenants must be at least 1")
            return None
        if sum(counts.values()) < 1:
            self.status_var.set("Set at least one job per tenant (light/medium/heavy)")
            return None
        if iter_multiplier < 1:
            self.status_var.set("Iterations x must be at least 1")
            return None
        if arrival_rate <= 0:
            self.status_var.set("Arrival rate must be greater than 0")
            return None
        if not 0 < threshold_pct <= 100:
            self.status_var.set("Admission threshold must be in (0, 100] %")
            return None
        if safety_margin_mb < 0:
            self.status_var.set("Safety margin must be 0 MB or more")
            return None
        return {
            "capacity": capacity,
            "num_tenants": num_tenants,
            "counts": counts,
            "seed": seed,
            "iter_multiplier": iter_multiplier,
            "arrival_rate": arrival_rate,
            "threshold_pct": threshold_pct,
            "safety_margin_mb": safety_margin_mb,
        }

    def _apply_tenant_config(self, num_tenants, arrival_rate=DEFAULT_ARRIVAL_RATE):
        """Rebuild the tenant set (and the widgets keyed by it) for a new run."""
        self.tenants = make_tenants(num_tenants, arrival_rate)
        self.tenant_colors = {t.tenant_id: t.color for t in self.tenants}
        self._populate_tenant_rows()
        self._apply_queue_tenant_tags()

    def _start_simulation(self):
        config = self._read_experiment_config()
        if config is None:
            return
        capacity, seed = config["capacity"], config["seed"]

        name = self.scheduler_var.get()
        if name not in self.schedulers:
            self.status_var.set("No scheduler selected -- add one to schedulers/ and click 'Reload schedulers'")
            return
        scheduler = self.schedulers[name]()

        # Rebuild tenants for this run, then pre-generate the finite job trace.
        # Same (num_tenants, counts, arrival_rate, seed) => identical trace
        # every run.
        self._apply_tenant_config(config["num_tenants"], config["arrival_rate"])
        jobs = generate_trace(self.tenants, config["counts"], seed,
                              arrival_rate=config["arrival_rate"],
                              iter_multiplier=config["iter_multiplier"])

        self._user_stopped = False
        # Drop the previous run's artifacts so a run that finishes without any
        # completed jobs can't re-show stale numbers. Label this run for the
        # default CSV filename (sanitized: no spaces/parens).
        self.last_results = []
        self.last_summary = None
        self.last_admission_log = []
        self.last_throttle_log = []
        self.last_telemetry = []
        self.last_capacity = capacity
        safe_name = "".join(c if c.isalnum() else "_" for c in name).strip("_")
        self._last_run_label = f"{safe_name}_seed{seed}"
        self._set_controls_enabled(False)
        self.status_var.set(
            f"Running {len(jobs)} jobs on {capacity} GPU(s) -- {name} (seed {seed})"
        )
        # Reset the queue table so a previous run's rows don't linger.
        self.queue_tree.delete(*self.queue_tree.get_children())
        # Reset the flow panel: clear any in-flight assignment animations and
        # forget last run's placements so this run's first assignments animate.
        self.flow_canvas.delete("anim")
        self._flow_prev = {}

        self.event_q = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = threading.Thread(
            target=simulation_thread_main,
            args=(self.event_q, self.stop_flag, capacity, self.tenants, jobs, scheduler),
            kwargs={"threshold_pct": config["threshold_pct"],
                    "safety_margin_mb": config["safety_margin_mb"]},
            daemon=True,
        )
        self.worker.start()
        self.after(100, self._poll_sim_queue)

    def _stop_simulation(self):
        self._user_stopped = True
        if self.stop_flag:
            self.stop_flag.set()
        self.status_var.set("Stopping...")

    def _finish_run(self, snapshot=None):
        """Re-enable the controls once a run ends -- whether it finished its
        whole trace or the user stopped it early."""
        self._set_controls_enabled(True)
        if self._user_stopped:
            self.status_var.set("Stopped. -- Adjust the config and click 'Start'")
        elif snapshot is not None:
            done = snapshot.get("completed", 0)
            self.status_var.set(f"Run complete -- {done} job(s) finished. Adjust the config and click 'Start'")
        else:
            self.status_var.set("Run complete. -- Adjust the config and click 'Start'")
        # A finite experiment that produced results ends by showing its summary
        # (per-tenant mean waits, makespan, GPU utilization) with a CSV export.
        if self.last_summary and self.last_summary.get("jobs"):
            self._show_results_window()

    # ---- end-of-run results view + CSV export ----
    def _show_results_window(self):
        """Pop up the results summary for the just-finished run: headline
        makespan/utilization, a per-tenant mean-wait table, and CSV export."""
        s = self.last_summary
        if not s:
            return

        win = tk.Toplevel(self)
        win.title(f"Run results — {self._last_run_label}")
        win.geometry("620x420")

        head = ttk.Frame(win)
        head.pack(fill="x", padx=12, pady=(12, 6))
        ttk.Label(
            head,
            text=(f"Jobs: {s['jobs']}      Makespan: {s['makespan_s']:.2f}s      "
                  f"GPU utilization: {s['gpu_util'] * 100:.1f}%      "
                  f"Total compute: {s['total_run_s']:.2f}s"),
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w")
        # Second headline row: the remaining §3.9 metrics. The measured NVML
        # figures are None on CPU-only machines, so they render as "n/a".
        nvml_util = s.get("gpu_util_nvml_mean")
        vram_util = s.get("vram_util_mean")
        ttk.Label(
            head,
            text=(f"Throughput: {s.get('throughput_jps', 0.0):.2f} jobs/s      "
                  f"OOM rate: {s.get('oom_rate', 0.0) * 100:.1f}% "
                  f"({s.get('oom_count', 0)} OOM, {s.get('rejected', 0)} rejected)      "
                  f"Fairness (wait/service): {s.get('fairness_wait', 1.0):.3f} / "
                  f"{s.get('fairness_service', 1.0):.3f}      "
                  f"Measured util: "
                  f"{'n/a' if nvml_util is None else f'{nvml_util:.1f}%'}      "
                  f"Avg VRAM: "
                  f"{'n/a' if vram_util is None else f'{vram_util * 100:.1f}%'}"),
            foreground="#555555",
        ).pack(anchor="w", pady=(2, 0))

        table = ttk.LabelFrame(win, text="Per-tenant averages")
        table.pack(fill="both", expand=True, padx=12, pady=6)
        cols = ("tenant", "jobs", "gwait", "swait")
        headings = {"tenant": "Tenant", "jobs": "Jobs",
                    "gwait": "Avg global wait", "swait": "Avg server wait"}
        widths = {"tenant": 140, "jobs": 70, "gwait": 140, "swait": 140}
        tree = ttk.Treeview(table, columns=cols, show="headings", height=10)
        for c in cols:
            tree.heading(c, text=headings[c])
            tree.column(c, width=widths[c], anchor="center")
        for tenant, ts in sorted(s["per_tenant"].items()):
            tree.tag_configure(tenant, background=self.tenant_colors.get(tenant, "#e0e0e0"))
            tree.insert("", "end", values=(
                tenant, ts["jobs"],
                f"{ts['avg_global_wait']:.2f}s", f"{ts['avg_server_wait']:.2f}s",
            ), tags=(tenant,))
        tree.pack(fill="both", expand=True, padx=6, pady=6)

        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(btns, text="Export to CSV…", command=self._export_csv).pack(side="left")
        ttk.Button(btns, text="Close", command=win.destroy).pack(side="right")

    def _export_csv(self):
        """Export the last run's full CSV artifact set (jobs, admission log,
        throttle log, telemetry, summary) into a run directory under a
        user-chosen location. These are the artifacts compared across seeded
        runs / schedulers."""
        if not self.last_results:
            messagebox.showinfo("Nothing to export", "No completed jobs in the last run.")
            return
        parent = filedialog.askdirectory(title="Choose where to save the run's CSVs")
        if not parent:
            return
        run_dir = os.path.join(parent, f"results_{self._last_run_label}")
        try:
            paths = export_run(
                run_dir, self.last_results, self.last_capacity,
                admission_log=self.last_admission_log,
                throttle_log=self.last_throttle_log,
                telemetry=self.last_telemetry,
                summary=self.last_summary,
            )
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        names = ", ".join(os.path.basename(p) for p in paths.values())
        messagebox.showinfo("Export complete",
                            f"Wrote {len(paths)} file(s) ({names}) to\n{run_dir}")

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
        else:
            # Worker has exited: the run is over. The engine's final done=True
            # snapshot may have landed after our drain above, so sweep once more
            # and render it before re-enabling the controls.
            try:
                while True:
                    snapshot = self.event_q.get_nowait()
            except queue.Empty:
                pass
            if snapshot:
                self._render_snapshot(snapshot)
            self._finish_run(snapshot)

    def _render_snapshot(self, snap):
        self.sim_time_var.set(f"t = {snap['sim_time']:.2f}")
        # The engine's final snapshot carries the results log, the admission/
        # throttle logs, the telemetry timeline, and the folded summary; stash
        # them so the run can be exported/summarized once the worker exits.
        if snap.get("done"):
            self.last_results = snap.get("results", [])
            self.last_summary = snap.get("summary")
            self.last_admission_log = snap.get("admission_log", [])
            self.last_throttle_log = snap.get("throttle_log", [])
            self.last_telemetry = snap.get("telemetry", [])
            self.last_capacity = snap.get("capacity", self.last_capacity)
        # Map each busy GPU to the one-letter tag of the workload running on it,
        # so the grid box shows both who owns the GPU (color) and what it's
        # running (letter).
        gpu_types = {}
        for j in snap.get("jobs", []):
            if j.get("gpu") is not None and j.get("state") == "running":
                wt = WORKLOAD_TYPES.get(j.get("wtype"))
                gpu_types[j["gpu"]] = wt.label if wt else "?"
        self._draw_grid(snap["capacity"], snap["gpu_owner"], gpu_types,
                        snap.get("server_queue"), snap.get("partitions"))
        self._draw_flow(snap)
        self._render_admission(snap.get("admission"))
        for tenant_id, s in snap["tenant_stats"].items():
            if tenant_id in self.tenant_vars:
                self.tenant_vars[tenant_id].set(
                    f"queued: {s['queued']}   waiting: {s['waiting']}   "
                    f"running: {s['running']}   completed: {s['completed']}"
                )
                self.tenant_wait_vars[tenant_id].set(
                    f"avg wait — global: {s['avg_global_wait']:.2f}s   "
                    f"server: {s['avg_server_wait']:.2f}s"
                )
        self._render_queue(snap.get("jobs", []))

    def _render_queue(self, jobs):
        """Rebuild the job-queue table from the latest snapshot. Active jobs
        (queued + running) keep their tenant shading; completed jobs stay in
        the table grayed out rather than disappearing."""
        tree = self.queue_tree
        tree.delete(*tree.get_children())
        for j in sorted(jobs, key=lambda r: r["job_id"]):
            gpu = j["gpu"]
            done = j["state"] == "completed"
            tree.insert(
                "", "end",
                values=(j["job_id"], j["tenant"], (j.get("wtype") or "—").capitalize(),
                        f"{j['size']}²", j["iters"], j["state"],
                        "—" if gpu is None else gpu,
                        f"{j.get('global_wait', 0.0):.2f}s",
                        f"{j.get('server_wait', 0.0):.2f}s"),
                tags=("completed",) if done else (j["tenant"],),
            )

    def _render_admission(self, adm):
        """Update the admission-control panel from a snapshot's `admission`
        block: the outcome counters line, plus a merged, time-ordered feed of
        recent decisions and throttle raise/release events."""
        if not adm:
            return
        c = adm.get("counters", {})
        throttled = adm.get("throttled_partitions", [])
        self.admission_counters_var.set(
            f"decisions: {adm.get('decisions', 0)}   admit: {c.get('admit', 0)}   "
            f"queue: {c.get('queue', 0)}   reject: {c.get('reject', 0)}   "
            f"throttle: {c.get('throttle_running', 0)}   "
            f"throttled now: {', '.join(f'P{i}' for i in throttled) or '—'}")

        events = []
        for r in adm.get("recent", []):
            events.append((r["timestamp"], (
                f"[{r['timestamp']:7.2f}s] job {r['job_id']:>3} "
                f"({r['tenant']}, {r['mem_mb']} MB) -> P{r['partition']}: "
                f"{r['outcome'].upper()}  "
                f"({r['projected_mb']}+{r['margin_mb']} vs {r['limit_mb']:.0f} MB)")))
        for r in adm.get("recent_throttle", []):
            who = f" (job {r['job_id']})" if r.get("job_id") is not None else ""
            events.append((r["timestamp"],
                           f"[{r['timestamp']:7.2f}s] P{r['partition']} "
                           f"throttle {r['event']}{who}"))
        events.sort(key=lambda e: e[0])
        self.admission_list.delete(0, tk.END)
        for _, line in events[-12:]:
            self.admission_list.insert(tk.END, line)
        self.admission_list.see(tk.END)

    def _draw_grid(self, capacity, gpu_owner, gpu_types=None, server_queue=None,
                   partitions=None):
        gpu_types = gpu_types or {}
        server_queue = server_queue or []
        partitions = partitions or []
        self.grid_canvas.delete("all")
        pad = 6
        # Extra vertical room under each box for the partition-usage bar and
        # its used/cap MB caption.
        foot = 18
        pitch = self.BOX + pad + foot
        for gpu_id in range(capacity):
            col = gpu_id % self.GRID_COLS
            row = gpu_id // self.GRID_COLS
            x0 = pad + col * (self.BOX + pad)
            y0 = pad + row * pitch
            x1, y1 = x0 + self.BOX, y0 + self.BOX
            tenant_id = gpu_owner.get(gpu_id)
            color = self.tenant_colors.get(tenant_id, "#e0e0e0")  # gray = idle
            fg = "white" if tenant_id else "black"
            part = partitions[gpu_id] if gpu_id < len(partitions) else None
            # Orange border marks a partition running cooperatively throttled.
            outline, width = (("#ff8f00", 2) if part and part.get("throttled")
                              else ("#999999", 1))
            self.grid_canvas.create_rectangle(x0, y0, x1, y1, fill=color,
                                              outline=outline, width=width)
            # GPU id near the top, workload-type letter (L/M/H) below it.
            self.grid_canvas.create_text((x0 + x1) / 2, y0 + self.BOX * 0.33,
                                         text=str(gpu_id), fill=fg)
            letter = gpu_types.get(gpu_id)
            if letter:
                self.grid_canvas.create_text((x0 + x1) / 2, y0 + self.BOX * 0.68,
                                             text=letter, fill=fg,
                                             font=("TkDefaultFont", 12, "bold"))
            # Jobs waiting in this server's queue behind the running one, shown
            # as a "+N" badge in the corner so per-server backlog is visible.
            load = server_queue[gpu_id] if gpu_id < len(server_queue) else 0
            waiting = load - 1 if load > 0 else 0
            if waiting > 0:
                self.grid_canvas.create_text(x1 - 8, y0 + 8, text=f"+{waiting}",
                                             fill=fg, font=("TkDefaultFont", 8, "bold"))
            # Partition VRAM usage bar (ledger used / cap MB): green while
            # under the admission threshold, red once past it.
            if part:
                cap = part.get("cap_mb") or 0
                used = part.get("used_mb", 0)
                frac = min(used / cap, 1.0) if cap else 0.0
                try:
                    limit_frac = float(self.threshold_var.get()) / 100.0
                except (tk.TclError, ValueError):
                    limit_frac = 1.0
                by0, by1 = y1 + 3, y1 + 8
                self.grid_canvas.create_rectangle(x0, by0, x1, by1,
                                                  fill="#eeeeee", outline="#bbbbbb")
                if frac > 0:
                    fill = "#e53935" if frac > limit_frac else "#43a047"
                    self.grid_canvas.create_rectangle(
                        x0, by0, x0 + self.BOX * frac, by1, fill=fill, outline="")
                self.grid_canvas.create_text((x0 + x1) / 2, by1 + 5,
                                             text=f"{used}/{cap}",
                                             font=("TkDefaultFont", 7),
                                             fill="#555555")
        rows_needed = (capacity + self.GRID_COLS - 1) // self.GRID_COLS
        self.grid_canvas.configure(height=max(80, rows_needed * pitch + pad))


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