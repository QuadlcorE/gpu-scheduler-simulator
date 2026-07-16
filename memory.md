# Lightweight GPU scheduler for Simulated Multi-Tenant Environment

## Project Overview
[The project is a GPU scheduler that basically simulates multiple gpu's allowing a user to write and test scheduling algorithms. It is lightweight in the sense that the multitenancy is simulated and it is run locally.] 

## Tech stack 
- Python 
- Pytorch
-Tkinter

## Writing a scheduler
Drop a `.py` file in `schedulers/`: subclass `Scheduler`, set a `NAME`, and
implement `select_server(job, servers) -> int | None`. Use "Reload schedulers"
in the dashboard to pick it up without restarting. Full contract and the
experiment-config / results details are in `architecture.md`.

## Running experiments
A run = (number of tenants, per-tenant light/medium/heavy counts, seed). Same
config + seed => identical job trace, so schedulers are comparable. On
completion the dashboard shows a results summary and can export per-job rows to
CSV. `python test_replicability.py` demonstrates the reproducibility and
comparison workflow.

## Ignored files 
- Boilerplate.py
- multitenant.py
