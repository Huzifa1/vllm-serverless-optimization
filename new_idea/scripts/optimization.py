# Python scheduling solver visible to the user.
# It implements an exact branch-and-bound scheduler (DFS + memo) to minimize SLO violations
# for the single-GPU, one-active-model-at-a-time setting you described.
#
# It also provides several heuristics and a helper that finds the maximum uniform SLO (cap)
# for which zero violations are achievable (binary search over SLO) using the exact solver.
#
# Notes & assumptions (implemented):
# - Models: each model i has wake_time[i], sleep_time[i], gen_time_per_token[i].
# - Switch cost from model a to b (a != b and a != -1) = sleep_time[a] + wake_time[b].
# - If active model is -1 (none yet), switching cost to wake target is wake_time[target].
# - Active model remains awake across idle periods unless switched away when serving a different-model request.
# - Arrival times are release times; a job cannot start before its arrival.
# - Completion time = start_time + generation_time (start_time includes any switching overhead)
# - A request violates SLO if (completion_time - arrival_time) > SLO_for_that_request.
#
# Limitations:
# - The exact solver is exponential (branch-and-bound). Works fine for small N (N <= ~12-14).
# - For larger N use heuristics included below.
#
# The script exposes functions:
# - load_trace_jsonl(path) -> jobs list
# - solve_exact(jobs, models, timeout=None) -> optimal schedule (may be slow for big N)
# - greedy_heuristics(...) -> several heuristic schedules for comparison
# - find_max_uniform_slo_cap(...) -> binary search to find largest uniform SLO (single value applied to all jobs)
#
# We'll run a small synthetic example at the end to demonstrate.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import json, math, time, itertools

@dataclass
class Job:
    id: int
    arrival: float
    model: int
    input_tokens: int
    slo: float  # per-job SLO (seconds)

@dataclass
class ModelParams:
    name: str
    wake: float
    sleep: float
    gen_per_token: float  # seconds per token

def parse_jsonl_string(s: str, model_name_to_index: Dict[str,int]) -> List[Job]:
    jobs = []
    for i,line in enumerate(s.strip().splitlines()):
        j = json.loads(line)
        jobs.append(Job(id=i,
                        arrival=float(j["timestamp"]),
                        model=model_name_to_index[j["model_name"]],
                        input_tokens=int(j["input_length"]),
                        slo=float(j.get("slo", j.get("SLO", 1.0)))))
    return jobs

def job_processing_time(job: Job, models: List[ModelParams]) -> float:
    return job.input_tokens * models[job.model].gen_per_token

def switch_cost(from_model: int, to_model: int, models: List[ModelParams]) -> float:
    if from_model == to_model:
        return 0.0
    if from_model == -1:
        return models[to_model].wake
    return models[from_model].sleep + models[to_model].wake

# Exact solver (DFS + memo + pruning)
def solve_exact(jobs: List[Job], models: List[ModelParams], time_limit: Optional[float]=10.0):
    """
    Return dict with keys:
      'min_violations', 'best_order' (list of job ids), 'best_schedule' (list of (jobid, start, end)),
      'stats' (nodes explored, elapsed)
    """
    N = len(jobs)
    jobs_by_id = {job.id: job for job in jobs}
    all_mask = (1<<N) - 1
    start_time_clock = time.time()
    best = {"viol": N+1, "order": None, "schedule": None}
    nodes = 0
    memo = {}  # key -> minimal violations found from that state

    # Pre-sort job ids to have deterministic order
    job_ids = [job.id for job in sorted(jobs, key=lambda x:(x.arrival, x.id))]

    def dfs(served_mask: int, active_model: int, current_time: float, violations: int, order: List[int], schedule: List[Tuple[int,float,float]]):
        nonlocal nodes, best, start_time_clock
        nodes += 1
        # time limit cutoff
        if time_limit is not None and (time.time() - start_time_clock) > time_limit:
            return False  # indicate time exhausted upward
        # pruning: if already worse than best
        if violations >= best["viol"]:
            return True
        if served_mask == all_mask:
            # complete
            best["viol"] = violations
            best["order"] = order.copy()
            best["schedule"] = schedule.copy()
            return True
        # memoization key: served_mask, active_model, rounded current_time to micro-second (string)
        key = (served_mask, active_model, round(current_time,6))
        if key in memo and memo[key] <= violations:
            return True
        memo[key] = violations

        # Determine available jobs (arrival <= current_time) - if none, we must advance time to next arrival
        available = [jid for jid in job_ids if not (served_mask & (1<<jid)) and jobs_by_id[jid].arrival <= current_time]
        if not available:
            # advance to earliest arrival among unserved
            next_arrival = min(jobs_by_id[jid].arrival for jid in job_ids if not (served_mask & (1<<jid)))
            # we can optionally stay with same active model (no cost)
            # Advance time and continue
            return dfs(served_mask, active_model, next_arrival, violations, order, schedule)

        # Branch on each available job - try promising order: prefer same-model first to reduce switches
        # sort available by heuristic: same-model first, then earliest deadline slack (arrival+SLO)
        def heuristic_key(jid):
            j = jobs_by_id[jid]
            same = 0 if j.model == active_model else 1
            slack = (j.arrival + j.slo) - current_time
            return (same, slack, j.arrival)
        for jid in sorted(available, key=heuristic_key):
            job = jobs_by_id[jid]
            scost = switch_cost(active_model, job.model, models)
            # when switching, we perform the switch before starting generation; switching does not require waiting for anything else
            # start candidate time is max(current_time + scost, job.arrival)
            start_t = max(current_time + scost, job.arrival)
            proc = job_processing_time(job, models)
            end_t = start_t + proc
            viol = 1 if (end_t - job.arrival) > job.slo + 1e-12 else 0
            order.append(jid)
            schedule.append((jid, start_t, end_t))
            cont = dfs(served_mask | (1<<jid), job.model, end_t, violations + viol, order, schedule)
            order.pop(); schedule.pop()
            if not cont:
                return False
        return True

    # start DFS with active_model = -1 (none), current_time = earliest arrival (or 0)
    initial_time = min(job.arrival for job in jobs) if jobs else 0.0
    dfs(0, -1, initial_time, 0, [], [])
    elapsed = time.time() - start_time_clock
    return {"min_violations": best["viol"], "best_order": best["order"], "best_schedule": best["schedule"], "stats": {"nodes": nodes, "elapsed": elapsed}}

# Heuristic schedulers for comparison
def schedule_fcfs(jobs: List[Job], models: List[ModelParams]):
    jobs_sorted = sorted(jobs, key=lambda j: j.arrival)
    return simulate_sequence([j.id for j in jobs_sorted], jobs, models)

def schedule_edf(jobs: List[Job], models: List[ModelParams]):
    # At each decision epoch, pick available job with earliest deadline (arrival + slo)
    N = len(jobs)
    job_by_id = {j.id:j for j in jobs}
    served = set()
    current_time = min(j.arrival for j in jobs) if jobs else 0.0
    active = -1
    order = []
    schedule = []
    while len(served) < N:
        available = [j for j in jobs if j.id not in served and j.arrival <= current_time]
        if not available:
            current_time = min(j.arrival for j in jobs if j.id not in served)
            continue
        next_job = min(available, key=lambda j: j.arrival + j.slo)
        scost = switch_cost(active, next_job.model, models)
        start_t = max(current_time + scost, next_job.arrival)
        end_t = start_t + job_processing_time(next_job, models)
        schedule.append((next_job.id, start_t, end_t))
        order.append(next_job.id)
        active = next_job.model
        current_time = end_t
        served.add(next_job.id)
    # compute violations
    viols = sum(1 for (jid,s,e) in schedule if (e - [j for j in jobs if j.id==jid][0].arrival) > [j for j in jobs if j.id==jid][0].slo + 1e-12)
    return {"order": order, "schedule": schedule, "violations": viols}

def simulate_sequence(order: List[int], jobs: List[Job], models: List[ModelParams]):
    job_by_id = {j.id:j for j in jobs}
    current_time = min(j.arrival for j in jobs) if jobs else 0.0
    active = -1
    schedule = []
    for jid in order:
        job = job_by_id[jid]
        if job.arrival > current_time:
            # idle wait, keep active as is
            current_time = job.arrival
        scost = switch_cost(active, job.model, models)
        # apply scost (if active same, scost 0)
        # If scost > 0 and current_time < job.arrival, we can do scost before arrival (waking in advance) -- but in our model we only switch when we are about to serve a job.
        # For simplicity we apply scost just before starting the job (will increase start time if current_time already >= job.arrival)
        # More realistic: if scost can be done during idle time before arrival, we could pre-wake. We do not pre-wake here.
        start_t = max(current_time + scost, job.arrival)
        end_t = start_t + job_processing_time(job, models)
        schedule.append((jid, start_t, end_t))
        active = job.model
        current_time = end_t
    viols = sum(1 for (jid,s,e) in schedule if (e - job_by_id[jid].arrival) > job_by_id[jid].slo + 1e-12)
    return {"order": order, "schedule": schedule, "violations": viols}

# Binary search to find max uniform SLO cap for zero violations
def find_max_uniform_slo_cap(jobs: List[Job], models: List[ModelParams], time_limit_per_solve: float = 5.0, precision: float = 1e-3):
    # We'll apply the same SLO to all jobs and find the largest value for which solve_exact returns 0 violations.
    # Search range:
    min_slo = 0.0
    # upper bound: everything serialized w/ all switching worst-case
    worst = 0.0
    # safe upper bound: last arrival + sum of all generation + sum of max switches
    total_gen = sum(job_processing_time(j,models) for j in jobs)
    max_switch = sum(max(models[i].sleep + models[j].wake for j in range(len(models))) for i in range(len(models))) if models else 0.0
    upper = max(job.arrival for job in jobs) + total_gen + max_switch + 100.0
    lo, hi = min_slo, upper
    best = 0.0
    while hi - lo > precision:
        mid = (lo+hi)/2
        # set all jobs slo = mid temporarily
        jobs_copy = [Job(id=j.id, arrival=j.arrival, model=j.model, input_tokens=j.input_tokens, slo=mid) for j in jobs]
        res = solve_exact(jobs_copy, models, time_limit=time_limit_per_solve)
        if res["min_violations"] == 0:
            best = mid
            lo = mid
        else:
            hi = mid
    return best

# Small demo with synthetic trace
def demo():
    # Define models
    models = [
        ModelParams(name="m1", wake=1.0, sleep=0.5, gen_per_token=0.01),
        ModelParams(name="m2", wake=1.5, sleep=0.6, gen_per_token=0.008),
        ModelParams(name="m3", wake=0.8, sleep=0.4, gen_per_token=0.02)
    ]
    # Synthetic JSONL trace (timestamp, model_name, input_length, slo)
    jsonl = """
{"timestamp": 0.0, "model_name": "m1", "input_length": 50, "slo": 2.0}
{"timestamp": 0.1, "model_name": "m2", "input_length": 30, "slo": 1.5}
{"timestamp": 0.2, "model_name": "m1", "input_length": 10, "slo": 1.0}
{"timestamp": 0.25, "model_name": "m3", "input_length": 20, "slo": 1.2}
{"timestamp": 0.3, "model_name": "m2", "input_length": 40, "slo": 2.0}
"""
    model_map = {m.name:i for i,m in enumerate(models)}
    jobs = parse_jsonl_string(jsonl, model_map)
    print("Jobs:")
    for j in jobs:
        print(vars(j))
    print("\nRunning exact solver (time limit 4s)...")
    exact = solve_exact(jobs, models, time_limit=4.0)
    print("Exact result:", exact["min_violations"], "violations; nodes:", exact["stats"]["nodes"], "elapsed:", exact["stats"]["elapsed"])
    print("Best order:", exact["best_order"])
    print("Schedule: (jobid, start, end)")
    for rec in exact["best_schedule"]:
        print(rec)
    print("\nEDF heuristic:")
    print(schedule_edf(jobs, models))
    print("\nFCFS heuristic:")
    print(schedule_fcfs(jobs, models))
    print("\nFind uniform SLO cap (binary search, per-solve timeout 2s):")
    cap = find_max_uniform_slo_cap(jobs, models, time_limit_per_solve=2.0, precision=1e-2)
    print("Estimated max uniform SLO without violations:", cap)
    return {"jobs": jobs, "models": models, "exact": exact}

# Run demo
_demo_res = demo()
print("\nDemo finished.")
