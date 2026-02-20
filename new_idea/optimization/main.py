#!/usr/bin/env python3
"""
Optimal scheduler MILP using Gurobi.

Usage:
    python opt_scheduler_gurobi.py trace.jsonl

Input JSONL: each line is a JSON object with keys:
    - "timestamp": float (arrival / release time)
    - "model_name": string (which model the request targets)
    - "generation_time": float (processing time)
"""

import sys
import json
import gurobipy as gp
from gurobipy import GRB

# ----------------------------
# User-tweakable parameters
# ----------------------------

DEFAULT_WAKE = {
    "/local/huzaifa/workspace/vLLM/vllm-serverless-optimization/models/llama3-3b": 1.065,
    "/local/huzaifa/workspace/vLLM/vllm-serverless-optimization/models/qwen-1.8b": 0.73,
    "/local/huzaifa/workspace/vLLM/vllm-serverless-optimization/models/qwen2.5-3b": 1.035,
}
DEFAULT_SLEEP = {
    "/local/huzaifa/workspace/vLLM/vllm-serverless-optimization/models/llama3-3b": 0.056,
    "/local/huzaifa/workspace/vLLM/vllm-serverless-optimization/models/qwen-1.8b": 0.048,
    "/local/huzaifa/workspace/vLLM/vllm-serverless-optimization/models/qwen2.5-3b": 0.054,
}

# ----------------------------
# Helper: parse trace
# ----------------------------
def read_trace(jsonl_path):
    records = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            t = float(obj['timestamp'])
            model = str(obj['model_name'])
            p = float(obj['generation_time'])
            
            records.append({
                'id': i,
                'timestamp': t,
                'model_name': model,
                'proc_time': p,
                'raw': obj
            })
    return records

# ----------------------------
# Build u matrix
# ----------------------------
def build_u_matrix(model_names, default_wake, default_sleep):
    # model_names: list of model name strings
    # Return dict u[(a,b)] where a in {"idle"} U model_names, b in model_names
    u = {}
    # get wake/sleep times (fallback to reasonable defaults)
    wake = {m: default_wake.get(m, 5.0) for m in model_names}
    sleep = {m: default_sleep.get(m, 1.0) for m in model_names}
    states = ['idle'] + model_names
    for a in states:
        for b in model_names:
            if a == 'idle':
                u[(a,b)] = wake[b]  # from idle, just wake b
            elif a == b:
                u[(a,b)] = 0.0
            else:
                # switching: sleep a then wake b (you can change logic if needed)
                u[(a,b)] = sleep[a] + wake[b]
    return u

# ----------------------------
# Main MILP builder & solve
# ----------------------------
def solve_schedule(records, wake_map, sleep_map, time_limit=None):
    # records: list of dict with keys id,timestamp,model_name,proc_time
    n = len(records)
    if n == 0:
        print("No requests in trace.")
        return None

    # map models to indices
    model_names = sorted(set(r['model_name'] for r in records))
    M = len(model_names)
    model_to_idx = {m: idx for idx, m in enumerate(model_names)}

    # stage count (upper bound)
    S = n  # worst-case one stage per request

    # processing times, release times, model indices
    r = [r_['timestamp'] for r_ in records]
    p = [r_['proc_time'] for r_ in records]
    mj = [model_to_idx[r_['model_name']] for r_ in records]

    # build u matrix
    u = build_u_matrix(model_names, wake_map, sleep_map)

    # compute bigM: safe upper bound on times
    max_r = max(r)
    sum_p = sum(p)
    sum_u = sum(u.values())
    BIGM = max_r + sum_p + sum_u + 1000.0

    # Create Gurobi model
    model = gp.Model("llm_scheduler")
    model.setParam('OutputFlag', 1)  # set to 0 to suppress Gurobi output
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)

    # Decision variables
    # z[j,s] binary: job j assigned to stage s
    z = {}
    for j in range(n):
        for s in range(S):
            z[j, s] = model.addVar(vtype=GRB.BINARY, name=f"z_{j}_{s}")

    # y[s,m] binary: stage s uses model m
    y = {}
    for s in range(S):
        for m in range(M):
            y[s, m] = model.addVar(vtype=GRB.BINARY, name=f"y_{s}_{m}")

    # continuous stage start and completion times
    Svar = {}
    Cvar = {}
    for s in range(S):
        Svar[s] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"S_{s}")
        Cvar[s] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"C_{s}")

    Cmax = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")

    model.update()

    # Objective: minimize Cmax
    model.setObjective(Cmax, GRB.MINIMIZE)

    # Constraints

    # 1) Each job assigned to exactly one stage
    for j in range(n):
        model.addConstr(gp.quicksum(z[j, s] for s in range(S)) == 1, name=f"assign_job_{j}")

    # 2) If job assigned to stage s then that stage must use that job's model
    for j in range(n):
        m_idx = mj[j]
        for s in range(S):
            model.addConstr(z[j, s] <= y[s, m_idx], name=f"job_model_consistency_j{j}_s{s}")

    # 3) At most one model active per stage
    for s in range(S):
        model.addConstr(gp.quicksum(y[s, m] for m in range(M)) <= 1, name=f"one_model_stage_{s}")
        
    # CORRECTION?: Prevent empty active stages: if a stage is active, it must have at least one job
    for s in range(S):
        model.addConstr(gp.quicksum(z[j, s] for j in range(n)) >= gp.quicksum(y[s, m] for m in range(M)),
                        name=f"no_empty_stage_{s}")

    # 4) If a stage has any job assigned, the respective y must be 1
    # Equivalent already enforced by z <= y for job's model; but to be explicit:
    # y[s,m] >= max_j { z[j,s] for jobs j with model m }
    for j in range(n):
        m_idx = mj[j]
        for s in range(S):
            model.addConstr(y[s, m_idx] >= z[j, s], name=f"y_ge_z_j{j}_s{s}")

    # 5) Release times: stage start must be >= arrival time for any job assigned there
    for j in range(n):
        for s in range(S):
            model.addConstr(Svar[s] >= r[j] - BIGM * (1 - z[j, s]), name=f"release_j{j}_s{s}")

    # 6) Stage completion must be >= start + p_j for any job assigned there (linearizing max)
    for j in range(n):
        for s in range(S):
            model.addConstr(Cvar[s] >= Svar[s] + p[j] * z[j, s], name=f"stage_completion_j{j}_s{s}")

    # 7) Sequencing & setup times between consecutive stages
    # For s == 0 (first stage), use idle->b wake times
    for b_idx, b_name in enumerate(model_names):
        model.addConstr(Svar[0] >= u[('idle', b_name)] - BIGM * (1 - y[0, b_idx]), name=f"initial_wake_{b_name}")

    # For s >= 1, for all a in models (previous) and b in models (current):
    # S_s >= C_{s-1} + u[a,b]  if y[s-1,a]==1 and y[s,b]==1
    for s in range(1, S):
        for a_idx, a_name in enumerate(model_names):
            for b_idx, b_name in enumerate(model_names):
                model.addConstr(
                    Svar[s] >= Cvar[s-1] + u[(a_name, b_name)] - BIGM * (2 - y[s-1, a_idx] - y[s, b_idx]),
                    name=f"switch_s{s}_a{a_name}_b{b_name}"
                )
        # Also handle case when previous stage was 'idle' theoretically, though we enforce contiguous usage
        # (No explicit y for idle beyond stage 0 handling.)

    # 8) Makespan constraints
    for s in range(S):
        model.addConstr(Cmax >= Cvar[s], name=f"Cmax_ge_Cs_{s}")

    # 9) Avoid unused trailing stages -> force stages to be contiguous
    # sum_m y_s_m >= sum_m y_{s+1,m}
    for s in range(S - 1):
        model.addConstr(gp.quicksum(y[s, m] for m in range(M)) >= gp.quicksum(y[s + 1, m] for m in range(M)),
                        name=f"contiguous_stages_{s}")

    # Good practice: set variable types / integrality done above
    model.update()

    # Heuristics: optional - set a time limit or MIPFocus etc. The user can tweak model.params
    # Solve
    model.optimize()

    # Check solve status
    status = model.Status
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        print("Gurobi did not find a feasible/optimal solution. Status:", status)
        return None

    # Extract solution
    solution = {
        'model_names': model_names,
        'stages': []
    }

    # We'll compute per-stage assigned jobs and times
    for s in range(S):
        used = False
        active_model = None
        for m in range(M):
            val = y[s, m].X
            if val > 0.5:
                used = True
                active_model = model_names[m]
                break
        if not used:
            continue
        start_t = Svar[s].X
        end_t = Cvar[s].X
        assigned_jobs = []
        for j in range(n):
            if z[j, s].X > 0.5:
                assigned_jobs.append(j)
        stage_info = {
            'stage_index': s,
            'model': active_model,
            'start': float(start_t),
            'end': float(end_t),
            'jobs': assigned_jobs
        }
        solution['stages'].append(stage_info)

    # Job-wise completion times
    job_reports = []
    for stage in solution['stages']:
        s_idx = stage['stage_index']
        s_start = stage['start']
        for j in stage['jobs']:
            comp = s_start + p[j]
            job_reports.append({
                'job_id': records[j]['id'],
                'model': records[j]['model_name'],
                'release': records[j]['timestamp'],
                'proc_time': p[j],
                'sent_time': s_start,
                'completion_time': comp,
                'raw': records[j]['raw']
            })

    # Sort job_reports by job id or completion time for readability
    job_reports.sort(key=lambda x: x['job_id'])
    solution['jobs'] = job_reports
    solution['C_max'] = Cmax.X

    return solution

# ----------------------------
# CLI & run
# ----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python opt_scheduler_gurobi.py trace.jsonl")
        sys.exit(1)

    trace_path = sys.argv[1]
    records = read_trace(trace_path)
    if not records:
        print("No records found.")
        sys.exit(0)

    # Build wake/sleep maps: use defaults and fill missing with defaults
    model_names = sorted(set(r['model_name'] for r in records))
    wake_map = {m: DEFAULT_WAKE.get(m, DEFAULT_WAKE.get('default', 5.0)) for m in model_names}
    sleep_map = {m: DEFAULT_SLEEP.get(m, DEFAULT_SLEEP.get('default', 1.0)) for m in model_names}

    solution = solve_schedule(records, wake_map, sleep_map, time_limit=150)

    if solution is None:
        print("No solution found.")
        sys.exit(2)

    # Print readable schedule
    print("\n=== Optimal schedule ===")
    print(f"Models: {solution['model_names']}")
    print(f"Makespan (C_max): {solution['C_max']:.4f}\n")
    for s in solution['stages']:
        print(f"Stage {s['stage_index']}: model={s['model']}, start={s['start']:.4f}, end={s['end']:.4f}")
        print("  jobs:", s['jobs'])

    print("\nJob-level report (job_id, model, release, proc_time, sent_time, completion_time):")
    for j in solution['jobs']:
        print(f"  {j['job_id']}, {j['model']}, r={j['release']:.4f}, p={j['proc_time']:.4f}, sent={j['sent_time']:.4f}, done={j['completion_time']:.4f}")

    # Write JSON output
    out_path = "schedule_output.json"
    with open(out_path, 'w') as f:
        json.dump(solution, f, indent=2)
    print(f"\nWrote schedule to {out_path}")

if __name__ == "__main__":
    main()
