import pulp
import math
from prepare_data import filter_data

tasks = filter_data("dataset/job_table.parquet")

n = len(tasks)

time_units = tasks["time_units"].tolist()
n_cores = tasks["num_cores_req"].tolist()
mean_power = tasks["mean_power"].tolist()

# Restringim la capacitat del model
max_cores = max(max(n_cores), math.ceil(sum(n_cores)*0.6))
max_power = max(max(mean_power), math.ceil(sum(mean_power)*0.6))

# Heurística de sostre màxim: execució de totes les tasques seqüencialment
H = sum(time_units)

# Definició del model: minimitzar temps d'execució total
model = pulp.LpProblem("Task_Scheduling", pulp.LpMinimize)

# La tasca 'i' comença a l'instant 't': x[i,t] = 1
x = {(i,t): pulp.LpVariable(f"x_{i}_{t}", cat="Binary")
     for i in range(n)
     for t in range(H - time_units[i] + 1)
}

# Instant en què finalitzen totes les tasques
T = pulp.LpVariable("total_time", lowBound=0)

# Cada tasca comença exactament 1 vegada
for i in range(n):
    model += pulp.lpSum(x[i,t] for t in range(H - time_units[i] + 1)) == 1

    # La tasca ha d'acabar abans de T
    model += pulp.lpSum((t + time_units[i]) * x[i,t] for t in range(H - time_units[i] + 1)) <= T


# Imposem restriccions de recursos per a cada bloc temporal
for s in range(H):

    ongoing_tasks = [(i,t) for i in range(n)
                     for t in range(H - time_units[i] + 1)
                     if t <= s < t + time_units[i]]

    model += pulp.lpSum(n_cores[i] * x[i,t]
                        for i, t in ongoing_tasks) <= max_cores

    model += pulp.lpSum(mean_power[i] * x[i,t]
                        for i, t in ongoing_tasks) <= max_power

# Executem la resolució del model plantejat
model.solve(pulp.PULP_CBC_CMD(msg=False))

print("MODEL_STATE:", pulp.LpStatus[model.status])


# Si la solució és òptima, la imprimim
if model.status == pulp.LpStatusOptimal:
    print("MINIMAL_TIME:", pulp.value(T) * 5, "minutes")

    for(i, t), variable in x.items():
        if pulp.value(variable) > 0.5:
            print(f"Task {i}: starts at {t*5}, finishes at {((t+time_units[i])*5)}")