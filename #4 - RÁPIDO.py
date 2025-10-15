import time, math
T0 = time.time()
def dbg(msg): print(msg, flush=True)

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Binary,
    Constraint, ConstraintList, Objective, Expression,
    minimize, SolverFactory, value
)



# ---------------- Config ----------------
ficheiro_excel = '/Users/jorgecastro/Desktop/Tese/Planeamento Automático.xlsx'
Delta = 0.4
b = 2
H_HOURS = 100.0
M_names = ['M4','M7','M9']

# -------- Leitura e dados --------
dbg("📥 A carregar dados do Excel...")

df_excel = pd.read_excel(ficheiro_excel, sheet_name=4, header=None).iloc[1:]
df_ordens = pd.DataFrame({
    'Projeto':     df_excel.iloc[:, 0].tolist(),
    'PN':          df_excel.iloc[:, 1].tolist(),
    'FAMÍLIA':     df_excel.iloc[:, 5].tolist(),
    'ESP':         df_excel.iloc[:, 3].tolist(),
    'DESIGNAÇÃO':  df_excel.iloc[:, 4].tolist(),
    'MATRIZ':      df_excel.iloc[:, 6].tolist(),
    'QUANTIDADE':  df_excel.iloc[:, 7].tolist(),
    'DUE DATE':    df_excel.iloc[:, 9].tolist(),
})
df_ordens["TEMPO PROCESSAMENTO (horas)"] = df_ordens["QUANTIDADE"].astype(float) / 60.0

data_inicio = datetime.today()
def calcular_horas_uteis_com_extra(due_date):
    due = pd.to_datetime(due_date, format='%d/%m', errors='coerce')
    if pd.isna(due): return np.nan
    return pd.bdate_range(start=data_inicio, end=due).size * 8 + 15

df_ordens['DUE DATE'] = df_ordens['DUE DATE'].apply(calcular_horas_uteis_com_extra)

def ler_folha(idx, due_col_idx):
    dfx = pd.read_excel(ficheiro_excel, sheet_name=idx, header=None).iloc[1:]
    dfo = pd.DataFrame({
        'Projeto':     dfx.iloc[:, 0].tolist(),
        'PN':          dfx.iloc[:, 1].tolist(),
        'FAMÍLIA':     dfx.iloc[:, 5].tolist(),
        'ESP':         dfx.iloc[:, 3].tolist(),
        'DESIGNAÇÃO':  dfx.iloc[:, 4].tolist(),
        'MATRIZ':      dfx.iloc[:, 6].tolist(),
        'QUANTIDADE':  dfx.iloc[:, 7].tolist(),
        'DUE DATE':    dfx.iloc[:, due_col_idx].tolist(),
    })
    dfo["TEMPO PROCESSAMENTO (horas)"] = dfo["QUANTIDADE"].astype(float) / 60.0
    dfo['DUE DATE'] = dfo['DUE DATE'].apply(calcular_horas_uteis_com_extra)
    return dfo

df_ordens_primeira_folha = ler_folha(0, 8)
df_ordens_segunda_folha  = ler_folha(1, 8)
df_ordens_terceira_folha = ler_folha(2, 8)

df_ordens_primeira_folha['Fonte']  = df_ordens_primeira_folha.index.to_series().apply(lambda x: f'M4 - {x+1}')
df_ordens_segunda_folha['Fonte']   = df_ordens_segunda_folha.index.to_series().apply(lambda x: f'M7 - {x+1}')
df_ordens_terceira_folha['Fonte']  = df_ordens_terceira_folha.index.to_series().apply(lambda x: f'M9 - {x+1}')

df_planeamento_atual = pd.concat(
    [df_ordens_primeira_folha, df_ordens_segunda_folha, df_ordens_terceira_folha],
    ignore_index=True
)
df_planeamento_atual = df_planeamento_atual[['Fonte'] + [c for c in df_planeamento_atual.columns if c != 'Fonte']]

df_excel_ultima_folha = pd.read_excel(ficheiro_excel, sheet_name=-1, header=None).iloc[1:]
df_Mj_restricoes = pd.DataFrame({
    'DESIGNAÇÃO': df_excel_ultima_folha.iloc[:, 0].tolist(),
    'FAMÍLIA':    df_excel_ultima_folha.iloc[:, 1].tolist(),
    'M4':         df_excel_ultima_folha.iloc[:, 3].tolist(),
    'M7':         df_excel_ultima_folha.iloc[:, 4].tolist(),
    'M9':         df_excel_ultima_folha.iloc[:, 5].tolist(),
})

for maquina in ['M4', 'M7', 'M9']:
    ultima = df_planeamento_atual[df_planeamento_atual['Fonte'].str.startswith(maquina)].iloc[-1]
    df_ordens = pd.concat([df_ordens, ultima.drop(labels=['Fonte']).to_frame().T], ignore_index=True)
df_ordens = pd.concat([df_ordens.tail(3), df_ordens.iloc[:-3]], ignore_index=True)

df_excel_designacoes_casais = pd.read_excel(ficheiro_excel, sheet_name=7, header=None).iloc[1:]
df_designacoes_casais = pd.DataFrame({
    'DESIGNAÇÃO 1': df_excel_designacoes_casais.iloc[:, 0].tolist(),
    'FAMÍLIA':      df_excel_designacoes_casais.iloc[:, 1].tolist(),
    'DESIGNAÇÃO 2': df_excel_designacoes_casais.iloc[:, 2].tolist(),
})
casais_set = set(tuple(sorted((r['DESIGNAÇÃO 1'], r['DESIGNAÇÃO 2']))) for _, r in df_designacoes_casais.iterrows())

J_all_idx = list(range(len(df_ordens)))
setup_times = []
for j in J_all_idx:
    for i in J_all_idx:
        if j == i: continue
        dj = df_ordens.loc[j, 'DESIGNAÇÃO']
        di = df_ordens.loc[i, 'DESIGNAÇÃO']
        casal = (tuple(sorted((di, dj))) in casais_set)
        st = 0.0
        if df_ordens.loc[j, 'FAMÍLIA'] != df_ordens.loc[i, 'FAMÍLIA']: st += 3.0
        if dj != di and not casal: st += 1.5
        if df_ordens.loc[j, 'MATRIZ'] != df_ordens.loc[i, 'MATRIZ']: st += 0.5
        if casal: st += 0.3
        setup_times.append((i, j, st))
s = pd.DataFrame(setup_times, columns=['i', 'j', 'setup_time'])

eligibilidade = {}
for idx, row in df_ordens.iterrows():
    designacao = row['DESIGNAÇÃO']; familia = row['FAMÍLIA']
    linha = df_Mj_restricoes[(df_Mj_restricoes['DESIGNAÇÃO']==designacao) & (df_Mj_restricoes['FAMÍLIA']==familia)]
    poss = []
    if not linha.empty:
        if linha.iloc[0]['M4'] == 'SIM': poss.append('M4')
        if linha.iloc[0]['M7'] == 'SIM': poss.append('M7')
        if linha.iloc[0]['M9'] == 'SIM': poss.append('M9')
    eligibilidade[idx] = poss

dbg(f"✅ Dados carregados: {len(df_ordens)} jobs (inclui 3 dummys).  [{time.time()-T0:.1f}s]")

# -------- Tamanho e conversões --------
DUMMY_MAP = {0: 'M4', 1: 'M7', 2: 'M9'}
D = set(DUMMY_MAP.keys())
todos = list(range(len(df_ordens)))
reais = [j for j in todos if j not in D]

Pj_h = df_ordens["TEMPO PROCESSAMENTO (horas)"].astype(float).tolist()
Dj_h = df_ordens["DUE DATE"].astype(float).tolist()
Rj_h = df_ordens["RELEASE_TIME"].astype(float).tolist() if "RELEASE_TIME" in df_ordens.columns else [0.0]*len(df_ordens)

Pj = [int(math.ceil(p / Delta)) for p in Pj_h]
Rj = [int(math.floor(r / Delta)) for r in Rj_h]
Dj = [int(math.ceil(d / Delta)) for d in Dj_h]

Sij = {(int(row.i), int(row.j)): int(math.ceil(float(row.setup_time) / Delta)) for _, row in s.iterrows()}

H_slots = int(math.ceil(H_HOURS / Delta))
dbg(f"🧭 Horizonte H={H_slots} slots (Δ={Delta}h). [{time.time()-T0:.1f}s]")

valid_starts = {}
for j in todos:
    pj = Pj[j]
    tmin = max(0, Rj[j])
    tmax = H_slots - pj
    if math.isfinite(Dj[j]): tmax = min(tmax, max(-1, Dj[j] - pj))
    if tmin > tmax: continue
    for k in M_names:
        if j in D:
            if k == DUMMY_MAP[j]:
                valid_starts[(j, k)] = range(tmin, tmax + 1)
        else:
            if k in eligibilidade.get(j, []):
                valid_starts[(j, k)] = range(tmin, tmax + 1)

# -------- Modelo --------
model = ConcreteModel()
model.J    = Set(initialize=reais, ordered=True)
model.Jall = Set(initialize=todos, ordered=True)
model.M    = Set(initialize=M_names, ordered=True)
model.T    = Set(initialize=range(H_slots), ordered=True)

model.Pj = Param(model.Jall, initialize=lambda m,j: Pj[j])
model.Rj = Param(model.Jall, initialize=lambda m,j: max(0, Rj[j]))
model.Dj = Param(model.Jall, initialize=lambda m,j: Dj[j])

def X_index_init(m): return [(j,k,t) for (j,k), ts in valid_starts.items() for t in ts]
model.X_index = Set(dimen=3, initialize=X_index_init)
model.X = Var(model.X_index, domain=Binary)
def W_index_init(m):
    idx = []
    valid_i = {}
    for i in m.Jall:
        Pi = Pj[i]; Ri = max(0, Rj[i])
        for k in M_names:
            i_ok = (i in D and DUMMY_MAP[i]==k) or (k in eligibilidade.get(i, []))
            if i_ok:
                valid_i[(i,k)] = range(max(0, Ri), H_slots - Pi + 1)
    for (j,k), ts_j in valid_starts.items():
        if j not in reais:
            continue
        for i in todos:
            if i == j or (i,k) not in valid_i:
                continue
            S = Sij.get((i,j), 0)
            Pi = Pj[i]
            taus_i = valid_i[(i,k)]
            for t in ts_j:
                if i in D:
                    # dummy: basta haver tempo para o setup antes de t
                    if t - S >= 0:
                        idx.append((i,j,k,t))
                else:
                    # real: precisa de algum τ viável
                    if any(tau <= t - S - Pi for tau in taus_i):
                        idx.append((i,j,k,t))
    return idx
model.W_index = Set(dimen=4, initialize=W_index_init)
model.W = Var(model.W_index, domain=Binary)


# ============================================================
# 4) RESTRIÇÕES 
# ============================================================
dbg("🧩 A construir restrições...")

# --- PREDECESSOR SÓ ATIVA SE i CABE ANTES EM k ---
model.pred_link = ConstraintList()
for (i, j, k, t) in model.W_index:
    Pi = Pj[i]
    S  = Sij.get((i, j), 0)
    feas = [tau for (ii, kk, tau) in model.X_index
            if ii == i and kk == k and tau <= t - S - Pi]
    if feas:
        model.pred_link.add(sum(model.X[i, k, tau] for tau in feas) >= model.W[i, j, k, t])
    else:
        model.pred_link.add(0 >= model.W[i, j, k, t])

# --- NO MÁXIMO 1 SUCESSOR POR JOB i (qualquer máquina) ---
model.one_succ = ConstraintList()
for i in model.Jall:
    model.one_succ.add(
        sum(model.W[i2, j2, k2, t2]
            for (i2, j2, k2, t2) in model.W_index
            if i2 == i) <= 1
    )

# --- CADA JOB (incl. dummys) COMEÇA EXATAMENTE 1 VEZ ---
model.start_once = ConstraintList()
for j in model.Jall:
    model.start_once.add(
        sum(model.X[j, k, t] for (jj, k, t) in model.X_index if jj == j) == 1
    )

# --- LIGAÇÃO X ↔ W: 1 predecessor por arranque (apenas jobs reais) ---
pred_for_X = defaultdict(list)
for (i, j, k, t) in model.W_index:
    pred_for_X[(j, k, t)].append(i)

model.link_X_W = ConstraintList()
for (j, k, t) in model.X_index:
    if j in reais:
        cand = pred_for_X.get((j, k, t), [])
        model.link_X_W.add(
            (sum(model.W[i, j, k, t] for i in cand) if cand else 0) == model.X[j, k, t]
        )
    # dummys não têm predecessor real

# --- DUMMY É O PRIMEIRO NA SUA MÁQUINA (EXATAMENTE 1 SUCESSOR) ---
model.dummy_first = ConstraintList()
for d, mk in DUMMY_MAP.items():
    terms = [model.W[d, j, mk, t] for (i, j, k, t) in model.W_index if i == d and k == mk]
    if terms:
        model.dummy_first.add(sum(terms) == 1)

# --- COBERTURA DE PROCESSAMENTO (SÓ PROCESSAMENTO CONTA; setups ignorados) ---
proc_cover = defaultdict(list)   # (k,t) -> [(j,tau)]
for (j, k, tau) in model.X_index:
    pj = Pj[j]
    for tt in range(tau, tau + pj):
        if 0 <= tt < H_slots:
            proc_cover[(k, tt)].append((j, tau))

# Capacidade por máquina: <= 1 job a processar
model.machine_capacity = ConstraintList()
for k in M_names:
    for t in model.T:
        if proc_cover[(k, t)]:
            model.machine_capacity.add(
                sum(model.X[j, k, tau] for (j, tau) in proc_cover[(k, t)]) <= 1
            )

# Limite global de equipas/máquinas ativas (apenas processamento)
model.team_cap = ConstraintList()
for t in model.T:
    terms = [sum(model.X[j, k, tau] for (j, tau) in proc_cover[(k, t)])
             for k in M_names if proc_cover[(k, t)]]
    if terms:
        model.team_cap.add(sum(terms) <= b)

# --- START/END 
model.Start = Var(model.J, domain=NonNegativeReals)
model.End   = Var(model.J, domain=NonNegativeReals)

model.start_eq = ConstraintList()
for j in model.J:
    model.start_eq.add(
        model.Start[j] == sum(t * model.X[j, k, t] for (jj, k, t) in model.X_index if jj == j)
    )

model.end_eq = ConstraintList()
for j in model.J:
    model.end_eq.add(model.End[j] == model.Start[j] + model.Pj[j])

model.due_hard = ConstraintList()
for j in model.J:
    if math.isfinite(Dj[j]):
        model.due_hard.add(model.End[j] <= model.Dj[j])

# --- MAKESPAN ---
model.Makespan = Var(domain=NonNegativeReals)
model.makespan_link = ConstraintList()
for j in model.J:
    model.makespan_link.add(model.Makespan >= model.End[j])

# opcional (ajuda a focar o solver): MS <= H_slots
model.ms_ub = Constraint(expr=model.Makespan <= H_slots)

# --- MÉTRICAS EM HORAS (obj) ---
model.SetupTotalHours = Expression(
    expr=sum(Sij.get((i, j), 0) * Delta * model.W[i, j, k, t]
             for (i, j, k, t) in model.W_index)
)

H = H_slots * Delta
UB_sumC = len(reais) * (H + 1.0)
model.SumCompletion = Expression(expr=sum(model.End[j] for j in model.J))

W = UB_sumC + 1.0  # garante prioridade absoluta aos setups
model.obj = Objective(expr= W * model.SetupTotalHours + model.SumCompletion, sense=minimize)

solver = SolverFactory('gurobi')
solver.options.update(dict(Threads=7, MIPGap=0.02))
results = solver.solve(model, tee=True)

# -------- Extração e gráfico --------
job_start_h, job_dur_h, job_machine = {}, {}, {}
for j in todos:
    for k in M_names:
        for t in valid_starts.get((j, k), []):
            if value(model.X[j, k, t]) >= 0.5:
                job_start_h[j] = t * Delta
                job_dur_h[j] = Pj[j] * Delta
                job_machine[j] = k

sequencias = defaultdict(list)
for j, st in sorted(job_start_h.items(), key=lambda x: x[1]):
    sequencias[job_machine[j]].append(j)

print("\n=== SEQUÊNCIA DE JOBS POR MÁQUINA ===")
for k in sorted(sequencias):
    print(f"Máquina {k}: " + " → ".join(map(str, sequencias[k])))

print(f"\nMakespan (h): {value(model.Makespan) * Delta:.2f}")
total_setup_slots = sum(Sij.get((i, j), 0) * value(model.W[i, j, k, tau]) for (i, j, k, tau) in model.W_index)
total_setup_hours = total_setup_slots * Delta
total_proc_hours = sum(Pj[j] * Delta for j in reais)
print(f"Tempo total em setups (h): {total_setup_hours:.2f} ({100 * total_setup_hours / (total_setup_hours + total_proc_hours):.1f}%)")

SETUP_COLOR = 'orange'
SETUP_ALPHA = 0.45

machines_sorted = sorted(sequencias.keys())
yindex = {k: i for i, k in enumerate(machines_sorted)}

fig, ax = plt.subplots(figsize=(12, 5))

for (i2, j2, k2, t2) in model.W_index:
    if value(model.W[i2, j2, k2, t2]) >= 0.5:
        Sslots = Sij.get((i2, j2), 0)
        if Sslots <= 0: continue
        y = yindex.get(k2, None)
        if y is None: continue
        start_slot = max(0, t2 - Sslots)
        dur_slots = t2 - start_slot
        start_h = start_slot * Delta
        dur_h = dur_slots * Delta
        if dur_h > 0:
            ax.barh(y, dur_h, left=start_h, color=SETUP_COLOR, alpha=SETUP_ALPHA, edgecolor='none', height=0.6, zorder=0)

colors = plt.cm.tab20.colors
for k in machines_sorted:
    y = yindex[k]
    for j in sequencias[k]:
        st  = job_start_h[j]
        dur = job_dur_h[j]
        ax.barh(y, dur, left=st, color=colors[j % len(colors)], edgecolor='black', height=0.6, zorder=1)
        ax.text(st + dur/2, y, f"{j}", va='center', ha='center', fontsize=8, color='white', zorder=2)

ax.set_yticks(list(yindex.values()))
ax.set_yticklabels(list(yindex.keys()))
ax.set_xlabel("Tempo (horas)")
ax.set_title("Gantt — processamento + setups")

setup_patch = mpatches.Patch(color=SETUP_COLOR, alpha=SETUP_ALPHA, label='Setup')
proc_patch  = mpatches.Patch(facecolor='gray', edgecolor='black', label='Processo')
ax.legend(handles=[setup_patch, proc_patch], loc='upper right', frameon=True)

ax.grid(True, axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
