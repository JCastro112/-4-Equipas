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
Delta = 0.5  # Increased from 0.4 to reduce number of time slots
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

# Compute feasible time windows for each (job, machine) pair
def compute_feasible_windows(todos, reais, D, DUMMY_MAP, Pj, Rj, Dj, H_slots, eligibilidade, M_names):
    """Calcula janelas de tempo factíveis para cada (job, máquina)"""
    windows = {}
    
    for j in todos:
        pj = Pj[j]
        tmin = max(0, Rj[j])
        tmax = H_slots - pj
        
        if math.isfinite(Dj[j]):
            tmax = min(tmax, max(-1, Dj[j] - pj))
        
        if tmin > tmax:
            dbg(f"⚠️ Job {j} sem janela viável: tmin={tmin}, tmax={tmax}")
            continue
            
        for k in M_names:
            eligible = False
            if j in D:
                eligible = (k == DUMMY_MAP[j])
            else:
                eligible = (k in eligibilidade.get(j, []))
            
            if eligible:
                windows[(j, k)] = (tmin, tmax)
    
    return windows

feasible_windows = compute_feasible_windows(
    todos, reais, D, DUMMY_MAP, Pj, Rj, Dj, H_slots, eligibilidade, M_names
)

dbg(f"🧭 Janelas factíveis: {len(feasible_windows)} pares. [{time.time()-T0:.1f}s]")

# Build W_index with temporal validation (only 3D now: i,j,k without time)
def build_W_index_validated(reais, todos, feasible_windows, M_names, Pj, Sij):
    """
    Cria W[i,j,k] APENAS se existe sobreposição temporal viável
    """
    W_index = []
    
    for k in M_names:
        jobs_on_k = [j for j in todos if (j, k) in feasible_windows]
        
        for i in jobs_on_k:
            tmin_i, tmax_i = feasible_windows[(i, k)]
            Pi = Pj[i]
            
            for j in jobs_on_k:
                if i == j:
                    continue
                
                # Só criar arco se j for real
                if j not in reais:
                    continue
                
                tmin_j, tmax_j = feasible_windows[(j, k)]
                S = Sij.get((i, j), 0)
                
                # Verificar se existe overlap temporal viável:
                # i pode terminar (tmax_i + Pi) + setup S antes de j começar mais tarde (tmax_j)
                earliest_i_end = tmin_i + Pi
                latest_j_start = tmax_j
                
                if earliest_i_end + S <= latest_j_start:
                    W_index.append((i, j, k))
    
    return W_index

W_index = build_W_index_validated(reais, todos, feasible_windows, M_names, Pj, Sij)

dbg(f"📉 W_index validado: {len(W_index)} triplos (i,j,k). [{time.time()-T0:.1f}s]")

# -------- Modelo Otimizado --------
model = ConcreteModel()
model.J    = Set(initialize=reais, ordered=True)
model.Jall = Set(initialize=todos, ordered=True)
model.M    = Set(initialize=M_names, ordered=True)

model.Pj = Param(model.Jall, initialize=lambda m,j: Pj[j])
model.Rj = Param(model.Jall, initialize=lambda m,j: max(0, Rj[j]))
model.Dj = Param(model.Jall, initialize=lambda m,j: Dj[j])

# Y[j,k] = 1 se job j executa em máquina k (2D instead of 3D X[j,k,t])
model.Y = Var(model.Jall, model.M, domain=Binary)

# Start[j] = tempo de início contínuo (instead of discrete time slot selection)
model.Start = Var(model.Jall, domain=NonNegativeReals, bounds=(0, H_slots))

# W[i,j,k] = 1 se i precede imediatamente j na máquina k (3D instead of 4D)
model.W_index = Set(initialize=W_index, dimen=3)
model.W = Var(model.W_index, domain=Binary)


# ============================================================
# RESTRIÇÕES OTIMIZADAS
# ============================================================
dbg("🧩 A construir restrições...")

M_big = H_slots * 2  # Big-M for disjunctive constraints

# (1) Cada job executa em exatamente uma máquina
model.one_machine = ConstraintList()
for j in model.Jall:
    eligible_k = [k for k in model.M if (j, k) in feasible_windows]
    if eligible_k:
        model.one_machine.add(sum(model.Y[j, k] for k in eligible_k) == 1)
    else:
        raise ValueError(f"Job {j} sem máquinas elegíveis!")

# (2) Y=1 => Start dentro da janela
model.window_bounds = ConstraintList()
for (j, k), (tmin, tmax) in feasible_windows.items():
    model.window_bounds.add(model.Start[j] >= tmin - M_big * (1 - model.Y[j, k]))
    model.window_bounds.add(model.Start[j] <= tmax + M_big * (1 - model.Y[j, k]))

# (3) Release times
model.release = ConstraintList()
for j in model.Jall:
    model.release.add(model.Start[j] >= model.Rj[j])

# (4) Due dates
model.due = ConstraintList()
for j in model.Jall:
    if math.isfinite(Dj[j]):
        model.due.add(model.Start[j] + model.Pj[j] <= model.Dj[j])

# (5) W implica precedência
model.precedence = ConstraintList()
for (i, j, k) in model.W_index:
    Pi = Pj[i]
    S = Sij.get((i, j), 0)
    
    # W=1 => ambos na máquina k
    model.precedence.add(model.W[i, j, k] <= model.Y[i, k])
    model.precedence.add(model.W[i, j, k] <= model.Y[j, k])
    
    # W=1 => Start[j] >= Start[i] + Pi + S
    model.precedence.add(
        model.Start[j] >= model.Start[i] + Pi + S - M_big * (1 - model.W[i, j, k])
    )

# (6) No overlap: Para cada par de jobs na mesma máquina
model.no_overlap = ConstraintList()
for k in model.M:
    jobs_k = [j for j in model.Jall if (j, k) in feasible_windows]
    
    for idx_i, i in enumerate(jobs_k):
        for j in jobs_k[idx_i + 1:]:
            Pi = Pj[i]
            Pj_val = Pj[j]
            
            has_ij = (i, j, k) in model.W_index
            has_ji = (j, i, k) in model.W_index
            
            if has_ij and has_ji:
                # Ambas direções possíveis: escolher uma SE ambos ativos
                model.no_overlap.add(
                    model.W[i, j, k] + model.W[j, i, k] >= 
                    model.Y[i, k] + model.Y[j, k] - 1
                )
            elif has_ij:
                # Só i→j possível
                model.no_overlap.add(
                    model.Start[j] >= model.Start[i] + Pi - 
                    M_big * (2 - model.Y[i, k] - model.Y[j, k])
                )
            elif has_ji:
                # Só j→i possível
                model.no_overlap.add(
                    model.Start[i] >= model.Start[j] + Pj_val - 
                    M_big * (2 - model.Y[i, k] - model.Y[j, k])
                )

# (7) Jobs reais: no máximo 1 predecessor
model.pred_count = ConstraintList()
for j in reais:
    for k in model.M:
        if (j, k) in feasible_windows:
            preds = [(i, jj, kk) for (i, jj, kk) in model.W_index 
                     if jj == j and kk == k]
            if preds:
                model.pred_count.add(
                    sum(model.W[i, j, k] for (i, jj, kk) in preds) <= model.Y[j, k]
                )

# (8) Todos jobs: no máximo 1 sucessor
model.succ_count = ConstraintList()
for i in model.Jall:
    for k in model.M:
        succs = [(ii, j, kk) for (ii, j, kk) in model.W_index 
                 if ii == i and kk == k]
        if succs:
            model.succ_count.add(
                sum(model.W[i, j, k] for (ii, j, kk) in succs) <= 1
            )

# (9) CRÍTICO: Dummys devem ter exatamente 1 sucessor
model.dummy_rules = ConstraintList()
for d, mk in DUMMY_MAP.items():
    model.dummy_rules.add(model.Y[d, mk] == 1)
    
    succs = [(i, j, k) for (i, j, k) in model.W_index 
             if i == d and k == mk]
    if succs:
        model.dummy_rules.add(
            sum(model.W[d, j, mk] for (i, j, k) in succs) == 1
        )
    else:
        dbg(f"❌ Dummy {d} sem sucessores em {mk}!")

# (10) Capacidade por máquina com amostragem temporal (evita explosão de restrições)
SAMPLE_DENSITY = max(1, H_slots // 50)  # ~50 pontos de verificação
sample_times = list(range(0, H_slots, SAMPLE_DENSITY))

model.capacity_sample = ConstraintList()
for k in model.M:
    for t in sample_times:
        # Jobs que PODEM estar ativos em t nesta máquina
        candidates = []
        for j in model.Jall:
            if (j, k) not in feasible_windows:
                continue
            tmin, tmax = feasible_windows[(j, k)]
            pj = Pj[j]
            # Job pode cobrir tempo t se: tmin <= t <= tmax + pj
            if tmin <= t <= tmax + pj:
                candidates.append(j)
        
        # No máximo 1 job ativo (aproximação via amostragem)
        if len(candidates) > 1:
            model.capacity_sample.add(
                sum(model.Y[j, k] for j in candidates) <= len(candidates)
            )

# --- END/MAKESPAN ---
model.End = Var(model.J, domain=NonNegativeReals)
model.end_def = ConstraintList()
for j in model.J:
    model.end_def.add(model.End[j] == model.Start[j] + model.Pj[j])

model.Makespan = Var(domain=NonNegativeReals, bounds=(0, H_slots))
model.makespan_def = ConstraintList()
for j in model.J:
    model.makespan_def.add(model.Makespan >= model.End[j])

# --- MÉTRICAS E OBJETIVO ---
model.SetupTotalSlots = Expression(
    expr=sum(Sij.get((i, j), 0) * model.W[i, j, k] 
             for (i, j, k) in model.W_index)
)

# Objetivo: priorizar minimização de setups
W_weight = len(reais) * H_slots + 1

model.obj = Objective(
    expr=W_weight * model.SetupTotalSlots + sum(model.End[j] for j in model.J),
    sense=minimize
)

dbg(f"✅ Modelo construído. [{time.time()-T0:.1f}s]")
dbg(f"   Variáveis: Y={len(model.Jall)*len(M_names)}, W={len(W_index)}, Start={len(model.Jall)}")
dbg(f"   Restrições principais: ~{len(W_index)*3 + len(model.Jall)*3}")

dbg("🚀 A resolver modelo...")
solver = SolverFactory('gurobi')
solver.options.update({
    'Threads': 7,
    'MIPGap': 0.02,
    'TimeLimit': 300,  # 5 minutos máximo
    'MIPFocus': 1,     # Focar em encontrar soluções viáveis
    'Heuristics': 0.2  # Mais heurísticas para viabilidade
})
results = solver.solve(model, tee=True)

# -------- Extração e gráfico --------
job_start_h = {}
job_dur_h = {}
job_machine = {}

for j in model.Jall:
    for k in model.M:
        if (j, k) in feasible_windows:
            try:
                if value(model.Y[j, k]) >= 0.5:
                    job_start_h[j] = value(model.Start[j]) * Delta
                    job_dur_h[j] = Pj[j] * Delta
                    job_machine[j] = k
                    break
            except:
                continue

sequencias = defaultdict(list)
for j, st in sorted(job_start_h.items(), key=lambda x: x[1]):
    sequencias[job_machine[j]].append(j)

print("\n=== SEQUÊNCIA DE JOBS POR MÁQUINA ===")
for k in sorted(sequencias):
    print(f"Máquina {k}: " + " → ".join(map(str, sequencias[k])))

print(f"\nMakespan (h): {value(model.Makespan) * Delta:.2f}")
total_setup_slots = value(model.SetupTotalSlots)
total_setup_hours = total_setup_slots * Delta
total_proc_hours = sum(Pj[j] * Delta for j in reais)
print(f"Tempo total em setups (h): {total_setup_hours:.2f} ({100 * total_setup_hours / (total_setup_hours + total_proc_hours):.1f}%)")

dbg(f"✅ Otimização completa! [{time.time()-T0:.1f}s]")

SETUP_COLOR = 'orange'
SETUP_ALPHA = 0.45

machines_sorted = sorted(sequencias.keys())
yindex = {k: i for i, k in enumerate(machines_sorted)}

fig, ax = plt.subplots(figsize=(12, 5))

for (i2, j2, k2) in model.W_index:
    if value(model.W[i2, j2, k2]) >= 0.5:
        Sslots = Sij.get((i2, j2), 0)
        if Sslots <= 0: continue
        y = yindex.get(k2, None)
        if y is None: continue
        # Setup happens before j starts
        j_start_slot = value(model.Start[j2])
        start_slot = max(0, j_start_slot - Sslots)
        dur_slots = j_start_slot - start_slot
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
