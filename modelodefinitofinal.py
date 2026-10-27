import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

# ==========================================
# 1. PARÂMETROS GLOBAIS DE ENTRADA
# ==========================================
RANGE_N = range(5, 51, 5)
ALPHA = 0.5  # Percentual de tolerância do risco total (Altere manualmente)
P_PEN = 100  # Penalidade por ponto de risco (Altere manualmente)
NOME_ARQUIVO = 'teste.csv'

COL_NOME = 'Tradução de Requerimento'
COL_AREA = 'table.requirements.areaName'
COL_TEMPO = 'Tempo de Análise'
COL_RISCO = 'GUT_Score'

# ==========================================
# 2. CARREGAMENTO E LEITURA DOS DADOS
# ==========================================
try:
  if NOME_ARQUIVO.endswith(('.xlsx', '.xls')):
    df_raw = pd.read_excel(NOME_ARQUIVO)
  else:
    df_raw = pd.read_csv(NOME_ARQUIVO, encoding='latin1', sep=';')
except FileNotFoundError:
  print(f"❌ Erro: O arquivo '{NOME_ARQUIVO}' não foi encontrado.")
  exit()

df = df_raw[[COL_NOME, COL_AREA, COL_TEMPO, COL_RISCO]].copy()
df = df.dropna(subset=[COL_AREA, COL_TEMPO, COL_RISCO]).reset_index(drop=True)
df[COL_AREA] = df[COL_AREA].astype(str).str.strip()

df.loc[df[COL_AREA] == 'Compliance', COL_TEMPO] = 0.0

num_tipos_docs = len(df)
t = df[COL_TEMPO].values
G = df[COL_RISCO].values

# ==========================================
# 3. LOOPING DE SIMULAÇÃO VARIANDO N
# ==========================================
resultados = []
lista_r_max = []  # Armazena o R_max dinâmico para plotar no gráfico

for demanda_N in RANGE_N:
    
  # R_MAX Dinâmico: alpha * (N * soma(G_i))
  soma_G = np.sum(G)
  r_max_dinamico = ALPHA * (demanda_N * soma_G)
  lista_r_max.append(r_max_dinamico)

  CAPACIDADE_AREAS = {
      'Ambiental': 1320.0,
      'Qualidade': 1320.0,
      'Jurídico': 0.0,
      'Resp. Social': 0.0,
      'Logística': np.inf,
      'Nutrição': np.inf,
      'Compliance': np.inf,
      'Financeiro': 1320.0,
  }

  c = (demanda_N * t) - (demanda_N * G * P_PEN)

  linhas_A = []
  ub_list = []

  linhas_A.append(-demanda_N * G)
  ub_list.append(r_max_dinamico - np.sum(demanda_N * G))

  for area, limite_tempo_mensal in CAPACIDADE_AREAS.items():
    mascara_area = (df[COL_AREA] == area).astype(float).values
    linhas_A.append(demanda_N * t * mascara_area)
    ub_list.append(limite_tempo_mensal)

  A = np.vstack(linhas_A)
  ub = np.array(ub_list)
  lb = np.full_like(ub, -np.inf)

  constraints = LinearConstraint(A, lb, ub)
  bounds = Bounds(0, 1)
  integrality = np.ones(num_tipos_docs)

  res = milp(
      c=c, integrality=integrality, bounds=bounds, constraints=constraints
  )

  if res.success:
    solucao = res.x.astype(int)
    qtd_analisados = np.sum(solucao)
    qtd_descartados = num_tipos_docs - qtd_analisados

    analisados_mes = qtd_analisados * demanda_N
    descartados_mes = qtd_descartados * demanda_N

    risco_residual = np.sum(G[solucao == 0]) * demanda_N
    constante_z = demanda_N * P_PEN * np.sum(G)
    Z_otimo = res.fun + constante_z

    resultados.append({
        'N': demanda_N,
        'Status': 'Viável',
        'Tipos_Analisados': qtd_analisados,
        'Tipos_Descartados': qtd_descartados,
        'Docs_Analisados_Mes': analisados_mes,
        'Docs_Descartados_Mes': descartados_mes,
        'Risco_Residual': risco_residual,
        'Z_Otimo': Z_otimo,
    })
  else:
    resultados.append({
        'N': demanda_N,
        'Status': 'Inviável',
        'Tipos_Analisados': 0,
        'Tipos_Descartados': num_tipos_docs,
        'Docs_Analisados_Mes': 0,
        'Docs_Descartados_Mes': num_tipos_docs * demanda_N,
        'Risco_Residual': np.sum(G) * demanda_N,
        'Z_Otimo': np.nan,
    })

df_res = pd.DataFrame(resultados)

# ==========================================
# 4. GERAÇÃO DOS GRÁFICOS PAINEL COM ESPAÇAMENTO AJUSTADO
# ==========================================
plt.style.use(
    'seaborn-v0_8-whitegrid'
    if 'seaborn-v0_8-whitegrid' in plt.style.available
    else 'default'
)

fig, axs = plt.subplots(2, 2, figsize=(15, 11))

fig.suptitle(
    'Análise de Sensibilidade da Capacidade Operacional (AS-IS) vs. Demanda'
    ' (N)',
    fontsize=16,
    fontweight='bold',
    y=0.98,
)

# Gráfico 1: Documentos Analisados vs. Descartados
axs[0, 0].bar(
    df_res['N'],
    df_res['Docs_Analisados_Mes'],
    label='Analisados (Prévia)',
    color='#2ca02c',
    width=3,
)
axs[0, 0].bar(
    df_res['N'],
    df_res['Docs_Descartados_Mes'],
    bottom=df_res['Docs_Analisados_Mes'],
    label='Descartados (Regularização)',
    color='#d62728',
    width=3,
)
axs[0, 0].set_title(
    'Volume Mensal de Documentos Inspecionados vs. Vazados', pad=12
)
axs[0, 0].set_xlabel('Demanda Mensal de Homologações (N)')
axs[0, 0].set_ylabel('Total de Documentos no Mês')
axs[0, 0].legend()

# Gráfico 2: Risco Residual Mensal (Com R_MAX Dinâmico)
axs[0, 1].plot(
    df_res['N'],
    df_res['Risco_Residual'],
    marker='o',
    color='#ff7f0e',
    linewidth=2,
    label='Risco Residual (R_res)',
)
axs[0, 1].plot(
    list(RANGE_N),
    lista_r_max,
    color='r',
    linestyle='--',
    linewidth=2,
    label=f'Teto Tolerado Dinâmico (α = {ALPHA*100:.0f}%)',
)
axs[0, 1].set_title('Comportamento do Risco Residual Acumulado', pad=12)
axs[0, 1].set_xlabel('Demanda Mensal de Homologações (N)')
axs[0, 1].set_ylabel('Risco Residual Total')
axs[0, 1].legend()

# Gráfico 3: Quantidade de Tipos de Documentos Aceitos
axs[1, 0].plot(
    df_res['N'],
    df_res['Tipos_Analisados'],
    marker='s',
    color='#1f77b4',
    linewidth=2,
)
axs[1, 0].set_title(
    f'Tipos de Requisitos Inspecionados Prévia (Max: {num_tipos_docs})', pad=12
)
axs[1, 0].set_xlabel('Demanda Mensal de Homologações (N)')
axs[1, 0].set_ylabel('Nº de Tipos de Docs Aprovados')

# Gráfico 4: Valor da Função Objetivo (Z)
axs[1, 1].plot(
    df_res['N'],
    df_res['Z_Otimo'],
    marker='^',
    color='#9467bd',
    linewidth=2,
)
axs[1, 1].set_title('Evolução do Valor Ótimo da Função Objetivo (Z)', pad=12)
axs[1, 1].set_xlabel('Demanda Mensal de Homologações (N)')
axs[1, 1].set_ylabel('Pontuação Z (Horas - Penalidade Risco)')

plt.subplots_adjust(
    top=0.91, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.28
)

plt.savefig('graficos_sensibilidade_ASIS_arejado.png', dpi=300)
plt.show()