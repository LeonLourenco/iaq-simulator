# IAQ-Simulator — Simulador Híbrido de Transmissão Aérea de Doenças

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-academic--release-orange)
![UFRPE](https://img.shields.io/badge/UFRPE-Sistemas%20de%20Informa%C3%A7%C3%A3o-red)

> **Simulador híbrido de Epidemiologia Computacional** que combina o Método de Lattice Boltzmann (LBM) para resolução do escoamento de fluidos com Modelagem Baseada em Agentes (ABM) para estimar riscos de transmissão viral por aerossóis (SARS-CoV-2, Influenza) em ambientes internos — especificamente salas de aula.

---

## Sobre o Projeto

Este software foi desenvolvido para as disciplinas de **Epidemiologia Computacional** e **Autômatos Celulares** da **UFRPE**. Diferente de modelos SIR/SEIR tradicionais que assumem mistura homogênea perfeita, o IAQ-Simulator considera:

- A **geometria real da sala** (10 m × 8 m, malha D2Q9 de 100×80 células)
- O **escoamento do ar** resolvido explicitamente pelas equações de Navier–Stokes via LBM
- O **transporte de aerossóis** por advecção–difusão–reação acoplado ao campo de velocidades
- O **comportamento dos ocupantes** (sentar, socializar, distanciamento social emergente)
- A **dose viral inalada** calculada pelo modelo de dose–resposta de **Wells–Riley**

O resultado é um modelo compartimental **SEI estocástico e espacialmente explícito**, capaz de capturar zonas de alta concentração viral, a influência da posição do agente infeccioso e o impacto real da ventilação sobre o risco epidemiológico.

---

## Arquitetura do Projeto

```
iaq-simulator/
│
├── docs/ 
│   └── IAQ_Simulator.pdf      # Documento do Artigo em formato Nature Latex
│
├── src/                          # Pacote principal do simulador
│   ├── __init__.py               # Exportações do pacote
│   ├── agents.py                 # Agentes móveis com epidemiologia (SEI + Wells-Riley)
│   ├── lbm_core.py               # Motor CFD: LBM D2Q9 + transporte de aerossóis (Numba JIT)
│   └── simulation_engine.py      # Orquestrador: integra LBM, agentes e coleta de dados
│
├── scenarios/                    # Configurações de cenários em JSON
│   └── school.json               # Sala de aula padrão UFRPE (10m × 8m, 25 alunos)
│
├── results/                      # Saídas das simulações (gerado automaticamente)
│   ├── dados_estatisticos_finais.csv   # Médias e desvios-padrão por cenário
│   ├── dados_monte_carlo_raw.csv       # Dados brutos das 20 simulações individuais
│   └── dados_series_temporais.csv      # Séries temporais para gráficos de linha
│
├── app.py                        # Dashboard interativo (Streamlit)
├── batch_experiments.py          # Orquestrador Monte Carlo (4 cenários × 5 repetições)
├── requirements.txt              # Dependências Python
├── LICENSE                       # MIT License
├── .gitignore
└── README.md
```

### Descrição dos Módulos

| Arquivo | Responsabilidade |
|---|---|
| `src/lbm_core.py` | Implementa o passo LBM D2Q9 (colisão BGK, bounce-back, inlets/outlets) e o solver ADR (advecção–difusão–reação) para aerossóis, ambos compilados com **Numba JIT** |
| `src/agents.py` | Define o `BioAgent`: máquina de estados comportamental (Entrando → Sentado → Socializando), emissão de quanta/CO₂ e cálculo estocástico da infecção via Wells–Riley |
| `src/simulation_engine.py` | Acopla LBM e agentes a cada passo de tempo; gerencia condições de contorno de ventilação; coleta o histórico de infectados, CO₂ e vírus |
| `app.py` | Dashboard Streamlit com mapas de calor do campo viral, curvas SEI em tempo real e controles de cenário |
| `batch_experiments.py` | Executa a bateria Monte Carlo: 4 cenários × 5 repetições, exporta os três arquivos de resultados |
| `scenarios/school.json` | Parâmetros base da sala (dimensões, ACH padrão, número de alunos, infectados iniciais) |

---

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/leonlourenco/iaq-simulator.git
cd iaq-simulator

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# ou
.venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

> **Requisito:** Python 3.8+. O módulo `lbm_core.py` usa **Numba** para compilação JIT — na primeira execução haverá um aquecimento de ~10 s enquanto o código é compilado para código de máquina.

---

## Uso

### Dashboard Interativo

Para visualizar mapas de calor do escoamento de ar, campo viral e curvas de infectados em tempo real:

```bash
streamlit run app.py
```

### Bateria Monte Carlo (geração de dados para o artigo)

Executa os 4 cenários com 5 repetições cada (~15–25 min, dependendo da CPU):

```bash
python batch_experiments.py
```

Saídas geradas em `results/`:
- `dados_estatisticos_finais.csv` — estatísticas agregadas (média ± dp) para a Tabela 1 do artigo
- `dados_monte_carlo_raw.csv` — dados brutos para scatter plots
- `dados_series_temporais.csv` — séries temporais para gráficos de linha

### Simulação Programática

```python
from src.simulation_engine import IAQSimulator

sim = IAQSimulator("scenarios/school.json")

historico = sim.run_simulation(
    total_hours=4.0,
    ach_target=4.5,       # Trocas de ar por hora
    ac_power=15.0,        # Potência do ar-condicionado
    window_open=True,     # Janelas abertas
    steps_per_min=60      # Precisão física (1 Hz)
)

print(f"Infectados ao final: {historico['infected_total'][-1]}")
```

---

## Cenários Disponíveis

| Código | Cenário | ACH | Janelas | População |
|--------|---------|-----|---------|-----------|
| **A** | Hermético (Controle) | 0,5 | Fechadas | 25 |
| **B** | Escola Padrão | 4,5 | Abertas | 25 |
| **C** | Ventilação Otimizada | 10,0 | Abertas | 25 |
| **D** | Superlotação | 4,5 | Abertas | 40 |

Novos cenários podem ser criados adicionando um arquivo `.json` em `scenarios/` seguindo o esquema de `school.json`.

---

## Modelo Científico

### Lattice Boltzmann Method (LBM D2Q9)

O escoamento do ar é resolvido pela equação de Boltzmann discreta com operador BGK:

$$f_i(\mathbf{x} + \mathbf{c}_i \Delta t,\; t + \Delta t) = f_i(\mathbf{x}, t) - \omega \left[f_i - f_i^{\text{eq}}\right]$$

### Transporte de Aerossóis (Advecção–Difusão–Reação)

$$\frac{\partial C}{\partial t} + \mathbf{u} \cdot \nabla C = D_C \nabla^2 C - \lambda C + S(\mathbf{x}, t)$$

### Modelo de Dose–Resposta (Wells–Riley Estocástico)

$$P_k(t) = 1 - \exp\!\left(-\frac{D_k(t)}{k_{\text{imun}}}\right), \quad D_k(t) = \sum_{\tau=0}^{t} C\!\left(\mathbf{x}_k, \tau\right) \cdot \alpha$$

A infecção é sorteada estocasticamente a cada passo: se $r \sim \mathcal{U}(0,1) < P_k(t)$, o agente transita $S \to E$.

---

## Resultados (Monte Carlo, n = 5)

| Cenário | Taxa de Ataque (%) | Índice de Dose | CO₂ médio (ppm) |
|---------|-------------------|---------------|-----------------|
| A. Hermético | 99,17 ± 1,86 | 34,77 ± 6,39 | 740,7 |
| B. Escola Padrão | 99,17 ± 1,86 | 2,59 ± 0,36 | 440,8 |
| C. Ventilação Otimizada | 98,33 ± 3,73 | **1,10 ± 0,20** | 418,8 |
| D. Superlotação | 99,49 ± 1,15 | 2,73 ± 0,02 | 465,2 |

> A ventilação otimizada (10 ACH) reduziu o Índice de Dose acumulado em **96,8%** em relação ao cenário hermético.

---

## Referências

1. **Keeling, M.J. & Rohani, P.** (2008). *Modeling Infectious Diseases in Humans and Animals*. Princeton University Press.
2. **Buonanno, G., Stabile, L. & Morawska, L.** (2020). Estimation of airborne viral emission: Quanta emission rate of SARS-CoV-2 for infection risk assessment. *Environment International*, 141, 105794.
3. **Riley, E.C., Murphy, G. & Riley, R.L.** (1978). Airborne spread of measles in a suburban elementary school. *American Journal of Epidemiology*, 107, 421–432.
4. **Succi, S.** (2001). *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond*. Oxford University Press.
5. **ASHRAE** (2019). *Standard 62.1: Ventilation for Acceptable Indoor Air Quality*.

---

## Autoria

**Leon Lourenço da Silva Santos**
Bacharelado em Sistemas de Informação — 7º Período
Universidade Federal Rural de Pernambuco (UFRPE)

- **Disciplinas:** Epidemiologia Computacional · Autômatos Celulares
- **Professor Orientador:** Prof. Dr. Jones Albuquerque
- **Ano:** 2026

---

## Licença

Distribuído sob a licença **MIT**. Consulte o arquivo `LICENSE` para mais detalhes.
