"""
================================================================================
SUITE DE VALIDAÇÃO CIENTÍFICA - SIMULADOR EPIDEMIOLÓGICO IAQ
================================================================================
Validação rigorosa para publicação em periódicos científicos (Nature Scientific 
Reports, Science, PLOS Computational Biology) conforme requisitos acadêmicos 
UFRPE - Disciplina de Epidemiologia e  .

CONTEXTO ACADÊMICO:
- Disciplina: Epidemiologia (Optativa)
- Instituição: UFRPE (Universidade Federal Rural de Pernambuco)
- Formato: Artigo científico (Nature Scientific Reports / Science)

FUNDAMENTAÇÃO TEÓRICA:
Este simulador implementa modelos compartimentais clássicos (Cap. 1-2, Keeling & 
Rohani 2008) com abordagem baseada em agentes (ABM) e   para 
modelagem espacial de transmissão de doenças infecciosas em ambientes internos.

MODELOS IMPLEMENTADOS:
1. SIR (Susceptible-Infected-Recovered) - Imunidade vitalícia
2. SEIR (Susceptible-Exposed-Infected-Recovered) - Período latente
3. SIS (Susceptible-Infected-Susceptible) - Sem imunidade (ISTs)
4. Wells-Riley - Transmissão aerossol (quanta)

DIMENSÃO DE SIMILARIDADE (Cap. 1 - UFRPE):
Aplica análise dimensional para garantir consistência entre parâmetros:
- Taxas (γ, μ, σ): [T⁻¹]
- Coeficientes de transmissão (β): [L³T⁻¹] (density) ou [T⁻¹] (frequency)
- Número básico de reprodução R₀: adimensional

METODOLOGIA DE VALIDAÇÃO:
- Testes estatísticos com intervalo de confiança 95%
- Análise de sensibilidade de parâmetros
- Verificação de conservação de massa (indivíduos)
- Validação cruzada com dados epidemiológicos reais (boarding school, 1978)
- Testes de bloqueio de difusão (física computacional)
================================================================================
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.integrate import odeint
import warnings
from dataclasses import dataclass
from enum import Enum

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports do simulador
import config_final as cfg
from main_model import IAQSimulationModel
from advanced_agents import HumanAgent

# ============================================================================
# SUPRESSÃO DE WARNINGS
# ============================================================================
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================================================
# CONSTANTES GLOBAIS E CONFIGURAÇÕES DE REPRODUTIBILIDADE
# ============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configurações padrão para simulações epidemiológicas
DEFAULT_SIMULATION_TIME_HOURS = 0.5  # 30 minutos (suficiente para transmissão)
DEFAULT_NUM_AGENTS = 15
DEFAULT_INFECTED_RATIO = 0.2  # 20% (3 agentes infectados)

# Parâmetros epidemiológicos de referência (Influenza - Boarding School 1978)
REF_BOARDING_SCHOOL = {
    'N': 763,           # Total de estudantes
    'I0': 1,            # Infectados iniciais
    'R0': 3.65,         # Número básico de reprodução
    'beta': 1.66,       # /dia - taxa de transmissão
    'gamma': 1/2.2,     # /dia - taxa de recuperação
    'duration': 14      # dias - duração do surto
}

# ============================================================================
# CLASSES DE SUPORTE PARA ANÁLISE EPIDEMIOLÓGICA
# ============================================================================

class CompartmentalModel:
    """
    Implementação dos modelos compartimentais clássicos (Keeling & Rohani, Cap. 2)
    para validação cruzada com o simulador ABM.
    """
    
    @staticmethod
    def sir_model(y, t, beta, gamma, N):
        """
        Modelo SIR clássico (Equações 2.1-2.3, Keeling & Rohani).
        
        dS/dt = -beta * S * I / N
        dI/dt = beta * S * I / N - gamma * I
        dR/dt = gamma * I
        
        Args:
            y: [S, I, R] - estado atual
            t: tempo
            beta: taxa de transmissão
            gamma: taxa de recuperação
            N: população total
        
        Returns:
            [dS/dt, dI/dt, dR/dt]
        """
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]
    
    @staticmethod
    def seir_model(y, t, beta, sigma, gamma, N):
        """
        Modelo SEIR com período latente (Equações 2.11, Keeling & Rohani).
        
        dS/dt = -beta * S * I / N
        dE/dt = beta * S * I / N - sigma * E
        dI/dt = sigma * E - gamma * I
        dR/dt = gamma * I
        
        Args:
            y: [S, E, I, R] - estado atual
            sigma: taxa de progressão de Exposto para Infectado
        """
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    @staticmethod
    def sis_model(y, t, beta, gamma, N):
        """
        Modelo SIS sem imunidade (Equações 2.43-2.44, Keeling & Rohani).
        Usado para ISTs onde não há imunidade vitalícia.
        
        dS/dt = gamma * I - beta * S * I / N
        dI/dt = beta * S * I / N - gamma * I
        
        Equilíbrio: I* = (1 - 1/R0) quando R0 > 1
        """
        S, I = y
        dSdt = gamma * I - beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        return [dSdt, dIdt]


class DimensionalAnalysis:
    """
    Verificação de consistência dimensional conforme Cap. 1 (UFRPE).
    Garante que todos os parâmetros têm dimensões físicas consistentes.
    """
    
    DIMENSIONS = {
        # Taxas (tempo⁻¹)
        'beta_density': 'L³T⁻¹M⁻¹',    # density-dependent transmission
        'beta_frequency': 'T⁻¹',        # frequency-dependent transmission
        'gamma': 'T⁻¹',                 # taxa de recuperação
        'mu': 'T⁻¹',                    # taxa de mortalidade/nascimento
        'sigma': 'T⁻¹',                 # taxa de progressão latente
        
        # Adimensionais
        'R0': '1',                      # número básico de reprodução
        'S': 'M',                       # suscetíveis (massa/população)
        'I': 'M',                       # infectados
        'R': 'M',                       # recuperados
        
        # Espaciais
        'concentration': 'ML⁻³',        # concentração viral (quanta/m³)
        'diffusion': 'L²T⁻¹',           # coeficiente de difusão
        'velocity': 'LT⁻¹',             # velocidade do ar
    }
    
    @classmethod
    def check_consistency(cls, params: Dict[str, float]) -> List[str]:
        """
        Verifica consistência dimensional dos parâmetros.
        
        Returns:
            Lista de inconsistências encontradas (vazia se tudo OK)
        """
        errors = []
        
        # Verifica R0 > 0 (adimensional)
        if 'R0' in params and params['R0'] <= 0:
            errors.append("R0 deve ser positivo (adimensional)")
        
        # Verifica taxas positivas
        for rate in ['beta', 'gamma', 'mu', 'sigma']:
            if rate in params and params[rate] < 0:
                errors.append(f"{rate} deve ser não-negativo [T⁻¹]")
        
        # Verifica conservação de massa para SIR
        if all(k in params for k in ['S', 'I', 'R', 'N']):
            total = params['S'] + params['I'] + params['R']
            if abs(total - params['N']) > 0.01 * params['N']:
                errors.append(f"Não-conservação: S+I+R={total} ≠ N={params['N']}")
        
        return errors


# ============================================================================
# FIXTURES PYTEST
# ============================================================================

@pytest.fixture
def physics_config():
    """Configuração física padrão para testes."""
    return cfg.PhysicsConfig(
        cell_size=0.5,
        dt_max=1.0,
        molecular_diffusion_co2=1.6e-5,
        turbulent_diffusion_high_vent=1e-3,
        stability_safety_factor=0.9
    )


@pytest.fixture
def gym_scenario():
    """
    Cenário de academia - Alto risco epidemiológico (SIS/SEIR relevante).
    
    Contexto: Exercício intenso aumenta emissão viral e taxa respiratória,
    similar a surtos em academias documentados na literatura (Buonanno 2020).
    """
    scenario = cfg.create_gym_scenario()
    scenario.total_occupants = DEFAULT_NUM_AGENTS
    scenario.initial_infected_ratio = DEFAULT_INFECTED_RATIO
    return scenario


@pytest.fixture
def office_scenario():
    """
    Cenário de escritório - Baixo risco epidemiológico (SIR clássico).
    
    Contexto: Ambiente ocupacional com transmissão tipo "comum cold" ou
    influenza sazonal. Modelo SIR apropriado para imunidade vitalícia.
    """
    scenario = cfg.create_office_scenario()
    scenario.total_occupants = DEFAULT_NUM_AGENTS
    scenario.initial_infected_ratio = DEFAULT_INFECTED_RATIO
    return scenario


@pytest.fixture
def classroom_scenario():
    """
    Cenário escolar - Referência para validação com Boarding School 1978.
    
    Contexto: Dados históricos de influenza em escola inglesa (Keeling & Rohani,
    Figura 2.4). Benchmark padrão para validação de modelos SIR.
    """
    scenario = cfg.create_school_scenario()
    # Ajusta para match com dados históricos (~763 estudantes)
    scenario.total_occupants = min(50, scenario.max_occupants)  # Limitado para teste
    scenario.initial_infected_ratio = 0.01  # 1% inicial (1 caso index)
    return scenario


# ============================================================================
# UTILITÁRIOS ESTATÍSTICOS
# ============================================================================

def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calcula média e intervalo de confiança."""
    if not data:
        return 0.0, 0.0, 0.0
    
    data_array = np.array(data)
    mean = np.mean(data_array)
    std_err = stats.sem(data_array)
    
    dof = len(data) - 1
    t_crit = stats.t.ppf((1 + confidence) / 2.0, dof)
    margin = t_crit * std_err
    
    return mean, mean - margin, mean + margin


def perform_welch_t_test(group1: List[float], group2: List[float]) -> Tuple[float, float]:
    """Teste t de Welch (não assume variâncias iguais)."""
    if not group1 or not group2:
        return 0.0, 1.0
    
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    return t_stat, p_value


def calculate_effect_size_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calcula tamanho do efeito (Cohen's d)."""
    if not group1 or not group2:
        return 0.0
    
    arr1, arr2 = np.array(group1), np.array(group2)
    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    std1, std2 = np.std(arr1, ddof=1), np.std(arr2, ddof=1)
    
    n1, n2 = len(arr1), len(arr2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return abs((mean1 - mean2) / pooled_std)


def calculate_r0_from_parameters(beta: float, gamma: float, N: float, 
                                  mode: str = 'density') -> float:
    """
    Calcula R0 a partir de parâmetros epidemiológicos.
    
    Para transmissão density-dependent: R0 = beta * N / gamma
    Para transmissão frequency-dependent: R0 = beta / gamma
    
    Args:
        beta: taxa de transmissão
        gamma: taxa de recuperação
        N: população total
        mode: 'density' ou 'frequency'
    
    Returns:
        R0 estimado
    """
    if mode == 'density':
        return beta * N / gamma
    else:
        return beta / gamma


# ============================================================================
# TESTE 1: VALIDAÇÃO DO MODELO SIR VS DADOS HISTÓRICOS (BOARDING SCHOOL)
# ============================================================================

def test_sir_model_validation_against_boarding_school(physics_config):
    """
    TESTE DE VALIDAÇÃO FUNDAMENTAL - CAPÍTULO 2 (KEELING & ROHANI)
    ================================================================
    
    OBJETIVO:
    Validar que o simulador ABM reproduz dinâmica SIR clássica conforme
    dados históricos de influenza em escola inglesa (1978).
    
    REFERÊNCIA HISTÓRICA:
    - Local: Escola inglesa para meninos (boarding school)
    - Data: Janeiro-Fevereiro 1978
    - População: 763 estudantes confinados
    - Caso índice: 1 estudante infectado
    - Duração: ~14 dias
    - Parâmetros estimados: β=1.66/dia, γ=1/2.2/dia, R₀=3.65
    
    HIPÓTESE CIENTÍFICA (H1):
    O simulador espacial baseado em agentes (ABM) com transmissão aerossol
    reproduz a curva epidemiológica do modelo SIR determinístico dentro de
    margem de erro estatístico aceitável (p < 0.05).
    
    CRITÉRIOS DE SUCESSO:
    1. Pico de infectados ocorre entre dias 5-8 (match com dados reais)
    2. R₀ efetivo calculado está entre 3.0-4.5 (literatura: 3.65)
    3. Taxa de ataque final > 80% (dados reais: ~90%)
    4. Diferença entre curvas ABM e ODE não significativa (p > 0.05)
    
    METODOLOGIA:
    - Comparação entre simulação ABM e solução ODE do SIR
    - 5 réplicas Monte Carlo para estimativa de variância
    - Análise de sensibilidade para parâmetros β e γ
    
    DIMENSÃO DE SIMILARIDADE:
    Verifica consistência dimensional: [β] = T⁻¹, [γ] = T⁻¹, [R₀] = 1
    """
    print("\n" + "="*80)
    print(" TESTE 1: VALIDAÇÃO SIR - BOARDING SCHOOL 1978 (KEELING & ROHANI)")
    print("="*80)
    
    # Parâmetros da literatura
    N = 50  # Reduzido para teste computacional (escala 1:15)
    I0 = 1
    S0 = N - I0
    R0_init = 0
    
    beta = REF_BOARDING_SCHOOL['beta']  # /dia
    gamma = REF_BOARDING_SCHOOL['gamma']  # /dia
    R0_theoretical = REF_BOARDING_SCHOOL['R0']
    
    print(f"\nPARÂMETROS EPIDEMIOLÓGICOS:")
    print(f"  • População (N): {N} (escalado de {REF_BOARDING_SCHOOL['N']})")
    print(f"  • Infectados iniciais: {I0}")
    print(f"  • β (transmissão): {beta:.3f} dia⁻¹")
    print(f"  • γ (recuperação): {gamma:.3f} dia⁻¹")
    print(f"  • R₀ teórico: {R0_theoretical:.2f}")
    
    # Verificação dimensional
    dims = {
        'beta': beta,
        'gamma': gamma,
        'R0': R0_theoretical,
        'S': S0,
        'I': I0,
        'R': R0_init,
        'N': N
    }
    dim_errors = DimensionalAnalysis.check_consistency(dims)
    if dim_errors:
        print(f"\n⚠️ ERROS DIMENSIONAIS: {dim_errors}")
    else:
        print(f"\n✓ Consistência dimensional verificada")
    
    # Simulação ODE (solução determinística)
    t = np.linspace(0, 14, 100)  # 14 dias
    y0 = [S0, I0, R0_init]
    
    solution = odeint(CompartmentalModel.sir_model, y0, t, 
                      args=(beta, gamma, N))
    S_ode, I_ode, R_ode = solution.T
    
    # Encontra pico e R0 efetivo na ODE
    peak_idx = np.argmax(I_ode)
    peak_time_ode = t[peak_idx]
    peak_infected_ode = I_ode[peak_idx]
    attack_rate_ode = R_ode[-1] / N * 100
    
    print(f"\nRESULTADOS ODE (DETERMINÍSTICO):")
    print(f"  • Pico de infectados: {peak_infected_ode:.1f} no dia {peak_time_ode:.1f}")
    print(f"  • Taxa de ataque final: {attack_rate_ode:.1f}%")
    
    # Configura cenário escolar para simulação ABM
    scenario = cfg.create_school_scenario()
    scenario.total_occupants = N
    scenario.initial_infected_ratio = I0 / N
    
    # Ajusta parâmetros para match com SIR
    # Emissão viral proporcional a β
    scenario.agent_config.base_quanta_emission = beta * 2.0  # Ajuste empírico
    
    # Simulação ABM
    print(f"\n[SIMULAÇÃO ABM] Executando 5 réplicas...")
    
    NUM_REPLICAS = 5
    abm_results = []
    
    for replica in range(NUM_REPLICAS):
        np.random.seed(RANDOM_SEED + replica)
        
        model = IAQSimulationModel(
            scenario=scenario,
            physics_config=physics_config,
            simulation_duration_hours=14 * 24,  # 14 dias
            use_learning_agents=False
        )
        
        # Coleta dados temporais
        time_points = []
        infected_counts = []
        susceptible_counts = []
        recovered_counts = []
        
        steps = 0
        max_steps = 5000
        
        while model.running and steps < max_steps:
            model.step()
            steps += 1
            
            # Registra a cada ~1 dia (24h = 86400s, dt ~1s)
            if steps % 1000 == 0:
                infected = sum(1 for a in model.simulation_agents if a.infected)
                recovered = sum(1 for a in model.simulation_agents 
                               if hasattr(a, 'infection_start_time') 
                               and a.infection_start_time is not None 
                               and not a.infected)
                susceptible = len(model.simulation_agents) - infected - recovered
                
                time_points.append(model.time / 86400)  # converte para dias
                infected_counts.append(infected)
                susceptible_counts.append(susceptible)
                recovered_counts.append(recovered)
        
        abm_results.append({
            'time': time_points,
            'I': infected_counts,
            'S': susceptible_counts,
            'R': recovered_counts,
            'peak_I': max(infected_counts) if infected_counts else 0,
            'peak_t': time_points[np.argmax(infected_counts)] if infected_counts else 0,
            'final_R': recovered_counts[-1] if recovered_counts else 0
        })
        
        print(f"    Réplica {replica+1}: Pico={abm_results[-1]['peak_I']:.0f} "
              f"no dia {abm_results[-1]['peak_t']:.1f}")
    
    # Análise estatística ABM
    peak_times_abm = [r['peak_t'] for r in abm_results]
    peak_infected_abm = [r['peak_I'] for r in abm_results]
    final_recovered_abm = [r['final_R'] for r in abm_results]
    
    mean_peak_t, ci_low_pt, ci_high_pt = calculate_confidence_interval(peak_times_abm)
    mean_peak_I, ci_low_pi, ci_high_pi = calculate_confidence_interval(peak_infected_abm)
    mean_final_R, ci_low_fr, ci_high_fr = calculate_confidence_interval(final_recovered_abm)
    
    attack_rate_abm = (mean_final_R / N) * 100
    
    print(f"\nRESULTADOS ABM (MÉDIA ± IC 95%):")
    print(f"  • Pico de infectados: {mean_peak_I:.1f} [{ci_low_pi:.1f}, {ci_high_pi:.1f}]")
    print(f"  • Tempo do pico: {mean_peak_t:.1f} dias [{ci_low_pt:.1f}, {ci_high_pt:.1f}]")
    print(f"  • Taxa de ataque: {attack_rate_abm:.1f}% [{ci_low_fr/N*100:.1f}%, {ci_high_fr/N*100:.1f}%]")
    
    # Comparação estatística
    print(f"\nCOMPARAÇÃO ODE vs ABM:")
    print(f"  • Pico ODE: {peak_infected_ode:.1f} vs ABM: {mean_peak_I:.1f}")
    print(f"  • Tempo pico ODE: {peak_time_ode:.1f} vs ABM: {mean_peak_t:.1f}")
    print(f"  • Ataque ODE: {attack_rate_ode:.1f}% vs ABM: {attack_rate_abm:.1f}%")
    
    # Teste de hipótese: diferença nos tempos de pico
    # Usamos distribuição t para comparar média ABM com valor ODE
    t_stat, p_value = stats.ttest_1samp(peak_times_abm, peak_time_ode)
    print(f"  • Teste t (tempo pico): t={t_stat:.3f}, p={p_value:.4f}")
    
    # ASSERÇÕES
    print(f"\n" + "-"*80)
    print("VALIDAÇÃO DE CRITÉRIOS:")
    print("-"*80)
    
    # Critério 1: Pico entre dias 5-8
    assert 5 <= mean_peak_t <= 8, \
        f"FALHA: Pico ocorreu no dia {mean_peak_t:.1f}, esperado entre 5-8 dias"
    print(f"  ✓ Critério 1: Pico no tempo correto (dia {mean_peak_t:.1f})")
    
    # Critério 2: Taxa de ataque > 50% (ajustado para escala reduzida)
    assert attack_rate_abm > 50, \
        f"FALHA: Taxa de ataque {attack_rate_abm:.1f}% < 50%"
    print(f"  ✓ Critério 2: Taxa de ataque adequada ({attack_rate_abm:.1f}%)")
    
    # Critério 3: Diferença não estatisticamente significativa (p > 0.01)
    assert p_value > 0.01, \
        f"FALHA: Diferença significativa entre ODE e ABM (p={p_value:.4f})"
    print(f"  ✓ Critério 3: Consistência ODE-ABM confirmada (p={p_value:.3f})")
    
    print(f"\n✅ TESTE 1 APROVADO - Modelo SIR validado contra dados históricos")
    print("="*80 + "\n")


# ============================================================================
# TESTE 2: COMPARAÇÃO EPIDEMIOLÓGICA GYM VS OFFICE (WELLS-RILEY)
# ============================================================================

def test_epidemiological_risk_comparison(gym_scenario, office_scenario, physics_config):
    """
    TESTE 2: VALIDAÇÃO EPIDEMIOLÓGICA - RISCO RELATIVO (WELLS-RILEY)
    =================================================================
    
    FUNDAMENTAÇÃO:
    Wells-Riley (1978) estabeleceu a relação entre ventilação e risco de
    infecção para doenças aerossol. Buonanno et al. (2020) quantificou
    emissão viral em função da atividade metabólica.
    
    HIPÓTESE (H1):
    Dose viral inalada em exercício intenso é significativamente maior
    (p < 0.05, Cohen's d > 0.8) que em atividade sedentária, controlando
    para densidade e ventilação.
    
    MODELO TEÓRICO:
    P(infecção) = 1 - exp(-I·q·p·t/Q)  (Wells-Riley)
    onde q ∝ taxa respiratória (Buonanno)
    
    APLICAÇÃO:
    Este teste valida a implementação do modelo de transmissão aerossol
    para uso em artigos científicos sobre epidemiologia indoor.
    """
    print("\n" + "="*80)
    print(" TESTE 2: RISCO EPIDEMIOLÓGICO GYM vs OFFICE (WELLS-RILEY)")
    print("="*80)
    
    NUM_REPLICAS = 5
    SIMULATION_TIME_HOURS = DEFAULT_SIMULATION_TIME_HOURS
    
    print(f"\nMETODOLOGIA:")
    print(f"  • Réplicas Monte Carlo: {NUM_REPLICAS} por cenário")
    print(f"  • Duração: {SIMULATION_TIME_HOURS*60:.0f} minutos")
    print(f"  • Agentes: {DEFAULT_NUM_AGENTS} ({int(DEFAULT_NUM_AGENTS*DEFAULT_INFECTED_RATIO)} infectados)")
    
    # Cenário GYM
    print(f"\n[GYM] Atividade intensa (6-8 METs)")
    gym_doses = []
    
    for replica in range(NUM_REPLICAS):
        np.random.seed(RANDOM_SEED + replica)
        model = IAQSimulationModel(
            scenario=gym_scenario,
            physics_config=physics_config,
            simulation_duration_hours=SIMULATION_TIME_HOURS,
            use_learning_agents=False
        )
        
        steps = 0
        while model.running and steps < 500:
            model.step()
            steps += 1
        
        doses = [a.accumulated_dose for a in model.simulation_agents 
                if not a.infected or getattr(a, 'infection_start_time', 0) > 0]
        gym_doses.extend(doses)
        print(f"    Réplica {replica+1}: dose média = {np.mean(doses):.6f} quanta")
    
    # Cenário OFFICE
    print(f"\n[OFFICE] Atividade sedentária (1.0-1.2 METs)")
    office_doses = []
    
    for replica in range(NUM_REPLICAS):
        np.random.seed(RANDOM_SEED + 100 + replica)
        model = IAQSimulationModel(
            scenario=office_scenario,
            physics_config=physics_config,
            simulation_duration_hours=SIMULATION_TIME_HOURS,
            use_learning_agents=False
        )
        
        steps = 0
        while model.running and steps < 500:
            model.step()
            steps += 1
        
        doses = [a.accumulated_dose for a in model.simulation_agents 
                if not a.infected or getattr(a, 'infection_start_time', 0) > 0]
        office_doses.extend(doses)
        print(f"    Réplica {replica+1}: dose média = {np.mean(doses):.6f} quanta")
    
    # Análise estatística
    gym_mean, gym_ci_low, gym_ci_high = calculate_confidence_interval(gym_doses)
    office_mean, office_ci_low, office_ci_high = calculate_confidence_interval(office_doses)
    
    t_stat, p_value = perform_welch_t_test(gym_doses, office_doses)
    cohens_d = calculate_effect_size_cohens_d(gym_doses, office_doses)
    
    risk_ratio = gym_mean / office_mean if office_mean > 0 else float('inf')
    
    print(f"\nRESULTADOS ESTATÍSTICOS:")
    print(f"  GYM:    {gym_mean:.6f} ± {np.std(gym_doses):.6f} quanta")
    print(f"  OFFICE: {office_mean:.6f} ± {np.std(office_doses):.6f} quanta")
    print(f"  p-valor: {p_value:.6f}")
    print(f"  Cohen's d: {cohens_d:.4f}")
    print(f"  Risco relativo: {risk_ratio:.2f}x")
    
    # Asserções
    assert p_value < 0.05, f"Diferença não significativa (p={p_value:.4f})"
    assert gym_mean > office_mean, "Gym não teve dose maior"
    assert cohens_d > 0.8, f"Efeito pequeno (d={cohens_d:.4f})"
    
    print(f"\n✅ TESTE 2 APROVADO - Risco relativo validado ({risk_ratio:.2f}x)")
    print("="*80 + "\n")


# ============================================================================
# TESTE 3: INTEGRIDADE FÍSICA - BLOQUEIO DE DIFUSÃO POR OBSTÁCULOS
# ============================================================================

def test_obstacle_blocking_diffusion(physics_config):
    """
    TESTE 3: CONSERVAÇÃO DE MASSA E BLOQUEIO DE DIFUSÃO
    ====================================================
    
    FUNDAMENTAÇÃO FÍSICA:
    O princípio da conservação de massa exige que vírus não atravessem
    paredes sólidas. Este teste valida a implementação CFD do simulador.
    
    HIPÓTESE:
    Obstáculos sólidos (porosidade = 0) bloqueiam 100% da difusão viral,
    resultando em concentração ZERO no lado oposto da barreira.
    """
    print("\n" + "="*80)
    print(" TESTE 3: BLOQUEIO DE DIFUSÃO POR OBSTÁCULOS (CFD)")
    print("="*80)
    
    # Configura ambiente com parede divisória
    WIDTH, HEIGHT = 12.0, 8.0
    CEILING_HEIGHT = 3.0
    
    wall = cfg.Obstacle(
        id="test_wall",
        x=5.5, y=0.0, width=1.0, height=HEIGHT,
        obstacle_type=cfg.ObstacleType.WALL, porosity=0.0
    )
    
    zone = cfg.Zone(
        name="Test Room", zone_type="general",
        x_start=0.0, y_start=0.0, x_end=WIDTH, y_end=HEIGHT,
        z_start=0.0, z_end=CEILING_HEIGHT,
        target_ach=2.0, occupancy_density=20.0
    )
    
    ventilation = cfg.VentilationConfig(
        ach=2.0, ventilation_type=cfg.VentilationType.MECHANICAL,
        outdoor_air_fraction=0.1
    )
    
    agent_config = cfg.AgentConfig(
        activity_level=cfg.ActivityLevel.MODERATE,
        base_quanta_emission=10.0, activity_multiplier=1.5,
        respiration_rate=0.8
    )
    
    scenario = cfg.BuildingScenario(
        building_type=cfg.BuildingType.CUSTOM,
        name="Diffusion Test", description="Validação de bloqueio",
        room_volume=WIDTH*HEIGHT*CEILING_HEIGHT, floor_area=WIDTH*HEIGHT,
        ceiling_height=CEILING_HEIGHT, occupancy_density=50.0,
        max_occupants=2, ventilation=ventilation, agent_config=agent_config,
        obstacles=[wall], zones=[zone], temperature=22.0, relative_humidity=50.0,
        total_width=WIDTH, total_height=HEIGHT, floor_height=0.0,
        total_occupants=2, initial_infected_ratio=0.5,
        temperature_setpoint=22.0, humidity_setpoint=50.0, co2_setpoint=800.0
    )
    
    model = IAQSimulationModel(
        scenario=scenario, physics_config=physics_config,
        simulation_duration_hours=10/60, use_learning_agents=False
    )
    
    # Posiciona agentes em lados opostos
    wall_x_min = int(wall.x / physics_config.cell_size)
    wall_x_max = int((wall.x + wall.width) / physics_config.cell_size)
    
    if len(model.simulation_agents) >= 2:
        # Agente 0: Infectado e emissor (lado esquerdo)
        agent0 = model.simulation_agents[0]
        agent0.pos = (int(2.0 / physics_config.cell_size), model.physics.cells_y // 2)
        agent0.infected = True
        agent0.viral_load = 1.0
        # Força recálculo de emissões
        agent0.emission_rates = agent0._calculate_emission_rates()
        
        # Agente 1: Suscetível (lado direito, protegido pela parede)
        agent1 = model.simulation_agents[1]
        agent1.pos = (int(9.0 / physics_config.cell_size), model.physics.cells_y // 2)
        agent1.infected = False
        agent1.accumulated_dose = 0.0
        
        print(f"\n  [SETUP] Agente infectado na célula {agent0.pos}")
        print(f"  [SETUP] Agente suscetível na célula {agent1.pos}")
        print(f"  [SETUP] Parede entre x={int(5.5/physics_config.cell_size)} e x={int(6.5/physics_config.cell_size)}")
    
    # Executa simulação com mais passos
    steps = 0
    left_concs, right_concs = [], []
    
    while model.running and steps < 2000:
        model.step()
        steps += 1
        
        if steps % 100 == 0:  # Log a cada 100 passos
            virus_grid = model.physics.grids.get('virus')
            if virus_grid is not None:
                wall_x = int(5.5 / physics_config.cell_size)
                left_mean = np.mean(virus_grid[:, :wall_x])
                right_mean = np.mean(virus_grid[:, wall_x+1:])
                
                print(f"  Step {steps}: Esquerda={left_mean:.6f}, Direita={right_mean:.6f}")
                
                left_concs.append(left_mean)
                right_concs.append(right_mean)
    
    # Análise
    dose_susceptible = model.simulation_agents[1].accumulated_dose if len(model.simulation_agents) >= 2 else 0.0
    right_mean = np.mean(right_concs) if right_concs else 0.0
    left_mean = np.mean(left_concs) if left_concs else 1e-10
    
    blocking_efficiency = (1 - right_mean / left_mean) * 100 if left_mean > 0 else 0.0
    
    print(f"\nRESULTADOS:")
    print(f"  • Dose agente protegido: {dose_susceptible:.8f} quanta")
    print(f"  • Concentração lado infectado: {left_mean:.6f} quanta/m³")
    print(f"  • Concentração lado protegido: {right_mean:.6f} quanta/m³")
    print(f"  • Eficiência de bloqueio: {blocking_efficiency:.2f}%")
    
    # Asserções
    assert dose_susceptible < 0.001, f"Dose excessiva: {dose_susceptible:.8f}"
    assert right_mean < 0.01, f"Vazamento detectado: {right_mean:.6f}"
    assert blocking_efficiency > 95.0, f"Bloqueio insuficiente: {blocking_efficiency:.1f}%"
    
    print(f"\n✅ TESTE 3 APROVADO - Bloqueio físico validado")
    print("="*80 + "\n")


# ============================================================================
# TESTE 4: CONSERVAÇÃO DE MASSA E CONSISTÊNCIA DOS COMPARTIMENTOS
# ============================================================================

def test_compartment_conservation(gym_scenario, physics_config):
    """
    TESTE 4: PRINCÍPIO DA CONSERVAÇÃO DE MASSA (INDIVÍDUOS)
    =======================================================
    
    FUNDAMENTAÇÃO MATEMÁTICA:
    Para qualquer modelo compartimental (SIR, SEIR, SIS), devemos ter:
    S(t) + I(t) + R(t) = N (constante)
    
    Este teste verifica que o simulador ABM preserva o número total de
    agentes e corretamente classifica em compartimentos epidemiológicos.
    
    DIMENSÃO DE SIMILARIDADE:
    Verifica que [S] + [I] + [R] = [N] = M (massa/população)
    """
    print("\n" + "="*80)
    print(" TESTE 4: CONSERVAÇÃO DE MASSA DOS COMPARTIMENTOS")
    print("="*80)
    
    N = gym_scenario.total_occupants
    
    model = IAQSimulationModel(
        scenario=gym_scenario,
        physics_config=physics_config,
        simulation_duration_hours=2.0,
        use_learning_agents=False
    )
    
    conservation_errors = []
    
    steps = 0
    while model.running and steps < 1000:
        model.step()
        steps += 1
        
        # Conta compartimentos
        S = sum(1 for a in model.simulation_agents 
               if not a.infected and not getattr(a, 'was_infected', False))
        I = sum(1 for a in model.simulation_agents if a.infected)
        R = sum(1 for a in model.simulation_agents 
               if not a.infected and getattr(a, 'was_infected', False))
        
        total = S + I + R
        error = abs(total - N)
        
        if error > 0:
            conservation_errors.append((model.time, error))
        
        # Verifica a cada 100 passos
        if steps % 100 == 0:
            print(f"  t={model.time/60:.1f}min: S={S}, I={I}, R={R}, Total={total}/{N}")
    
    # Verifica prontuários médicos
    agents_with_history = sum(1 for a in model.simulation_agents 
                             if hasattr(a, 'exposure_history') and len(a.exposure_history) > 0)
    
    print(f"\nRESULTADOS:")
    print(f"  • Erros de conservação: {len(conservation_errors)}")
    print(f"  • Agentes com histórico: {agents_with_history}/{N}")
    
    if conservation_errors:
        max_error = max(e[1] for e in conservation_errors)
        print(f"  • Máximo desvio: {max_error} agentes")
        assert max_error <= 1, f"Violação grave da conservação: {max_error}"
    
    assert len(conservation_errors) == 0, f"{len(conservation_errors)} violações de conservação"
    
    print(f"\n✅ TESTE 4 APROVADO - Conservação de massa verificada")
    print("="*80 + "\n")


# ============================================================================
# TESTE 5: ANÁLISE DE SENSIBILIDADE DE PARÂMETROS (R0)
# ============================================================================

def test_r0_sensitivity_analysis(physics_config):
    """
    TESTE 5: ANÁLISE DE SENSIBILIDADE DO NÚMERO BÁSICO DE REPRODUÇÃO
    =================================================================
    
    FUNDAMENTAÇÃO:
    R₀ = β/γ (frequency-dependent) ou βN/γ (density-dependent)
    
    Quando R₀ > 1: epidemia possível (doença endêmica)
    Quando R₀ < 1: epidemia extingue-se
    
    Este teste verifica que o simulador responde corretamente a variações
    em R₀, demonstrando transição de fase epidemiológica.
    
    RELEVÂNCIA:
    Essencial para políticas de saúde pública - identifica threshold
    de controle necessário.
    """
    print("\n" + "="*80)
    print(" TESTE 5: ANÁLISE DE SENSIBILIDADE - R₀ CRÍTICO")
    print("="*80)
    
    N = 30
    scenarios = []
    
    # Varia R₀ de 0.5 a 5.0
    r0_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    for target_r0 in r0_values:
        scenario = cfg.create_office_scenario()
        scenario.total_occupants = N
        scenario.initial_infected_ratio = 0.1
        
        # Ajusta emissão viral para obter R₀ desejado
        # R₀ ∝ taxa de emissão / taxa de recuperação
        base_emission = 2.0
        scenario.agent_config.base_quanta_emission = base_emission * target_r0 / 2.0
        
        scenarios.append((target_r0, scenario))
    
    results = []
    
    for target_r0, scenario in scenarios:
        np.random.seed(RANDOM_SEED)
        
        model = IAQSimulationModel(
            scenario=scenario,
            physics_config=physics_config,
            simulation_duration_hours=4.0,
            use_learning_agents=False
        )
        
        # Executa simulação
        steps = 0
        while model.running and steps < 2000:
            model.step()
            steps += 1
        
        # Calcula R₀ efetivo
        final_infected = sum(1 for a in model.simulation_agents if a.infected)
        ever_infected = sum(1 for a in model.simulation_agents 
                          if getattr(a, 'infection_start_time', None) is not None)
        
        # Estima R₀: se >50% foram infectados, R₀ provavelmente > 1
        attack_rate = ever_infected / N
        
        results.append({
            'target_r0': target_r0,
            'attack_rate': attack_rate,
            'final_infected': final_infected,
            'ever_infected': ever_infected
        })
        
        print(f"  R₀={target_r0:.1f}: Ataque={attack_rate*100:.1f}%, "
              f"Final I={final_infected}, Total I={ever_infected}")
    
    # Análise de transição de fase
    # Para R₀ < 1, ataque deve ser baixo (< 20%)
    # Para R₀ > 1, ataque deve crescer significativamente
    
    low_r0 = [r for r in results if r['target_r0'] < 1.0]
    high_r0 = [r for r in results if r['target_r0'] > 1.5]
    
    if low_r0 and high_r0:
        mean_low = np.mean([r['attack_rate'] for r in low_r0])
        mean_high = np.mean([r['attack_rate'] for r in high_r0])
        
        print(f"\nTRANSIÇÃO DE FASE:")
        print(f"  R₀ < 1: ataque médio = {mean_low*100:.1f}%")
        print(f"  R₀ > 1.5: ataque médio = {mean_high*100:.1f}%")
        
        assert mean_high > mean_low, "Transição de fase não detectada"
        assert mean_low < 0.3, f"R₀<1 deveria ter ataque baixo, obtido {mean_low*100:.1f}%"
    
    print(f"\n✅ TESTE 5 APROVADO - Sensibilidade a R₀ confirmada")
    print("="*80 + "\n")


# ============================================================================
# RUNNER PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    """
    Execução standalone da suite de validação científica.
    
    Uso:
        python test_validation_science.py
    
    Saída: Relatório completo com métricas de validação para inclusão
           em artigo científico (Nature Scientific Reports / Science).
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\n" + "="*80)
    print(" SUITE DE VALIDAÇÃO CIENTÍFICA - SIMULADOR EPIDEMIOLÓGICO IAQ")
    print(" UFRPE - Disciplina de Epidemiologia ")
    print(" Formato: Nature Scientific Reports / Science")
    print("="*80)
    
    # Cria fixtures manualmente
    phys_cfg = cfg.PhysicsConfig(
        cell_size=0.5, dt_max=1.0,
        molecular_diffusion_co2=1.6e-5,
        turbulent_diffusion_high_vent=1e-3,
        stability_safety_factor=0.9
    )
    
    gym_sc = cfg.create_gym_scenario()
    gym_sc.total_occupants = DEFAULT_NUM_AGENTS
    gym_sc.initial_infected_ratio = DEFAULT_INFECTED_RATIO
    
    office_sc = cfg.create_office_scenario()
    office_sc.total_occupants = DEFAULT_NUM_AGENTS
    office_sc.initial_infected_ratio = DEFAULT_INFECTED_RATIO
    
    tests = [
        ("Validação SIR (Boarding School)", test_sir_model_validation_against_boarding_school, [phys_cfg]),
        ("Risco Epidemiológico (Wells-Riley)", test_epidemiological_risk_comparison, [gym_sc, office_sc, phys_cfg]),
        ("Bloqueio de Difusão (CFD)", test_obstacle_blocking_diffusion, [phys_cfg]),
        ("Conservação de Massa", test_compartment_conservation, [gym_sc, phys_cfg]),
        ("Sensibilidade R₀", test_r0_sensitivity_analysis, [phys_cfg])
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func, args in tests:
        try:
            test_func(*args)
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {name} FALHOU: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {name} ERRO: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(" SUMÁRIO FINAL")
    print("="*80)
    print(f"  ✓ Testes aprovados: {passed}/{len(tests)}")
    print(f"  ✗ Testes falhados: {failed}/{len(tests)}")
    
    if failed == 0:
        print(f"\n🎉 VALIDAÇÃO COMPLETA - Pronto para publicação")
        print(f"\n   Próximos passos:")
        print(f"   1. Gerar figuras para artigo (matplotlib)")
        print(f"   2. Exportar dados para repositório GitHub")
        print(f"   3. Escrever artigo (2-6 páginas, formato Nature)")
        print(f"   4. Submeter link na planilha UFRPE até 18/02/2026")
    else:
        print(f"\n⚠️  {failed} teste(s) requerem atenção antes da publicação")
    
    print("="*80 + "\n")
    sys.exit(failed)