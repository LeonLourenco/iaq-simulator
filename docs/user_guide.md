# Guia do Usuário - Simulador IAQ Avançado

## Índice

1. [Introdução](#introdução)
2. [Instalação](#instalação)
3. [Primeiros Passos](#primeiros-passos)
4. [Interface Gráfica](#interface-gráfica)
5. [Linha de Comando](#linha-de-comando)
6. [Configuração Avançada](#configuração-avançada)
7. [Solução de Problemas](#solução-de-problemas)

## Introdução

Bem-vindo ao Simulador IAQ Avançado! Esta ferramenta permite simular e analisar a qualidade do ar interno (Indoor Air Quality - IAQ) em diferentes tipos de edificações.

### O que é IAQ?

Qualidade do Ar Interno refere-se à qualidade do ar dentro e ao redor de edifícios, especialmente em relação à saúde e conforto dos ocupantes. Fatores importantes incluem:
- Concentração de poluentes (CO₂, VOCs, partículas)
- Temperatura e umidade
- Ventilação e renovação do ar
- Risco de transmissão de doenças

### Casos de Uso

O simulador é útil para:
- **Projetistas de HVAC**: Otimizar sistemas de ventilação
- **Gestores de edifícios**: Planejar intervenções para saúde ocupacional
- **Pesquisadores**: Estudar transmissão de doenças aerossóis
- **Consultores**: Preparar análises para certificações (LEED, WELL)
- **Educadores**: Ensinar conceitos de IAQ e controle de infecções

## Instalação

### Pré-requisitos

- **Sistema Operacional**: Windows 10+, macOS 10.14+, ou Linux
- **Python**: Versão 3.8 ou superior
- **Memória RAM**: 4GB mínimo, 8GB recomendado
- **Espaço em disco**: 500MB para instalação

### Instalação Passo a Passo

1. **Baixe ou clone o repositório**
   ```bash
   git clone https://github.com/seu-usuario/iaq-simulator.git
   cd iaq-simulator
   ```

2. **Crie um ambiente virtual (recomendado)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verifique a instalação**
   ```bash
   python -c "import numpy; import mesa; import streamlit; print('✅ Instalação OK!')"
   ```

## Primeiros Passos

### Execução Rápida

**Opção 1: Interface Gráfica (Recomendado)**
```bash
streamlit run final_dashboard.py
```
Acesse `http://localhost:8501` no navegador.

**Opção 2: Linha de Comando**
```bash
python run_simulation.py --scenario office --duration 4 --visualize
```

**Opção 3: Como Script Python**
```python
from main_model import IAQSimulationModel
import config_final as cfg

scenario = cfg.get_scenario_config('school')
physics_config = cfg.PhysicsConfig()

model = IAQSimulationModel(scenario, physics_config)
while model.running:
    model.step()

print(f"CO₂: {model.current_metrics['average_co2']:.0f} ppm")
```

### Exemplo Completo

Execute o exemplo básico incluído:
```bash
python examples/basic_simulation.py
```

## Interface Gráfica

### Acessando o Dashboard

1. No terminal, navegue até a pasta do projeto
2. Execute: `streamlit run final_dashboard.py`
3. Abra o navegador em `http://localhost:8501`

### Configuração do Cenário

#### Tipo de Edificação
Selecione entre cenários pré-configurados:
- **🏫 Escola**: Sala de aula típica
- **🏢 Escritório**: Open space
- **💪 Academia**: Atividade intensa
- **🏥 Hospital**: Alta ventilação
- **🏠 Residencial**: Ventilação natural

#### Parâmetros de Ocupação
- **Número de ocupantes**: 1-1000 pessoas
- **Taxa de infectados**: 0-50%
- **Uso de máscaras**: 0-100%

#### Configuração de Ventilação
- **Estratégia**:
  - Demand Controlled: Ajusta baseado em CO₂
  - Constant Volume: Vazão fixa
  - Natural: Por aberturas
  - Mixed Mode: Combinação
- **ACH Alvo**: 0.5-20 trocas/hora
  - 2-4 ACH: Mínimo normal
  - 6-8 ACH: Recomendado
  - 10+ ACH: Alto risco
- **Setpoint CO₂**: 400-2000 ppm

### Visualizações

O dashboard oferece 4 abas principais:

1. **📊 Visão Geral**: Mapa de calor em tempo real
2. **📈 Temporal**: Gráficos de evolução
3. **🏢 Zonas**: Análise por zona
4. **👥 Agentes**: Comportamento dos ocupantes

### Intervenções

Aplique intervenções durante a simulação:
- **😷 Máscaras**: Obrigatórias ou recomendadas
- **💨 Ventilação**: Aumentar ACH
- **👥 Ocupação**: Reduzir densidade
- **📏 Distanciamento**: Separação mínima
- **🔧 Purificadores**: Adicionar filtração

## Linha de Comando

### Uso Básico
```bash
python run_simulation.py --scenario TIPO --duration HORAS
```

### Opções Disponíveis

```bash
--scenario TIPO          # school, office, gym, hospital, residential
--duration HORAS         # Horas de simulação (padrão: 8)
--occupants NUM          # Número de ocupantes
--infected-ratio RATIO   # Taxa inicial de infectados (0-1)
--visualize             # Gerar visualizações
--export ARQUIVO.json   # Exportar resultados
--gui                   # Iniciar interface gráfica
```

### Exemplos

```bash
# Simulação básica de escola por 4 horas
python run_simulation.py --scenario school --duration 4

# Escritório com 50 pessoas, 5% infectados
python run_simulation.py --scenario office --occupants 50 --infected-ratio 0.05

# Com exportação e visualizações
python run_simulation.py --scenario gym --duration 2 --visualize --export resultados.json
```

## Configuração Avançada

### Criando Cenário Personalizado

```python
import config_final as cfg

# Definir zonas
zones = [
    cfg.ZoneConfig(
        name="Sala Principal",
        zone_type=cfg.ZoneType.WORKSPACE,
        x=0, y=0, width=15, height=10,
        target_ach=4.0
    )
]

# Criar cenário
scenario = cfg.BuildingScenario(
    name="MeuCenario",
    building_type=cfg.BuildingType.OFFICE,
    total_width=20.0,
    total_height=15.0,
    floor_height=3.0,
    total_occupants=50,
    zones=zones
)
```

### Ajustando Física

```python
physics_config = cfg.PhysicsConfig(
    cell_size=0.2,              # Resolução espacial
    kalman_enabled=True,        # Filtro de Kalman
    pem_correction_active=True  # Correção de pluma
)
```

### Configurando Agentes

```python
agent_config = cfg.AgentConfig(
    intelligence_level="adaptive",  # reactive, adaptive, learning
    movement_pattern="social",      # random, waypoint, social
    mask_wearing_prob=0.3,
    compliance_rate=0.7
)
```

## Solução de Problemas

### Erro: "ModuleNotFoundError"
**Solução**: Instale as dependências
```bash
pip install -r requirements.txt
```

### Erro: "Simulação muito lenta"
**Soluções**:
- Aumente `cell_size` para 0.3-0.5
- Desative `kalman_enabled=False`
- Reduza número de ocupantes
- Aumente `real_time_factor`

### Erro: "MemoryError"
**Soluções**:
- Aumente `cell_size`
- Reduza área simulada
- Reduza duração da simulação

### Dashboard não abre
**Verificações**:
1. Streamlit instalado? `pip install streamlit`
2. Porta 8501 livre? Tente: `streamlit run final_dashboard.py --server.port 8502`
3. Firewall bloqueando? Verifique configurações

### Resultados inesperados
**Verificações**:
1. Parâmetros de entrada corretos?
2. Condições de contorno apropriadas?
3. Tempo de simulação suficiente?
4. Consulte logs para erros

## FAQ

**P: Qual resolução espacial usar?**
R: 0.2m para precisão, 0.5m para rapidez.

**P: Quantos agentes posso simular?**
R: Até 1000 em máquinas normais, 5000+ em servidores.

**P: Posso importar plantas de edifícios?**
R: Futura versão terá importação BIM/CAD.

**P: Como validar resultados?**
R: Compare com medições reais ou modelos estabelecidos.

**P: Licença comercial?**
R: Licença MIT - uso livre, inclusive comercial.

---
*Para mais informações, consulte a documentação técnica.*
