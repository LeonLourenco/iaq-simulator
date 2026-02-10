# Arquitetura do Simulador IAQ Avançado

## Visão Geral

O Simulador IAQ Avançado é um sistema de simulação integrada que combina física computacional de fluidos (CFD), agentes inteligentes e otimização multiobjetivo para análise de qualidade do ar interno (IAQ).

## Objetivos do Sistema

1. **Modelagem Realista**: Simular o transporte de contaminantes, calor e umidade
2. **Comportamento Humano**: Representar ocupantes com comportamentos adaptativos
3. **Otimização**: Balancear qualidade do ar, conforto térmico e eficiência energética
4. **Interatividade**: Interface para exploração de cenários e intervenções
5. **Extensibilidade**: Arquitetura modular para futuras expansões

## Componentes Principais

### 1. Motor Físico Unificado (`unified_physics.py`)

**Responsabilidade**: Simular transporte físico de espécies no ambiente.

**Modelos Implementados**:
- **Transporte Advecção-Difusão**: Para CO₂, HCHO, vírus
- **Transferência de Calor**: Condução, convecção, fontes
- **Transferência de Umidade**: Evaporação, condensação
- **Pluma Térmica**: Modelo PEM para estratificação
- **Filtro de Kalman**: Estimação de estado

**Métodos Chave**:
- `step(dt, time, agent_data)`: Avança simulação física
- `get_zone_statistics()`: Métricas por zona
- `set_external_conditions()`: Define condições de contorno

### 2. Sistema de Agentes (`advanced_agents.py`)

**Responsabilidade**: Representar ocupantes humanos com comportamentos realistas.

**Classes de Agentes**:
- **`HumanAgent`**: Agente básico com emissões fisiológicas
- **`LearningAgent`**: Agente com aprendizado por reforço
- **`AdaptiveScheduler`**: Escalonador inteligente

**Comportamentos**:
- Emissão de CO₂, calor, umidade, vírus
- Movimento baseado em atividades
- Uso de máscaras e distanciamento
- Cálculo de conforto térmico
- Adaptação a condições ambientais

### 3. Modelo Principal (`main_model.py`)

**Responsabilidade**: Integrar todos os componentes e orquestrar a simulação.

**Funcionalidades**:
- Inicialização de cenários e agentes
- Loop principal de simulação
- Aplicação de intervenções
- Coleta de métricas e dados
- Otimização de parâmetros
- Exportação de resultados

### 4. Sistema de Configuração (`config_final.py`)

**Responsabilidade**: Gerenciar todas as configurações e constantes.

**Componentes**:
- **`BuildingScenario`**: Configuração completa de cenário
- **`PhysicsConfig`**: Parâmetros do motor físico
- **`AgentConfig`**: Comportamento dos agentes
- **`ZoneConfig`**: Definição de zonas
- **Constantes**: Propriedades físicas, limites, padrões

## Modelos Matemáticos

### Transporte de Espécies
```
∂C/∂t = ∇·(D∇C) - ∇·(vC) + S - R
```
Onde:
- C: Concentração [ppm, ppb, quanta/m³]
- D: Coeficiente de difusão [m²/s]
- v: Velocidade do ar [m/s]
- S: Fontes (agentes, materiais) [unidade/s]
- R: Remoção (ventilação, deposição) [unidade/s]

### Risco de Infecção (Wells-Riley Modificado)
```
P = 1 - exp(-p * C_v * Q_b * t)
```

### Conforto Térmico (PMV-PPD Simplificado)
```
PMV = f(T, RH, v, M, I_cl)
PPD = 100 - 95 * exp(-0.03353*PMV⁴ - 0.2179*PMV²)
```

## Decisões de Design

### 1. Escolha do Framework Mesa
- **Vantagem**: Especializado em modelos baseados em agentes
- **Compromisso**: Menos flexível para CFD complexo
- **Solução**: Motor físico customizado integrado ao scheduler do Mesa

### 2. Solver Numérico
- **Abordagem**: Diferenças finitas explícitas
- **Estabilidade**: Condição CFL com fator de segurança
- **Performance**: Operações vetorizadas com NumPy

### 3. Representação Espacial
- **Grid Regular**: Células quadradas de tamanho uniforme
- **Zonas**: Múltiplas zonas com propriedades diferentes
- **Interface entre zonas**: Transferência através de células de fronteira

## Considerações de Performance

### Otimizações Implementadas

1. **Vetorização**: Operações em arrays NumPy
2. **Step Adaptativo**: DT automático baseado em condições CFL
3. **Amostragem Seletiva**: Dados históricos em intervalos
4. **Cálculos Condicionais**: Kalman e PEM podem ser desativados

### Limitações Conhecidas

1. **Grid Size**: < 0.1m torna a simulação muito lenta
2. **Número de Agentes**: > 1000 impacta performance
3. **Tempo Simulado**: Dias/semanas requerem otimizações

## Extensibilidade

### Novas Espécies Químicas
1. Adicionar constantes em `config_final.py`
2. Adicionar grid em `unified_physics.py`
3. Definir fontes em `advanced_agents.py`

### Novos Modelos Físicos
1. Implementar classe especializada
2. Integrar via herança ou composição
3. Adicionar controles de configuração

## Referências Científicas

1. **IAQ Fundamentals**: ASHRAE Handbook - HVAC Applications
2. **Infection Risk**: Wells, W. F. (1955) - Airborne Contagion
3. **Thermal Comfort**: Fanger, P. O. (1970) - Thermal Comfort
4. **CFD for Buildings**: Chen, Q. (2009) - Ventilation performance
5. **Agent-Based Modeling**: Gilbert, N. (2008) - Agent-Based Models

---
*Esta documentação será atualizada conforme o sistema evolui.*
