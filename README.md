# Simulador de Transmissão Aérea de Doenças (ABM + CFD)

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Status](https://img.shields.io/badge/status-academic--release-orange)

> **Simulador híbrido de Epidemiologia Computacional** que combina Modelagem Baseada em Agentes (ABM) com Dinâmica de Fluidos Computacional (CFD) simplificada para estimar riscos de transmissão viral (SARS-CoV-2, Influenza) em ambientes internos.

---

## Sobre o Projeto

Este software foi desenvolvido para a disciplina de **Epidemiologia** da **UFRPE**. Diferente de modelos SIR tradicionais que assumem mistura homogênea, este simulador considera a **geometria do ambiente**, a **ventilação (ACH)** e o **comportamento dos ocupantes** para calcular a dose viral inalada (Quanta) usando a equação de **Wells-Riley**.

### Principais Funcionalidades

- **Micro-Física (CFD):** Resolve a equação de advecção-difusão para transporte de aerossóis
- **Comportamento Humano:** Agentes com rotinas de trabalho/estudo e máscaras de proteção
- **Validação Científica:** Calibrado com dados históricos de surtos reais
- **Visualização:** Dashboard interativo em Streamlit e CLI robusta

---

## Instalação

Clone o repositório e instale as dependências em um ambiente virtual Python.

```bash
# 1. Clone o repositório
git clone https://github.com/leonsantos/iaq-epidemic-simulator.git
cd iaq-epidemic-simulator

# 2. Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## Uso Rápido (CLI)

O simulador pode ser executado diretamente via linha de comando para automação e geração de dados.

### Exemplo 1: Simulação de Sala de Aula

Simula uma escola com 30 alunos, 4 trocas de ar por hora, durante 4 horas.

```bash
python run.py --scenario school --occupants 30 --ach 4.0 --duration 4.0 --plot
```

Gera um gráfico `sir_school_TIMESTAMP.png` ao final.

### Exemplo 2: Comparar Ventilação em Escritório

Simula um escritório mal ventilado (ACH 1.0) com exportação de dados.

```bash
python run.py --scenario office --ach 1.0 --output resultados_office_ach1.json
```

### Exemplo 3: Executar o Dashboard Interativo

Para visualizar mapas de calor e navegar na simulação em tempo real:

```bash
streamlit run src/dashboard.py
```

---

## Estrutura do Projeto

```
iaq-epidemic-simulator/
├── src/                # Código Fonte Principal
│   ├── agents.py       # Lógica dos agentes (infecção, movimento)
│   ├── config.py       # Parâmetros científicos e Schema JSON
│   ├── environment.py  # Fachada de geometria e obstáculos
│   ├── model.py        # Orquestrador da simulação (Mesa)
│   └── physics.py      # Motor CFD (Advecção-Difusão)
├── scenarios/          # Arquivos de configuração de cenários (JSON)
├── tests/              # Testes unitários e validação científica
├── docs/               # Artigos e referências bibliográficas
├── results/            # Saída de logs e gráficos gerados
├── run.py              # Ponto de entrada CLI
└── requirements.txt    # Dependências do Python
```

---

## Validação Científica

Este simulador não utiliza parâmetros arbitrários. O núcleo do modelo é validado por testes automatizados contra literatura estabelecida.

| Teste | Descrição | Referência |
|-------|-----------|------------|
| **Boarding School 1978** | Reproduz a curva epidêmica de um surto histórico de Influenza | Keeling & Rohani (2008) |
| **Wells-Riley** | Valida a probabilidade de infecção baseada na curva Dose-Resposta | Buonanno et al. (2020) |
| **Sensibilidade ACH** | Verifica se o aumento da ventilação reduz monotonicamente o risco | ASHRAE 62.1 |

Para rodar a suíte de validação:

```bash
pytest tests/ -v
```

---

## Referências

O modelo matemático baseia-se nas seguintes publicações:

1. **Buonanno, G., Stabile, L., & Morawska, L.** (2020). *Estimation of airborne viral emission: Quanta emission rate of SARS-CoV-2 for infection risk assessment*. Environment International.

2. **Keeling, M. J., & Rohani, P.** (2008). *Modeling Infectious Diseases in Humans and Animals*. Princeton University Press.

3. **ASHRAE** (2019). *Standard 62.1: Ventilation for Acceptable Indoor Air Quality*.

4. **Wells, W. F.** (1955). *Airborne Contagion and Air Hygiene*. Harvard University Press.

---

## Autoria

**Leon Lourenço da Silva Santos**  
Estudante em Sistemas de Informação – 7º Período
Universidade Federal de Pernambuco (UFPE)

- **Disciplina:** Epidemiologia Computacional
- **Professor Orientador:** Prof. Dr. Jones Albuquerque
- **Ano:** 2026

---

## Licença

**MIT License** - Sinta-se à vontade para usar e modificar para fins acadêmicos.

---

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:

- Reportar bugs
- Sugerir novas funcionalidades
- Enviar pull requests
- Melhorar a documentação

---

## Contato

Para dúvidas ou sugestões sobre o projeto, entre em contato através do GitHub.
