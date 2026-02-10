#!/bin/bash
# Script de instalação do Simulador IAQ
# Para Linux e macOS

set -e

echo "🚀 Instalando Simulador IAQ Avançado..."
echo "========================================"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Por favor, instale Python 3.8 ou superior."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION encontrado"

# Verificar versão mínima
REQUIRED_VERSION="3.8"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo "❌ Python 3.8 ou superior é necessário. Versão atual: $PYTHON_VERSION"
    exit 1
fi

# Criar ambiente virtual
echo ""
echo "📦 Criando ambiente virtual..."
python3 -m venv venv

# Ativar ambiente virtual
echo "🔧 Ativando ambiente virtual..."
source venv/bin/activate

# Atualizar pip
echo "⬆️  Atualizando pip..."
pip install --upgrade pip --quiet

# Instalar dependências
echo "📚 Instalando dependências..."
pip install -r requirements.txt --quiet

# Verificar instalação
echo ""
echo "✔️  Verificando instalação..."
python3 -c "import numpy; import mesa; import streamlit; print('✅ Todas as dependências instaladas com sucesso!')"

# Criar diretórios necessários
echo "📁 Criando diretórios..."
mkdir -p data/results/{raw,processed,visualizations,reports}

echo ""
echo "🎉 Instalação concluída com sucesso!"
echo ""
echo "Próximos passos:"
echo "1. Ative o ambiente virtual:"
echo "   source venv/bin/activate"
echo ""
echo "2. Execute o dashboard:"
echo "   streamlit run final_dashboard.py"
echo ""
echo "   OU"
echo ""
echo "   Execute via linha de comando:"
echo "   python run_simulation.py --scenario office --duration 4"
echo ""
echo "3. Para desativar o ambiente virtual:"
echo "   deactivate"
echo ""
echo "📖 Documentação: docs/"
echo "🔬 Exemplos: examples/"
echo "🧪 Testes: pytest tests/"
echo ""
echo "Boa simulação! 🏢💨"
