@echo off
REM Script de instalação do Simulador IAQ
REM Para Windows

echo ================================
echo Instalando Simulador IAQ Avancado
echo ================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado. Instale Python 3.8 ou superior.
    pause
    exit /b 1
)

echo [OK] Python encontrado
echo.

REM Criar ambiente virtual
echo Criando ambiente virtual...
python -m venv venv
if errorlevel 1 (
    echo [ERRO] Falha ao criar ambiente virtual
    pause
    exit /b 1
)

REM Ativar ambiente virtual
echo Ativando ambiente virtual...
call venv\Scripts\activate.bat

REM Atualizar pip
echo Atualizando pip...
python -m pip install --upgrade pip --quiet

REM Instalar dependências
echo Instalando dependencias...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERRO] Falha ao instalar dependencias
    pause
    exit /b 1
)

REM Verificar instalação
echo.
echo Verificando instalacao...
python -c "import numpy; import mesa; import streamlit; print('[OK] Todas as dependencias instaladas!')"
if errorlevel 1 (
    echo [ERRO] Verificacao falhou
    pause
    exit /b 1
)

REM Criar diretórios
echo Criando diretorios...
if not exist "data\results" mkdir data\results
if not exist "data\results\raw" mkdir data\results\raw
if not exist "data\results\processed" mkdir data\results\processed
if not exist "data\results\visualizations" mkdir data\results\visualizations

echo.
echo ================================
echo Instalacao concluida com sucesso!
echo ================================
echo.
echo Proximos passos:
echo.
echo 1. Ative o ambiente virtual:
echo    venv\Scripts\activate
echo.
echo 2. Execute o dashboard:
echo    streamlit run final_dashboard.py
echo.
echo    OU execute via linha de comando:
echo    python run_simulation.py --scenario office --duration 4
echo.
echo 3. Para desativar:
echo    deactivate
echo.
echo Documentacao: docs\
echo Exemplos: examples\
echo Testes: pytest tests\
echo.
pause
