

# !pip install python-bcb pandas-datareader statsmodels seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings
from statsmodels.tsa.api import VAR
import pandas_datareader.data as web
from io import StringIO

# Configurações visuais / de sistema
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Selic via FRED, para evitar instabilidade na requisição
  # Código: INTDSRBRM193N (Interest Rates, Discount Rate for Brazil)
try:
    print("Baixando Selic via FRED (EUA)...")
    df_selic = web.DataReader('INTDSRBRM193N', 'fred', start='2012-01-01')
    df_selic.columns = ['Selic']

    # Garantir que o índice seja datetime
    df_selic.index = pd.to_datetime(df_selic.index)
    print(f"-> Selic OK. Último dado: {df_selic.index[-1].strftime('%m/%Y')}")

except Exception as e:
    print(f"Erro crítico no FRED: {e}")
    raise

# Puxa as outras variáveis via BCB (SGS)
  # IPCA (433), IBC-Br (24363), Desemprego (24369)
codes_bcb = {
    'IPCA': 433,
    'IBC_Br': 24363,
    'Desemprego': 24369
}

try:
    print("Baixando indicadores reais via BCB...")
    df_bcb = sgs.get(codes_bcb, start='2012-01-01')
    print("-> Dados BCB OK.")

except Exception as e:
    print(f"Erro no BCB: {e}")
    raise

# Unificando as bases
  # pandas alinha automaticamente pelas datas (index)
df_full = pd.concat([df_selic, df_bcb], axis=1).dropna()

print(f"
Base consolidada: {len(df_full)} meses de observação.")

df_model = pd.DataFrame()

# IPCA: em nível, mensal
df_model['IPCA'] = df_full['IPCA']

# Atividade: log-diff
df_model['Cresc_Econ'] = np.log(df_full['IBC_Br']).diff() * 100

# Desemprego: first diff
df_model['D_Desemp'] = df_full['Desemprego'].diff()

# Selic: first differences; qual o choque do juros?
  # Importante diferenciar a Selic para medir o impacto da *mudança* na meta
df_model['D_Selic'] = df_full['Selic'].diff()

# Removemos os NaNs gerados pela diferenciação
df_model = df_model.dropna()

print("Dados tratados e estacionários.")
print(df_model.head())

# Estimattiva do VAR
model = VAR(df_model)
# Seleciona automaticamente lags via critério AIC
results = model.fit(maxlags=12, ic='aic')

print(f"Modelo ajustado com {results.k_ar} defasagens (Lags).")

# Usa a função de Resposta ao Impulso (IRF)
  # Projetamos 12 meses à frente
irf = results.irf(12)

# Choque: D_Selic -> Resposta: D_Desemp
plt.figure(figsize=(10, 6))
irf.plot(impulse='D_Selic', response='D_Desemp', orth=True, signif=0.05)

plt.title('Resposta do Desemprego a um Choque na Selic', fontweight='bold')
plt.ylabel('Variação no Desemprego (p.p.)')
plt.xlabel('Meses após o choque')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.subplots_adjust(top=0.9)

plt.show()

print("--- Teste de Causalidade de Granger ---")
print("Hipótese nula (H0): a Selic NÃO causa desemprego")
print("-" * 40)

# Teste F
granger_results = results.test_causality('D_Desemp', 'D_Selic', kind='f')
print(granger_results.summary())

# Interpretação automática
p_valor = granger_results.pvalue
if p_valor < 0.05:
    print(f"
CONCLUSÃO: Rejeitamos H0 (p={p_valor:.4f}).")
    print("Há evidência estatística de que a Selic impacta o Desemprego.")
else:
    print(f"
CONCLUSÃO: Não rejeitamos H0 (p={p_valor:.4f}).")
    print("Não há evidência estatística suficiente (nesta amostra) de causalidade.")

# Códigos do FRED (St. Louis Fed)
tickers = {
    'FEDFUNDS': 'FedFunds',
    'UNRATE': 'Desemprego',
    'CPIAUCSL': 'CPI',
    'INDPRO': 'Ind_Production'
}

try:
    df_us = web.DataReader(list(tickers.keys()), 'fred', start='2000-01-01')
    df_us = df_us.rename(columns=tickers)
    print(f"Base EUA carregada: {len(df_us)} observações mensais.")
except Exception as e:
    print(f"Erro ao acessar FRED: {e}")

# Transformando para taxas de variação
df_model_us = pd.DataFrame()

# Inflação: log-diff do CPI
df_model_us['Inflacao'] = np.log(df_us['CPI']).diff() * 100

# Atividade: log-diff da produção industrial
df_model_us['Cresc_Econ'] = np.log(df_us['Ind_Production']).diff() * 100

# Desemprego e juros: first diffs
df_model_us['D_Desemp'] = df_us['Desemprego'].diff()
df_model_us['D_Juros'] = df_us['FedFunds'].diff()

df_model_us = df_model_us.dropna()

# Estimativa do VAR
model_us = VAR(df_model_us)
fit_us = model_us.fit(maxlags=12, ic='aic')

print(f"Lags utilizados (EUA): {fit_us.k_ar}")

# Resposta ao Impulso (IRF)
irf_us = fit_us.irf(24)

# Plotando
plt.figure(figsize=(10, 6))
irf_us.plot(impulse='D_Juros', response='D_Desemp', orth=True, signif=0.05)
plt.title('EUA: Resposta do Desemprego a um Choque nos Juros (FedFunds)', fontweight='bold')
plt.ylabel('Variação no Desemprego (p.p.)')
plt.xlabel('Meses após o choque')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

print("--- Teste de Causalidade de Granger (US) ---")
print("H0: Juros NÃO causam desemprego")
print("-" * 50)

# Teste F para causalidade conjunta
granger_results = fit_us.test_causality('D_Desemp', 'D_Juros', kind='f')
print(granger_results.summary())

print("
Interpretando:")
if granger_results.pvalue < 0.05:
    print(f"--> Rejeitamos H0 (p-valor: {granger_results.pvalue:.4f}).")
    print("--> Conclusão: Existe causalidade estatística dos Juros para o esemprego nos EUA.")
else:
    print(f"--> Falhamos em rejeitar H0 (p-valor: {granger_results.pvalue:.4f}).")
