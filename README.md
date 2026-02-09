# Comparative Analysis: Monetary Policy Transmission (Brazil vs. US)

## Objective
This project conducts a **comparative econometric analysis** of monetary policy transmission mechanisms in **Brazil** and the **United States**. 

Using **Vector Autoregression (VAR)** models, we contrast how interest rate shocks (Selic vs. Fed Funds Rate) impact unemployment in an emerging economy versus a developed one. The goal is to empirically test the efficacy of the **Phillips Curve** in environments with different levels of economic volatility.

## Data & Sources
To ensure a robust comparison, we established a unified pipeline extracting data from official sources for both countries (2000-Present):

### 🇧🇷 Brazil (Emerging Market)
* **Source:** Central Bank of Brazil (BCB/SGS).
* **Variables:** Selic Rate, Unemployment (PNADC), IPCA (Inflation), Economic Activity (IBC-Br).

### 🇺🇸 USA (Developed Market)
* **Source:** Federal Reserve Economic Data (FRED - St. Louis Fed).
* **Variables:** Fed Funds Rate, Unemployment Rate, CPI (Inflation), Industrial Production.

## Technologies
* **Language:** Python 3.x
* **Libraries:** `statsmodels` (Econometrics), `python-bcb`, `pandas-datareader` (Data Extraction), `seaborn` (Visualization).

## Methodology
1.  **Dual Data Pipeline:** Automated extraction scripts for both BCB and FRED APIs.
2.  **Stationarity:** Applied Log-Difference transformations to ensure valid time-series input.
3.  **Modeling:** Estimated two distinct VAR models with lag selection via Akaike Information Criterion (AIC).
4.  **Validation:** Performed **Granger Causality Tests** for both economies to verify statistical precedence.
5.  **Simulation:** Generated **Impulse Response Functions (IRF)** to visualize and compare the reaction of unemployment to monetary shocks over a 12-24 month horizon.

## Key Insights
The project highlights structural differences in policy effectiveness:
* **USA:** Demonstrates a statistically significant transmission channel where interest rate hikes lead to increased unemployment (validating standard theory).
* **Brazil:** Exhibits higher volatility ("noise"), making the transmission channel less linear due to fiscal dominance and external shocks.

## How to Run
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the main notebook: `notebooks/comparative_var_analysis.ipynb`

---
*Developed by Rafael F. Berioni*
