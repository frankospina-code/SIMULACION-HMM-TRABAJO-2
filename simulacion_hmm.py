"""
Modelo Oculto de Markov (HMM) para Simulación de Regímenes Hidrometeorológicos
Estándar de Reproducibilidad Q1 - Simulación de Montecarlo y Decodificación de Viterbi
Requisitos: numpy, scipy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 1. REPRODUCIBILIDAD: Fijar semilla maestra
SEED = 42
rng = np.random.default_rng(SEED)


class HydrologicalHMM:
    """Implementación formal de un Modelo Oculto de Markov discreto."""
    
    def __init__(self, A: np.ndarray, B: np.ndarray, pi: np.ndarray, state_names: list, obs_names: list):
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.pi = np.asarray(pi, dtype=np.float64)
        self.state_names = state_names
        self.obs_names = obs_names
        self.N = self.A.shape[0]  # Número de estados ocultos
        self.M = self.B.shape[1]  # Número de observaciones discretas
        self._validate_stochastic_constraints()

    def _validate_stochastic_constraints(self):
        assert np.allclose(self.A.sum(axis=1), 1.0), "Las filas de A deben sumar 1"
        assert np.allclose(self.B.sum(axis=1), 1.0), "Las filas de B deben sumar 1"
        assert np.isclose(self.pi.sum(), 1.0), "La distribución inicial pi debe sumar 1"

    def simulate(self, T: int, random_gen: np.random.Generator):
        """Genera una secuencia de longitud T de estados ocultos (Q) y observaciones (O)."""
        Q = np.zeros(T, dtype=int)
        O = np.zeros(T, dtype=int)

        # Paso t = 0
        Q[0] = random_gen.choice(self.N, p=self.pi)
        O[0] = random_gen.choice(self.M, p=self.B[Q[0]])

        # Pasos t = 1 ... T-1 (Propiedad de Markov de 1er orden)
        for t in range(1, T):
            Q[t] = random_gen.choice(self.N, p=self.A[Q[t - 1]])
            O[t] = random_gen.choice(self.M, p=self.B[Q[t]])

        return Q, O

    def viterbi(self, O: np.ndarray):
        """
        Algoritmo de Viterbi en espacio logarítmico para hallar la secuencia oculta
        más probable Q* dada la secuencia de observaciones O.
        """
        T = len(O)
        eps = 1e-12  # Evitar log(0)
        log_A = np.log(self.A + eps)
        log_B = np.log(self.B + eps)
        log_pi = np.log(self.pi + eps)

        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)

        # Inicialización
        delta[0] = log_pi + log_B[:, O[0]]

        # Recursión
        for t in range(1, T):
            for j in range(self.N):
                seq_probs = delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(seq_probs)
                delta[t, j] = np.max(seq_probs) + log_B[j, O[t]]

        # Terminación y Backtracking
        Q_hat = np.zeros(T, dtype=int)
        Q_hat[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            Q_hat[t] = psi[t + 1, Q_hat[t + 1]]

        return Q_hat

    def forward_log_likelihood(self, O: np.ndarray) -> float:
        """Algoritmo Forward con escalamiento para calcular log P(O | lambda)."""
        T = len(O)
        alpha = np.zeros((T, self.N))
        scales = np.zeros(T)

        alpha[0] = self.pi * self.B[:, O[0]]
        scales[0] = np.sum(alpha[0])
        alpha[0] /= scales[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, O[t]]
            scales[t] = np.sum(alpha[t])
            alpha[t] /= scales[t]

        return float(np.sum(np.log(scales)))


# ==============================================================================
# 2. DEFINICIÓN DE PARÁMETROS Y ESCENARIOS EXPERIMENTALES
# ==============================================================================
states = ["S1 (Seco)", "S2 (Moderado)", "S3 (Tormenta)"]
obs = ["V1 (Sin lluvia)", "V2 (Débil)", "V3 (Fuerte)", "V4 (Alerta Roja)"]

pi_base = [0.50, 0.35, 0.15]
A_base = [
    [0.75, 0.20, 0.05],
    [0.25, 0.55, 0.20],
    [0.10, 0.30, 0.60]
]
B_base = [
    [0.63, 0.32, 0.04, 0.01],
    [0.08, 0.52, 0.35, 0.05],
    [0.01, 0.14, 0.38, 0.47]
]

# Escenario 2: Anomalía La Niña (mayor transición y persistencia en S3)
A_nina = [
    [0.55, 0.30, 0.15],
    [0.15, 0.50, 0.35],
    [0.05, 0.17, 0.78]
]

# Escenario 3: Sensibilidad a condiciones iniciales extremas y ruido de emisión
pi_sens = [0.05, 0.15, 0.80]
B_noisy = [
    [0.50, 0.35, 0.10, 0.05],
    [0.15, 0.45, 0.30, 0.10],
    [0.05, 0.25, 0.35, 0.35]
]

scenarios = {
    "E1_Climatologia_Base": HydrologicalHMM(A_base, B_base, pi_base, states, obs),
    "E2_Anomalia_La_Nina": HydrologicalHMM(A_nina, B_base, pi_base, states, obs),
    "E3_Sensibilidad_Ruido": HydrologicalHMM(A_base, B_noisy, pi_sens, states, obs)
}

# ==============================================================================
# 3. SIMULACIÓN DE MONTECARLO Y ANÁLISIS ESTADÍSTICO (1000 RÉPLICAS x 365 DÍAS)
# ==============================================================================
N_SIM = 1000
T_DAYS = 365
raw_records = []
summary_rows = []

for name, model in scenarios.items():
    scen_rng = np.random.default_rng(SEED)  # Misma semilla inicial por escenario para comparación justa
    viterbi_acc = np.zeros(N_SIM)
    red_alerts = np.zeros(N_SIM)
    storm_days = np.zeros(N_SIM)
    log_liks = np.zeros(N_SIM)

    for i in range(N_SIM):
        Q, O = model.simulate(T_DAYS, scen_rng)
        Q_hat = model.viterbi(O)

        viterbi_acc[i] = np.mean(Q == Q_hat) * 100.0
        red_alerts[i] = np.sum(O == 3)  # Días con observación V4 (Alerta Roja)
        storm_days[i] = np.sum(Q == 2)  # Días reales en estado oculto S3
        log_liks[i] = model.forward_log_likelihood(O)

        raw_records.append({
            "Escenario": name, "Replica": i + 1,
            "Exactitud_Viterbi_Pct": viterbi_acc[i],
            "Dias_Alerta_Roja_V4": red_alerts[i],
            "Dias_Estado_Tormenta_S3": storm_days[i],
            "Log_Verosimilitud": log_liks[i]
        })

    # Cálculo de estadísticos e Intervalos de Confianza al 95%
    for metric_name, data in [
        ("Exactitud Viterbi (%)", viterbi_acc),
        ("Días en Alerta Roja V4 (días/año)", red_alerts),
        ("Días en Régimen S3 (días/año)", storm_days),
        ("Log-Verosimilitud P(O|λ)", log_liks)
    ]:
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        ci_low, ci_high = stats.t.interval(0.95, df=N_SIM - 1, loc=mean, scale=stats.sem(data))
        summary_rows.append({
            "Escenario": name,
            "Métrica": metric_name,
            "Media": round(mean, 2),
            "DE": round(std, 2),
            "Mín": round(np.min(data), 2),
            "Máx": round(np.max(data), 2),
            "IC 95% Inf": round(ci_low, 2),
            "IC 95% Sup": round(ci_high, 2)
        })

df_raw = pd.DataFrame(raw_records)
df_summary = pd.DataFrame(summary_rows)

df_raw.to_csv("resultados_crudos_montecarlo.csv", index=False)
df_summary.to_csv("tabla_resumen_estadistico_Q1.csv", index=False)
print("\n=== TABLA ESTADÍSTICA PARA EL ARTÍCULO Q1 ===")
print(df_summary.to_string(index=False))

# ==============================================================================
# 4. GENERACIÓN DE FIGURAS DE CALIDAD PUBLICABLE (300 DPI)
# ==============================================================================
plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

# FIGURA 1: Decodificación de Viterbi vs Estados Reales ( primeros 100 días de E1 )
sample_rng = np.random.default_rng(SEED)
Q_sample, O_sample = scenarios["E1_Climatologia_Base"].simulate(100, sample_rng)
Q_hat_sample = scenarios["E1_Climatologia_Base"].viterbi(O_sample)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, dpi=300)
days = np.arange(1, 101)

ax1.step(days, Q_sample, where="mid", label="Estado Oculto Real ($Q_t$)", color="#1f77b4", linewidth=2)
ax1.step(days, Q_hat_sample, where="mid", label=r"Estado Estimado Viterbi ($\hat{Q}_t$)", color="#ff7f0e", linestyle="--", linewidth=1.8)ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels(["$S_1$ (Seco)", "$S_2$ (Moderado)", "$S_3$ (Tormenta)"])
ax1.set_title("Figura 1. Reconstrucción de Regímenes Hidrometeorológicos mediante el Algoritmo de Viterbi (T = 100 días)", fontweight="bold")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)

ax2.scatter(days, O_sample, color="#2ca02c", s=25, label="Observación Pluviométrica ($O_t$)")
ax2.plot(days, O_sample, color="#2ca02c", alpha=0.4)
ax2.set_yticks([0, 1, 2, 3])
ax2.set_yticklabels(["$V_1$ (<2mm)", "$V_2$ (2-15mm)", "$V_3$ (15-40mm)", "$V_4$ (>40mm)"])
ax2.set_xlabel("Tiempo (Días del año hidrológico)")
ax2.grid(alpha=0.3)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("Figura1_Decodificacion_Viterbi_300dpi.png", dpi=300)
plt.close()

# FIGURA 2: Comparación de distribuciones entre Escenarios (Días de Alerta Roja y Exactitud)
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

scen_labels = ["E1: Base", "E2: La Niña", "E3: Ruido/Sens."]
data_alerts = [df_raw[df_raw["Escenario"] == k]["Dias_Alerta_Roja_V4"] for k in scenarios.keys()]
data_acc = [df_raw[df_raw["Escenario"] == k]["Exactitud_Viterbi_Pct"] for k in scenarios.keys()]

bp1 = ax_a.boxplot(data_alerts, tick_labels=scen_labels, patch_artist=True)
for patch, color in zip(bp1["boxes"], ["#a6cee3", "#fb9a99", "#b2df8a"]):
    patch.set_facecolor(color)
ax_a.set_title("(a) Frecuencia Anual de Alertas Rojas ($V_4$)", fontweight="bold")
ax_a.set_ylabel("Días por año ($T=365$)")
ax_a.grid(axis="y", alpha=0.3)

bp2 = ax_b.boxplot(data_acc, tick_labels=scen_labels, patch_artist=True)
for patch, color in zip(bp2["boxes"], ["#a6cee3", "#fb9a99", "#b2df8a"]):
    patch.set_facecolor(color)
ax_b.set_title("(b) Desempeño de Decodificación de Viterbi", fontweight="bold")
ax_b.set_ylabel("Exactitud de reconstrucción (%)")
ax_b.grid(axis="y", alpha=0.3)

plt.suptitle("Figura 2. Análisis de Sensibilidad Estocástica bajo Simulación de Montecarlo (N = 1000 réplicas)", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("Figura2_Comparacion_Escenarios_300dpi.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nArchivos CSV y Figuras a 300 DPI generados exitosamente.")
