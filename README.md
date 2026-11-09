# Modelado Estocástico de Regímenes Hidrometeorológicos mediante Modelos Ocultos de Markov (HMM)

Repositorio oficial de código y datos para el artículo: **"Modelado Estocástico de Regímenes Hidrometeorológicos y Alertas Pluviométricas en Cuencas Tropicales Andinas mediante Modelos Ocultos de Markov (HMM)"**.

**Autores:** Deiner Andrés Viera Torres, Frank Mateo Ospina, Santiago Pérez Pino y Santiago Parra Pineda  
**Institución:** Institución Universitaria Digital de Antioquia — Curso de Simulación (PREICA2602B020045)

## 1. Descripción del Proyecto
Este proyecto implementa desde cero y de manera vectorizada un Modelo Oculto de Markov discreto $\lambda = (S, V, A, B, \pi)$ fundamentado en distribuciones de probabilidad continuas Gamma para simular regímenes atmosféricos ocultos ($S_1$: Seco, $S_2$: Convección Moderada, $S_3$: Tormenta Severa) y alertas pluviométricas ($V_1$ a $V_4$). Incluye:
- Generador estocástico de Montecarlo ($N_{\text{sim}} = 1000$ réplicas anuales de $T = 365$ días por escenario).
- Algoritmo de decodificación óptima de **Viterbi** en escala logarítmica.
- Algoritmo **Forward** con escalamiento para cálculo de log-verosimilitud $\ln P(O \mid \lambda)$.

## 2. Requisitos e Instalación
Se requiere **Python 3.10+** y las librerías listadas en `requirements.txt`. Para instalarlas:

```bash
pip install -r requirements.txt
