# Bacias de Atração Gravitacionais em R³

**Trabalho de Cálculo Numérico — UERJ**  
**Professor:** Prof. Dr. Vahid Nikoofard  

---

## 1. Ambiente e Dependências

Para instalar todas as dependências necessárias (NumPy, SciPy, Matplotlib, Streamlit, Numba, ReportLab):

```powershell
python -m pip install -r requirements.txt
python -m pip install reportlab
```

---

## 2. Como Rodar a Interface Gráfica Interativa (Streamlit)

Execute no terminal:

```powershell
streamlit run app.py
```

A interface abrirá automaticamente no navegador no endereço:
👉 **[http://localhost:8501](http://localhost:8501)**

### Funcionalidades das Abas da Interface:
- **Parte 0:** Deduções analíticas da física, adimensionalização, identidade de dissipação e critérios de escape.
- **Parte 1 (Kepler):** Verificação de ordem observada ($e=0, 0.5, 0.9$), degradação em alta excentricidade, vetor de Laplace-Runge-Lenz e teste do plano invariante com precisão de máquina.
- **Parte 2 (Energia e Reversibilidade):** Deriva secular do RK4 vs oscilação simplética do Velocity Verlet por $10^4$ períodos e cálculo em tempo real do **Tempo de Cruzamento (*Crossover Time*)**.
- **Parte 3 (Eventos):** Trajetória adaptativa e comparação entre detecção ingênua e root-finding por bissecção com quantificação de $\Pi$.
- **Parte 4 (Caos):** Algoritmo de Benettin para medição de $\lambda$ no interior e na fronteira, e cálculo do horizonte $t_{\text{prev}}$.
- **Parte 5 (Bacias):** Convergência em $\Delta t$ ($\Pi$ vs $\Delta\mu$), sensibilidade de parâmetros e **Trajetórias Ilustrativas 3D com plano semitransparente e resíduo $\mathcal{R}(t)$**.
- **Parte 6 (Desempenho):** Benchmark vetorizado $(P, N, 3)$ vs escalar e curva de fração de partículas ativas.
- **Itens Bônus:** Verificação do integrador de Yoshida (T4), dimensão fractal Box-Counting (T2) e expoente de incerteza (T3).
- **📄 Relatório PDF:** Visualização das 9 figuras oficiais e **botão de download direto do Relatório Oficial em PDF** (`relatorio_bacias_atracao.pdf`), além de botão para recompilação sob demanda.

---

## 3. Como Reproduzir Cada Figura do Relatório

O script unificado `gerar_relatorio_e_figuras.py` gera todas as 9 figuras em alta resolução e compila o relatório oficial em PDF:

```powershell
python gerar_relatorio_e_figuras.py
```

Abaixo está o mapeamento exato de reprodução figura a figura:

| Figura no Relatório | Descrição | Script de Reprodução |
| :--- | :--- | :--- |
| **Figura 1** | Configuração 3D dos atratores, plano $z=0$ e produto misto ($V = 5.05 > 0$) | `python gerar_relatorio_e_figuras.py` (ou Aba "Parte 0" / "Parte 5" no Streamlit) |
| **Figura 2** | Órbitas de Kepler e razão de tempos dinâmicos no afélio/periélio ($\sim 83\times$ para $e=0.9$) | `python parte1_verificacao.py` ou `python verificacao_kepler.py` |
| **Figura 3** | Deriva de energia secular do RK4 vs oscilação do Verlet ($10^4$ períodos) e Tempo de Cruzamento | `python parte2_integradores.py --periods 10000 --dt 0.02` |
| **Figura 4** | Comparação de bacias: Detecção Ingênua vs Root-Finding e mapa de pixels alterados ($\Pi \approx 3.2\%$) | `streamlit run app.py` (Aba Parte 3) ou `python gerar_relatorio_e_figuras.py` |
| **Figura 5** | Algoritmo de Benettin para Lyapunov: ponto estável ($\lambda \le 0$) vs caótico ($\lambda > 0$) e $t_{\text{prev}}$ | `python parte4_caos.py` |
| **Figura 6** | Imagem final convergida de alta resolução ($350 \times 350$) com parâmetros na legenda | `python gerar_relatorio_e_figuras.py` ou Aba Parte 5 no Streamlit |
| **Figura 7** | Estudo de convergência temporal em $\Delta t$: mapa da fronteira fractal e curvas de $\Pi$ vs $\Delta\mu$ | `python parte5_bacias.py` |
| **Figura 8** | Trajetórias espaciais 3D cruzando o corte $z=0$ e gráficos do resíduo de dissipação $\mathcal{R}(t)$ | `python gerar_relatorio_e_figuras.py` ou Aba Parte 5 (Seção 5.3) |
| **Figura 9** | Fração de partículas ativas $P(t)/P_0$ e gráfico de Speedup ($\sim 52\times$) | `python parte6_desempenho.py` |

---

## 4. Arquivos Entregáveis

Conforme a **Seção 9** do enunciado, todos os arquivos finais estão disponíveis na raiz do repositório:
- `relatorio_bacias_atracao.pdf`: Relatório final completo diagramado nas Partes 0 a 6 com apêndices.
- `imagens_bacias/`: Diretório contendo todas as 9 figuras em alta resolução.
- `diario_de_falhas.md`: Registro das 4 falhas numéricas reais (Sintoma, Diagnóstico, Correção, Princípio Geral).
- `apendice_prompts_IA.md`: Apêndice transparente sobre o uso de IA conforme Seção 3 (Regra 2).
- `bonus_itens.py`: Implementação dos itens opcionais T2, T3 e T4.
