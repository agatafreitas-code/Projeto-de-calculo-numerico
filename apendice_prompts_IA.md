# Apendice de Prompts de Inteligencia Artificial

Este documento cumpre a exigencia da Secao 3 (Regra 2) e da Secao 9 do projeto, registrando os prompts utilizados, as respostas aproveitadas, as falhas/alucinacoes detectadas e as verificacoes numericas executadas pela equipe para validar o codigo.

---

## Prompt 1 - Implementacao do campo gravitacional amaciado e vetorizacao em lote

- **Prompt usado:**
  > *"Como implementar em Python com NumPy a aceleracao gravitacional exercida por N atratores fixos sobre uma particula tridimensional usando um potencial amaciado com parametro epsilon? Gostaria tambem de vetorizar o calculo para P particulas simultaneas sem usar laco for."*

- **Resposta aproveitada:**
  A IA sugeriu o uso de broadcasting multidimensional com matrizes de diferenca de forma `(P, N, 3)` para calcular a aceleracao de todas as particulas em uma unica operacao matricial:
  `difference = positions[:, None, :] - attractor_positions[None, :, :]`

- **O que foi corrigido ou confirmado:**
  1. **Sinal da forca e expoente:** O codigo inicial gerado pela IA utilizou erroneamente o expoente $2$ no denominador do vetor diferenca `difference / (distance**2 + eps**2)` e esqueceu o sinal negativo da atracao. A equipe corrigiu manualmente para a expressao analitica deduzida na Parte 0:
     $$\mathbf{g}(\mathbf{r}) = -\sum_{i} \frac{M_i (\mathbf{r} - \mathbf{R}_i)}{\left(\|\mathbf{r} - \mathbf{R}_i\|^2 + \epsilon^2\right)^{3/2}}$$
  2. **Amaciamento consistente:** Confirmou-se que o termo $\epsilon^2$ deve ser somado a norma ao quadrado antes de elevar a $3/2$, preservando a regularidade e a diferenciabilidade da forca.

- **Teste que justificou a aceitacao:**
  - Teste de consistencia analitica com $N=1$, comparando a aceleracao pontual calculada a mao com a saida da funcao vetorizada.
  - Teste automatizado com `np.allclose(scalar_result, vectorized_result, rtol=1e-12, atol=1e-15)` executado na Parte 6, confirmando igualdade numerica estrita entre o loop escalar e a operacao vetorizada.

---

## Prompt 2 - Comparacao de longo prazo entre RK4 e Velocity Verlet na orbita de Kepler

- **Prompt usado:**
  > *"Escreva um script para comparar a conservacao de energia entre Runge-Kutta 4 (RK4) e Velocity Verlet integrando uma orbita de Kepler por 10.000 periodos."*

- **Resposta aproveitada:**
  A estrutura basica dos integradores de passo unico: o algoritmo de Velocity Verlet em dois semi-passos de velocidade e a combinacao linear dos quatro coeficientes do RK4 tradicional.

- **O que foi corrigido ou confirmado:**
  1. **Comparacao a custo computacional igual:** A IA inicialmente gerou o script rodando ambos os metodos com exatamente o mesmo passo $\Delta t = 0.02$. Essa comparacao favorecia injustamente o RK4 por um fator $4$, pois cada passo de RK4 exige $4$ avaliacoes do campo gravitacional, enquanto o Verlet exige apenas $1$. A equipe corrigiu o protocolo experimental definindo:
     $$\Delta t_{\text{RK4}} = 4 \times \Delta t_{\text{Verlet}}$$
  2. **Reversibilidade temporal:** A IA nao havia incluido o teste de reversibilidade no tempo ($t \to -t$). A equipe implementou a inversao de velocidade $\mathbf{v} \to -\mathbf{v}$ ao fim da integracao de ida e a integracao de volta para quantificar a reversibilidade simpletica do Verlet versus a quebra no RK4.

- **Teste que justificou a aceitacao:**
  - Em $10.000$ periodos orbitais a custo igual, o Verlet manteve a energia oscilando em torno do valor inicial com erro maximo de $1.99 \times 10^{-8}$ e erro de reversibilidade de $1.18 \times 10^{-12}$, enquanto o RK4 acumulou deriva secular de $5.72 \times 10^{-6}$ e erro de reversibilidade de $3.05 \times 10^{-3}$.

---

## Prompt 3 - Estimativa do expoente de Lyapunov pelo algoritmo de Benettin

- **Prompt usado:**
  > *"Como calcular numericamente o maior expoente de Lyapunov de uma trajetoria caotica em Python usando integracao por EDO?"*

- **Resposta aproveitada:**
  A explicacao conceitual do crescimento exponencial da separacao de duas condicoes iniciais vizinhas $\delta(t) \approx \delta_0 e^{\lambda t}$ e a recomendacao do algoritmo de Benettin com reescalonamento periodico para evitar saturacao no tamanho caracteristico do sistema.

- **O que foi corrigido ou confirmado:**
  1. **Algoritmo de reescalonamento:** A primeira versao do codigo proposta pela IA apenas integrava duas particulas ate o tempo final e tentava ajustar uma reta em $\ln \|\delta(t)\|$, o que falhava completamente porque a distancia satura na escala espacial do problema ($L \sim 2$). A equipe implementou o algoritmo formal de Benettin com renormalizacao periodica a cada intervalo $\tau$:
     $$\lambda \approx \frac{1}{K\tau}\sum_{k=1}^K \ln \frac{\|\boldsymbol{\delta}(t_k)\|}{\delta_0}$$
  2. **Norma no espaco de fases:** A sugestao da IA usava apenas a distancia espacial $(\|\mathbf{r}_1 - \mathbf{r}_2\|)$. A equipe corrigiu para a norma Euclidiana no espaco de fases completo de 6 dimensoes $(\mathbf{r}, \mathbf{v})$, conforme exigido pelo modelo de primeira ordem.
  3. **Ausencia de amortecimento no teste caotico:** A IA tentou usar $\gamma > 0$, o que mascara a divergencia caotica pelo decaimento dissipativo. O parametro foi fixado em $\gamma = 0$ para a Parte 4.

- **Teste que justificou a aceitacao:**
  - Ponto interior de bacia estavel resultou em $\lambda \le 0$ (ou convergindo para zero), conferindo horizonte de previsibilidade virtualmente infinito ou superior ao tempo de integracao.
  - Ponto localizado na fronteira entre bacias resultou em $\lambda > 0$ com horizonte finito de previsibilidade $t_{\text{prev}} = \frac{1}{\lambda}\ln\frac{\Delta_{\text{tol}}}{\delta_0}$.
