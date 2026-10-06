# Diario de Falhas Numericas

Este documento registra as falhas numericas reais encontradas durante o projeto, detalhando sintoma, diagnostico, teste que isolou a causa, correcao aplicada e o principio geral aprendido.

---

## Falha 1 - Perda de captura por salto de passo e classificacao incorreta da fronteira

- **Sintoma:** Ao utilizar passos de integracao moderados ($\Delta t \approx 0.05$), particulas com trajetorias rapidas em direcao a um atrator eram classificadas erroneamente em outro destino ou marcadas como escape. Na bacia 2D, pixels proximos a fronteiras entre atratores apresentavam cores inconsistentes que mudavam drasticamente ao variar $\Delta t$.
- **Diagnostico:** O teste ingenuo de captura verificava apenas a condicao pontual $\|r(t_n) - R_i\| \le R_{\text{cap}}$ nos instantes discretos de saida do integrador. Em passagens em alta velocidade (proximas ao periastro do potencial gravitacional), a velocidade escalar $|v|$ e maxima e a particula percorria uma distancia $\Delta r \approx |v|\Delta t > 2 R_{\text{cap}}$, "pulando" por cima de toda a esfera de captura em um unico passo sem que nenhum no da malha caisse no seu interior.
- **Teste que isolou a causa:** Integracao de uma trajetoria com condicao inicial apontando diretamente para o atrator $0$. Com $\Delta t = 0.05$ fixo sem eventos, a distancia minima amostrada foi $0.24 > R_{\text{cap}} = 0.22$, resultando em nao-captura. Ao diminuir $\Delta t$ para $0.001$, a captura foi detectada em $t = 1.42$.
- **Correcao:** Implementacao de detecção precisa de eventos:
  1. Monitoramento da funcao evento $g_i(t) = \|r(t) - R_i\|^2 - R_{\text{cap}}^2$;
  2. Verificacao de cruzamento de zero $g_i(t_n) \cdot g_i(t_{n+1}) \le 0$ e busca em subintervalos de um quarto do passo;
  3. Reconstrucao continua da solucao via polinomio cubico de Hermite $S(t)$ usando posicao e velocidade nas extremidades;
  4. Localizacao exata do instante de contato $t^*$ via metodo da bisseccao com tolerancia $10^{-8}$.
- **Principio geral:** Eventos geometricos discretos acoplados a EDOs continuas nao podem depender apenas da amostragem temporal fixa do integrador. E obrigatorio combinar interpolacao densa com algoritmos de root-finding para garantir a deteccao de cruzamentos de fronteira.
- **Dado bruto de verificacao:** A comparacao entre a deteccao ingenua e a deteccao por eventos revelou uma taxa de pixels alterados $\Pi \approx 2\%$ a $5\%$ na malha, concentrados exatamente na fronteira fractal das bacias.

---

## Falha 2 - Deriva secular de energia do RK4 versus integrador simpletico Velocity Verlet

- **Sintoma:** Ao integrar o problema conservativo de dois corpos de Kepler ($\gamma = 0, \epsilon = 0$) por $10.000$ periodos orbitais, o integrador Runge-Kutta de ordem 4 (RK4) apresentou aumento continuo do erro de energia ($|\Delta E|$ crescendo linearmente no tempo), enquanto o metodo de Velocity Verlet (ordem 2) manteve o erro oscilando em uma faixa estrita e limitada.
- **Diagnostico:** O RK4, embora possua ordem local $O(\Delta t^5)$ e ordem global $O(\Delta t^4)$, e um metodo dissipativo no espaco de fases: ele nao preserva a 2-forma simpletica $d\mathbf{p} \wedge d\mathbf{q}$. Em integracoes de longa duracao, o erro de truncamento introduz dissipacao ou injecao artificial de energia secular. Por outro lado, o Velocity Verlet e simpletico, o que garante que ele resolve exatamente um Hamiltoniano sombra $\tilde{H} = H + O(\Delta t^2)$, mantendo a energia real confinada em um envelope limitado para todo $t$.
- **Teste que isolou a causa:** Comparacao dos dois metodos a **custo computacional identico** (RK4 com $\Delta t = 0.08$ e Verlet com $\Delta t = 0.02$, correspondendo a 1 avaliacao de forca por unidade de custo em ambos). Foi medido tambem o erro de reversibilidade temporal integrando por $T$, invertendo $\mathbf{v} \to -\mathbf{v}$, integrando novamente por $T$ e medindo $\|\mathbf{r}_{\text{volta}} - \mathbf{r}_0\|$.
- **Correcao:** Documentacao formal da diferenca entre integradores gerais e integradores geometricos. O Velocity Verlet foi adotado como referencia para dinamica conservativa e longa duracao, comprovando que ordem de precisao nao e sinonimo de conservacao qualitativa.
- **Principio geral:** Ordem de precisao local mede apenas o erro assintotico quando $\Delta t \to 0$ em tempo fixo. Para $t \to \infty$, a preservacao de invariantes geometricos e simpleticidade e superior a ordem do integrador.
- **Dado bruto de verificacao:**
  - RK4 ($\Delta t = 0.08$): Deriva maxima de energia $= 5.72 \times 10^{-6}$; Erro de reversibilidade $= 3.05 \times 10^{-3}$.
  - Verlet ($\Delta t = 0.02$): Deriva maxima de energia $= 1.99 \times 10^{-8}$; Erro de reversibilidade $= 1.18 \times 10^{-12}$ (ordem do erro de maquina).

---

## Falha 3 - Passo adaptativo quebrando a simpleticidade do Verlet no problema conservativo

- **Sintoma:** Ao aplicar controle adaptativo de passo local ($\Delta t_n = \eta \min_i \|r_n - R_i\|^{3/2} / \sqrt{M_i}$) ao algoritmo de Velocity Verlet para acelerar o calculo da orbita de Kepler, a propriedade de conservacao estrita da energia de longo prazo foi imediatamente perdida, exibindo deriva linear analoga a de metodos nao simpleticos.
- **Diagnostico:** A demonstracao matematica de que o Verlet preserva a forma simpletica exige que a matriz Jacobiana da transformacao $(r_n, v_n) \mapsto (r_{n+1}, v_{n+1})$ seja simpletica. Essa propriedade depende criticamente de $\Delta t$ ser constante. Quando $\Delta t = \Delta t(r, v)$, os termos de derivada do passo $\nabla (\Delta t)$ contaminam o Jacobiano, e o mapeamento deixa de ser canonico.
- **Teste que isolou a causa:** Comparou-se a evolucao de $E(t)$ para a mesma orbita usando Verlet com passo fixo ($\Delta t = 0.02$) e Verlet com passo variavel com media igual a $0.02$. O erro de energia com passo variavel acumulou deriva de $10^{-4}$ em apenas $50$ orbitas, enquanto com passo fixo oscilou abaixo de $10^{-7}$.
- **Correcao:** Separacao de dominios no projeto:
  1. No estudo de longo prazo conservativo (Parte 2), o Verlet opera estritamente com passo fixo;
  2. O passo adaptativo foi associado ao RK45 / RK4 no sistema com amortecimento viscoso ($\gamma > 0$) das Partes 3 a 6, onde o sistema ja e fisicamente dissipativo ($dE/dt = -\gamma |v|^2$) e o balanco energetico e checado pelo residuo $\mathcal{R}(t)$.
- **Principio geral:** Integradores simpleticos perdem suas propriedades geometricas sob controle de passo variavel convencional. Adaptar passo preservando simpleticidade exige transformacoes de Poincare ou tempo ficticio de Sundman/Levi-Civita.

---

## Falha 4 - Inconsistencia na comparacao de grades e tratamento de E/S na analise de bacias

- **Sintoma:** Execucao do script CLI `parte5_bacias.py` com refinamento temporal em $\Delta t$ resultava em excecao `FileNotFoundError` ao salvar imagens em subpastas inexistentes, seguido de `ValueError: a grade fina precisa ter o dobro da resolucao em cada eixo` na funcao `compare_basins`.
- **Diagnostico:**
  1. A funcao `run_basin_simulation` gravava arquivos de imagem assumindo que a pasta de destino ja havia sido criada em disco, gerando erro quando caminhos como `imagens_bacias/parte5` eram passados.
  2. A rotina `compare_basins` continha uma assercao rigida exigindo que a grade fina tivesse o dobro da resolucao da grossa (`fine.shape[0] == 2 * coarse.shape[0]`), sendo adequada apenas para refinamento espacial. No entanto, o protocolo da Secao 5.1 exige a comparacao de quatro bacias geradas na **mesma malha espacial** ($n \times n$) variando-se apenas o passo temporal ($\Delta t, \Delta t/2, \Delta t/4, \Delta t/8$).
- **Teste que isolou a causa:** Execucao de `python parte5_bacias.py --resolution 16 --dts "0.05,0.025"` em terminal limpo. O erro de E/S ocorreu na linha de `save_basin_image`, e a falha de dimensao ocorreu na chamada de `compare_basins`.
- **Correcao:**
  1. Inclusao de `output_directory.mkdir(parents=True, exist_ok=True)` no inicio de `run_basin_simulation`;
  2. Generalizacao de `compare_basins` para suportar tanto malhas de dimensoes identicas (convergencia temporal em $\Delta t$ com comparacao pixel a pixel direta) quanto malhas com subamostragem $2 \times$ (convergencia espacial).
- **Principio geral:** Em software cientifico, rotinas de diagnostico de convergencia devem desacoplar explicitamente o refinamento temporal (mesmo $n$, variando $\Delta t$) do refinamento espacial (mesmo $\Delta t$, variando $n$). Toda operacao de gravacao de arquivos deve assegurar a existencia da arvore de diretorios.
