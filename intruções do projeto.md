# Bacias de Atração Gravitacionais

## Trabalho de Cálculo Numérico

**Faculdade de Tecnologia — UERJ** **Prof. Dr. Vahid Nikoofard**

---

## 1\. O que você vai construir

Imagine uma região do espaço com algumas massas pesadas fixas — chame-as de **atratores**. Cada atrator recebe uma cor: vermelho, azul, verde, amarelo. Agora solte uma partícula minúscula (a **partícula de teste**) a partir de um ponto qualquer, em repouso. Ela é puxada pela gravidade de todas as massas ao mesmo tempo, descreve uma trajetória complicada e, eventualmente, cai sobre uma delas.

Pinte o ponto de partida com a cor do atrator que a capturou.

Repita para um milhão de pontos de partida.

> **Dimensões: preste atenção nesta distinção.** A **dinâmica é tridimensional** — as massas ocupam posições em $\\mathbb R^3$, o vetor de estado tem 6 componentes e as trajetórias são curvas no espaço. Mas a **imagem final é um corte bidimensional**: você escolhe um plano, amostra uma grade de pontos de partida *nesse plano*, e pinta cada um pela cor do destino. A partícula sai do plano assim que começa a se mover; o plano é só o berçário, não a arena.  
>   
> Um volume $1000^3$ exigiria $10^9$ trajetórias — mil vezes o custo de um corte — e não haveria como visualizar o resultado sem entrar em renderização volumétrica, que não é o objetivo da disciplina. O corte 2D também é o que torna diretas as métricas da Parte 5 (fração de área de cada bacia) e a dimensão fractal do item T2.

A imagem que emerge não é o que a intuição sugere. Em vez de quatro regiões limpas, cada uma em volta de seu atrator, você vai encontrar fronteiras infinitamente rendilhadas, onde pontos vizinhos terminam em atratores diferentes. Essa imagem é o produto visual do trabalho.

Mas o produto **real** é outro: é a sua capacidade de responder, com números na mão, à pergunta *"como eu sei que essa imagem está certa?"*

Essa é a pergunta que organiza o trabalho inteiro. Guarde-a.

---

## 2\. Por que este problema

Você já resolveu muitas EDOs em exercícios de lista, onde a resposta certa estava no fim do livro. Este trabalho é diferente em três aspectos que aparecem em todo projeto de engenharia real:

1. **Não existe resposta no fim do livro.** O único jeito de saber se o seu resultado está correto é você mesmo construir os testes de verificação.  
2. **Métodos numéricos podem falhar silenciosamente.** O código roda, não dá erro, produz um gráfico bonito — e está errado. Você vai aprender a detectar isso.  
3. **Existe um limite fundamental de previsibilidade.** Este sistema é caótico. Em uma parte do trabalho você vai *calcular* por quanto tempo a sua própria simulação faz sentido, e descobrir que depois desse tempo nenhum computador do mundo ajuda.

Esse terceiro ponto é o mais importante. A maioria dos engenheiros nunca é confrontada com a ideia de que existem perguntas para as quais mais poder computacional não é a resposta.

---

## 3\. Sobre o uso de Inteligência Artificial

**Você pode e deve usar IA neste trabalho.** Não faz sentido proibir uma ferramenta que vocês vão usar profissionalmente pelo resto da carreira. Mas o trabalho foi projetado de modo que a IA resolva a parte fácil e você resolva a parte difícil.

### O que a IA faz bem aqui

- Escrever a implementação de Runge-Kutta 4 ou de Velocity Verlet.  
- Vetorizar um laço `for` com `numpy`.  
- Explicar um erro de sintaxe ou um `NaN` inesperado.  
- Sugerir como estruturar um código em funções.  
- Traduzir uma equação do roteiro em código.

### O que a IA não faz por você

- Rodar o seu código e olhar para o resultado.  
- Descobrir **por que** a sua ordem de convergência observada deu 2 quando deveria dar 4\.  
- Decidir se o valor de $\\tilde\\varepsilon$ que você escolheu é defensável fisicamente.  
- Medir o expoente de Lyapunov *da sua configuração específica de massas*.  
- Explicar, na defesa oral, o que acontece com a imagem se eu mudar um parâmetro.

### Regras

1. **Regra do "eu explico".** Você não entrega nenhuma linha de código que não consiga explicar linha a linha, na defesa oral, sem consultar nada. Se a IA escreveu algo que você não entende, ou você estuda até entender, ou você reescreve de um jeito que entenda.  
2. **Apêndice de prompts.** O relatório deve ter um apêndice listando os principais prompts usados e, para cada um, uma frase sobre o que você teve que corrigir na resposta. Isso vale nota. Um apêndice honesto dizendo "a IA errou o sinal da força e eu só percebi quando a partícula começou a ser repelida" vale mais que um apêndice vazio.  
3. **Verificação é sua.** Todo código gerado por IA entra no trabalho **depois** de passar pelos testes de verificação da Parte 2\. Sem exceção.  
4. **A defesa é individual.** Cada integrante do grupo responde sobre qualquer parte do trabalho.

O objetivo não é testar se você sabe digitar Runge-Kutta de cabeça. É testar se você sabe julgar se um resultado numérico é confiável. Essa é a competência que sobrevive a qualquer ferramenta nova.

---

## 4\. A física, do zero

Esta seção assume apenas Cálculo e Física I. Leia com calma e refaça as contas no papel.

### 4.1 A lei da gravitação de Newton

Duas massas se atraem. Se uma massa $M$ está parada na posição $\\mathbf R$ e uma massa $m$ está na posição $\\mathbf r$, a força que $M$ exerce sobre $m$ tem:

- **módulo** $;\\dfrac{GMm}{d^2};$ onde $d \= |\\mathbf r \- \\mathbf R|$ é a distância entre elas;  
- **direção** ao longo da reta que liga as duas;  
- **sentido** de $\\mathbf r$ para $\\mathbf R$ (atração).

Para escrever isso como vetor, precisamos do vetor unitário que aponta de $M$ para $m$:

$$\\hat{\\mathbf u} \= \\frac{\\mathbf r \- \\mathbf R}{|\\mathbf r \- \\mathbf R|}$$

A força aponta no sentido **oposto**, logo:

$$\\mathbf F \= \-\\frac{GMm}{|\\mathbf r-\\mathbf R|^2},\\hat{\\mathbf u} \= \-GMm,\\frac{\\mathbf r-\\mathbf R}{|\\mathbf r-\\mathbf R|^3} \\tag{1}$$

Note o expoente **3** no denominador da forma final: dois vêm da lei do inverso do quadrado e um vem da normalização do vetor unitário. Esse é o erro de digitação mais comum do trabalho inteiro. Confira sempre.

A constante é $G \= 6{,}674\\times10^{-11}\\ \\mathrm{m^3,kg^{-1},s^{-2}}$ — mas, como você verá na Seção 4.5, vamos nos livrar dela.

### 4.2 Segunda lei de Newton e a ideia de partícula de teste

Com $N$ massas $M\_1,\\dots,M\_N$ fixas nas posições $\\mathbf R\_1,\\dots,\\mathbf R\_N$, a força total sobre a partícula é a soma vetorial das contribuições (princípio da superposição). A segunda lei diz $\\mathbf F\_{\\text{total}} \= m\\ddot{\\mathbf r}$, e portanto:

$$m,\\ddot{\\mathbf r} \= \-\\sum\_{i=1}^{N} GM\_i m,\\frac{\\mathbf r-\\mathbf R\_i}{|\\mathbf r-\\mathbf R\_i|^3}$$

**A massa $m$ cancela dos dois lados:**

$$\\ddot{\\mathbf r} \= \-\\sum\_{i=1}^{N} GM\_i,\\frac{\\mathbf r-\\mathbf R\_i}{|\\mathbf r-\\mathbf R\_i|^3} \\tag{2}$$

Esse cancelamento é profundo — é o mesmo fato que faz uma pena e um martelo caírem juntos no vácuo. Para nós tem uma consequência prática: **a trajetória não depende da massa da partícula de teste.** Por isso ela é chamada de partícula de teste: é leve o suficiente para não perturbar as massas pesadas, e sua própria massa é irrelevante para o resultado.

Chamamos a expressão à direita de **campo gravitacional** $\\mathbf g(\\mathbf r)$. É força por unidade de massa.

> **Pergunta para o relatório:** por que as massas pesadas podem ser consideradas fixas? Que hipótese sobre $m$ isso exige? O que mudaria se $m$ fosse comparável a $M\_i$?

### 4.3 Primeiro problema: a singularidade

Olhe para a equação (2) quando $\\mathbf r \\to \\mathbf R\_i$. O denominador vai a zero e a aceleração vai a infinito. Numericamente isso é um desastre: perto do atrator o passo de tempo necessário vai a zero, o código trava ou produz `inf` e `NaN`.

Fisicamente, o problema é que estamos tratando um corpo extenso (um planeta, uma estrela) como um ponto matemático. A solução padrão em astrofísica computacional é o **amaciamento de Plummer** (*Plummer softening*): substituímos

$$|\\mathbf r-\\mathbf R\_i|^2 ;\\longrightarrow; |\\mathbf r-\\mathbf R\_i|^2 \+ \\varepsilon^2$$

onde $\\varepsilon$ é um comprimento pequeno comparado às distâncias típicas. O campo fica:

$$\\mathbf g(\\mathbf r) \= \-\\sum\_{i=1}^{N} GM\_i,\\frac{\\mathbf r-\\mathbf R\_i}{\\big(|\\mathbf r-\\mathbf R\_i|^2+\\varepsilon^2\\big)^{3/2}} \\tag{3}$$

Longe da massa ($d \\gg \\varepsilon$) isso é indistinguível de (2). Perto, a força para de crescer, atinge um máximo e volta a zero no centro — que é exatamente o que acontece com a gravidade dentro de um corpo esférico real.

> **Atenção — isto é importante.** $\\varepsilon$ **não é um parâmetro numérico.** É uma escolha de modelo. Refinar o passo de tempo não faz $\\varepsilon$ ir a zero: ele continua lá, mudando a física. Você vai ter que escolher um valor e **justificá-lo**, e vai ter que mostrar como a imagem final muda quando ele muda. Saber distinguir um parâmetro numérico (que deve convergir) de um parâmetro físico (que deve ser justificado) é uma das coisas mais úteis que você leva deste trabalho.

### 4.4 Segundo problema: a partícula não cai

Aqui está uma armadilha que eu quero que vocês encontrem sozinhos, mas que vou explicar em seguida para que não percam uma semana nela.

Se você programar a equação (3) exatamente como está e soltar a partícula do repouso, na esmagadora maioria dos casos ela **nunca vai cair em lugar nenhum**. Ela vai passar raspando por um atrator, ganhar velocidade, ser arremessada para longe, voltar, passar raspando por outro, e assim indefinidamente. Ou vai escapar para o infinito.

A razão é a conservação de energia. Sem dissipação, a partícula que cai de uma altura chega ao ponto mais baixo com energia cinética suficiente para subir de volta à mesma altura. Ela oscila para sempre. Órbitas que realmente terminam em colisão formam um conjunto de medida nula — probabilidade zero de acertar por acaso.

Para que a pergunta *"em qual massa ela cai?"* tenha resposta, precisamos de **dissipação**. Modelamos a partícula como se movendo através de um meio tênue que oferece resistência proporcional à velocidade:

$$\\mathbf F\_{\\text{arrasto}} \= \-\\gamma m \\mathbf v$$

Esse é o modelo de arrasto de Stokes, o mesmo de um corpo lento em fluido viscoso. A equação final do movimento é:

$$\\boxed{;\\ddot{\\mathbf r} \= \-\\sum\_{i=1}^{N} GM\_i,\\frac{\\mathbf r-\\mathbf R\_i}{\\big(|\\mathbf r-\\mathbf R\_i|^2+\\varepsilon^2\\big)^{3/2}} ;-; \\gamma,\\dot{\\mathbf r};} \\tag{4}$$

Com $\\gamma \> 0$ a energia decresce monotonicamente, a partícula perde a capacidade de escapar dos poços e as massas viram **atratores** de verdade. Daí o nome "bacia de atração": o conjunto de todos os pontos de partida cujo destino é um mesmo atrator.

> **Lição geral:** o modelo teve que ser projetado para que a pergunta tivesse resposta. Isso não é trapaça — é modelagem. Todo projeto de engenharia envolve decidir o que incluir e o que desprezar, e essa decisão é do engenheiro, não do solver.

### 4.5 Adimensionalização — faça isto antes de programar

A equação (4) tem cinco constantes ($G$, as $M\_i$, as distâncias, $\\gamma$, $\\varepsilon$), com unidades que podem estar em quilogramas e anos-luz. Isso é um convite a erros de unidade e a números como $10^{30}$ dentro do código.

Escolha duas escalas de referência:

- $L$ \= distância típica entre as massas;  
- $M\_{\\text{tot}} \= \\sum\_i M\_i$ \= massa total.

Delas sai automaticamente uma escala de tempo, o **tempo dinâmico**:

$$T\_c \= \\sqrt{\\frac{L^3}{G M\_{\\text{tot}}}} \\tag{5}$$

*(Verifique a dimensão: $\[G\] \= \\mathrm{m^3kg^{-1}s^{-2}}$, então $L^3/(GM)$ tem dimensão de $\\mathrm{s^2}$. Confere.)*

Defina as variáveis adimensionais $\\tilde{\\mathbf r} \= \\mathbf r/L$, $\\tilde t \= t/T\_c$, $\\tilde M\_i \= M\_i/M\_{\\text{tot}}$. Substituindo em (4) e usando a regra da cadeia (**faça esta conta no papel — ela é entregável**), você obtém:

$$\\frac{d^2\\tilde{\\mathbf r}}{d\\tilde t^2} \= \-\\sum\_{i=1}^{N} \\tilde M\_i,\\frac{\\tilde{\\mathbf r}-\\tilde{\\mathbf R}\_i}{\\big(|\\tilde{\\mathbf r}-\\tilde{\\mathbf R}\_i|^2+\\tilde\\varepsilon^2\\big)^{3/2}} \- \\tilde\\gamma,\\frac{d\\tilde{\\mathbf r}}{d\\tilde t} \\tag{6}$$

com apenas **dois** parâmetros livres:

$$\\boxed{;\\tilde\\gamma \= \\gamma,T\_c \= \\gamma\\sqrt{\\frac{L^3}{GM\_{\\text{tot}}}},\\qquad \\tilde\\varepsilon \= \\frac{\\varepsilon}{L};}$$

O que você ganhou:

- $G$ desapareceu; $\\sum\_i\\tilde M\_i \= 1$; todas as coordenadas são da ordem de 1\.  
- O estudo de parâmetros passou de 5 dimensões para 2\.  
- E o mais útil: **agora você sabe qual passo de tempo é razoável.** Um passo de $10^{-3}$ significa um milésimo do tempo dinâmico. Sem adimensionalizar, "$\\Delta t \= 0{,}001$" não quer dizer absolutamente nada.

Daqui em diante, omitimos os tis. Todas as equações estão em unidades adimensionais.

### 4.6 Redução a um sistema de primeira ordem

A equação (6) é de segunda ordem, e nossos métodos numéricos são para sistemas de primeira ordem. O truque padrão: trate a velocidade como incógnita independente. Com o vetor de estado $\\mathbf y \= (\\mathbf r, \\mathbf v) \\in \\mathbb R^6$:

$$\\dot{\\mathbf y} \= \\mathbf f(\\mathbf y) \= \\begin{pmatrix} \\mathbf v \\\[6pt\] \\mathbf g(\\mathbf r) \- \\gamma\\mathbf v \\end{pmatrix}, \\qquad \\mathbf g(\\mathbf r) \= \-\\sum\_{i=1}^{N} M\_i\\frac{\\mathbf r-\\mathbf R\_i}{(|\\mathbf r-\\mathbf R\_i|^2+\\varepsilon^2)^{3/2}} \\tag{7}$$

Em 3D isso é um sistema de **6 EDOs acopladas**. A condição inicial do nosso problema é $\\mathbf r(0) \= \\mathbf r\_0$ (o ponto de partida, que varre a tela) e $\\mathbf v(0) \= \\mathbf 0$ (repouso).

### 4.7 Energia: a sua principal ferramenta de diagnóstico

Defina a energia **por unidade de massa** da partícula de teste:

$$E(t) \= \\underbrace{\\tfrac12|\\mathbf v|^2}*{\\text{cinética}} ;-; \\underbrace{\\sum*{i=1}^{N}\\frac{M\_i}{\\sqrt{|\\mathbf r-\\mathbf R\_i|^2+\\varepsilon^2}}}\_{\\text{potencial };=;\\Phi(\\mathbf r)} \\tag{8}$$

*(Verifique você mesmo que $\\mathbf g \= \-\\nabla\\Phi$ — derive o potencial em relação a $x$ e compare com a componente $x$ de (7). É um bom exercício de gradiente e confirma que o expoente $3/2$ está certo.)*

Agora derive $E$ ao longo de uma solução, usando a regra da cadeia:

$$\\frac{dE}{dt} \= \\mathbf v\\cdot\\dot{\\mathbf v} \+ \\nabla\\Phi\\cdot\\dot{\\mathbf r} \= \\mathbf v\\cdot(\\mathbf g \- \\gamma\\mathbf v) \+ (-\\mathbf g)\\cdot\\mathbf v \= \-\\gamma|\\mathbf v|^2 \\tag{9}$$

Os termos gravitacionais se cancelam **exatamente**. Sobra só a dissipação — e note que $-\\gamma|\\mathbf v|^2 \\le 0$ sempre: a energia nunca aumenta. Integrando de $0$ a $t$:

$$\\boxed{;\\underbrace{E(t) \- E(0) \+ \\gamma\\int\_0^{t}|\\mathbf v(s)|^2,ds ;=; 0}\_{\\text{identidade de dissipação}};} \\tag{10}$$

**Esta equação é o coração metodológico do trabalho.** Ela é uma identidade exata: qualquer desvio é erro numérico puro. E repare no que ela contém:

- $E(t)$ e $E(0)$ vêm da solução da EDO;  
- a integral $\\int|\\mathbf v|^2 ds$ tem que ser calculada por **quadratura numérica** sobre a trajetória.

Ou seja, o resíduo

$$\\mathcal R(t) \= \\Big|E(t)-E(0)+\\gamma\\int\_0^t|\\mathbf v|^2 ds\\Big|$$

mede simultaneamente a qualidade do integrador de EDO **e** da regra de integração usada. Você vai reportar $\\mathcal R$ em toda simulação que fizer. É o seu detector de bugs: se $\\mathcal R$ é grande, alguma coisa está errada, mesmo que o gráfico esteja bonito.

Quando $\\gamma \= 0$, (10) vira simplesmente $E(t) \= E(0)$ — conservação de energia. Esse caso é usado na Parte 2\.

### 4.8 Quando a partícula escapou?

Se a partícula está muito longe de todas as massas, o conjunto se comporta aproximadamente como uma única massa $M\_{\\text{tot}} \= 1$ na origem. Nessa aproximação, a energia é $E \\approx \\frac12|\\mathbf v|^2 \- 1/|\\mathbf r|$, e a partícula escapa se $E \> 0$, isto é:

$$\\boxed{;\\tfrac12|\\mathbf v|^2 \> \\frac{1}{|\\mathbf r|} \\quad\\text{e}\\quad \\mathbf r\\cdot\\mathbf v \> 0 \\quad\\text{e}\\quad |\\mathbf r| \> R\_{\\text{esc}};} \\tag{11}$$

As três condições são necessárias: energia positiva, movimento de afastamento, e estar longe o bastante para a aproximação monopolar valer (use $R\_{\\text{esc}} \\approx 20$).

Sem este teste o seu código vai passar a maior parte do tempo integrando partículas que já foram embora. Pontos que escapam recebem a cor **preta** na imagem final.

---

## 5\. Roteiro de trabalho

### Parte 0 — Análise no papel (antes de qualquer código)

Entregue manuscrito ou digitado, sem código:

**0.1** Dedução da forma vetorial (1) a partir do enunciado escalar da lei da gravitação. Mostre de onde vem o expoente 3\.

**0.2** Verificação de que $\\mathbf g \= \-\\nabla\\Phi$ com o potencial amaciado de (8).

**0.3** A adimensionalização completa: substitua $\\mathbf r \= L\\tilde{\\mathbf r}$ e $t \= T\_c\\tilde t$ em (4) e obtenha (6), mostrando que $\\tilde\\gamma \= \\gamma T\_c$ e $\\tilde\\varepsilon \= \\varepsilon/L$.

**0.4** Dedução da identidade de dissipação (10) a partir de (9).

**0.5** Dedução do critério de escape (11).

**0.6** Estimativa de ordem de grandeza: para a sua configuração de massas (ver Seção 8), calcule $T\_c$ e estime quanto tempo adimensional uma partícula solta a uma distância 2 leva para atingir o centro em queda livre radial. Use isso para justificar a sua escolha de $t\_{\\max}$.

> Sem a Parte 0 aprovada, o grupo não avança. Ela leva umas 3 horas e economiza duas semanas.

---

### Parte 1 — Verificação: o problema de Kepler

Você não tem solução exata para $N$ massas. Mas tem para **uma**, e é assim que se verifica um código: reduza ao caso conhecido.

Configure $N=1$, $M\_1 \= 1$ na origem, $\\gamma \= 0$, $\\varepsilon \= 0$.

**1.1 Órbita circular.** Para a partícula descrever um círculo de raio $a$, a aceleração centrípeta $v^2/a$ deve igualar a gravitacional $1/a^2$, logo $v \= 1/\\sqrt a$. Use:

$$\\mathbf r\_0 \= (a, 0, 0), \\qquad \\mathbf v\_0 \= (0,\\ 1/\\sqrt a,\\ 0)$$

A solução exata é um círculo de raio constante $a$ com período $T \= 2\\pi a^{3/2}$. Portanto o erro é **diretamente mensurável**:

$$\\text{erro radial} \= \\max\_{t\\in\[0,T\]}\\big|,|\\mathbf r(t)| \- a,\\big|$$

**1.2 Órbita elíptica.** Dê uma velocidade inicial diferente de $1/\\sqrt a$ e obtenha uma elipse. Duas quantidades são conservadas exatamente:

$$E \= \\tfrac12|\\mathbf v|^2 \- \\frac{1}{|\\mathbf r|}, \\qquad \\mathbf L \= \\mathbf r\\times\\mathbf v$$

e delas saem o semieixo maior $a \= \-1/(2E)$ e a excentricidade $e \= \\sqrt{1+2E|\\mathbf L|^2}$.

**1.3 A elipse que não deveria girar.** Existe uma terceira quantidade conservada, menos conhecida, o vetor de Laplace–Runge–Lenz:

$$\\mathbf A \= \\mathbf v\\times\\mathbf L \- \\frac{\\mathbf r}{|\\mathbf r|} \\tag{12}$$

Ele aponta na direção do periélio (o ponto de maior aproximação). Como é conservado, **uma elipse newtoniana não precessa**: o eixo maior fica parado para sempre. Se na sua simulação a elipse gira lentamente, essa precessão é **erro numérico puro se manifestando como um efeito físico falso**. Meça a taxa de rotação de $\\mathbf A$ e mostre que ela vai a zero quando $\\Delta t$ diminui.

**1.4 O teste do plano invariante — verificação de graça, e exata.**

Antes de ir para a tabela de ordem, faça este teste. Ele custa cinco minutos e pega uma classe inteira de erros.

Monte uma configuração em que **todas as massas sejam coplanares** — por exemplo, todas com $z=0$. Solte uma partícula a partir de um ponto desse mesmo plano, em repouso. Afirmação: a partícula **nunca sai do plano**.

A demonstração é curta e você deve escrevê-la no relatório. A aceleração em (7) é uma combinação linear dos vetores $\\mathbf r \- \\mathbf R\_i$. Se $\\mathbf r$ e todos os $\\mathbf R\_i$ têm $z=0$, então todos esses vetores têm componente $z$ nula, logo $g\_z \= 0$. Como $v\_z(0)=0$ e $\\dot v\_z \= g\_z \- \\gamma v\_z \= 0$, segue que $v\_z(t)\\equiv 0$ e $z(t)\\equiv 0$ para sempre.

**O teste:** integre por um tempo longo e verifique que $\\max\_t |z(t)|$ e $\\max\_t |v\_z(t)|$ permanecem na ordem do zero de máquina ($\\sim 10^{-16}$). Se $z$ crescer, há erro de sinal, de indexação ou de broadcasting no seu campo gravitacional, ou no integrador. Diferentemente dos testes de Kepler, este não tem erro de truncamento nenhum: o resultado correto é zero exato, então qualquer desvio acima do arredondamento é bug.

> **Consequência para o seu trabalho.** Esse mesmo fato é uma armadilha. Se as massas do seu grupo forem coplanares **e** o corte da imagem for o plano delas, a sua "simulação 3D" é, na prática, uma simulação 2D disfarçada: nenhuma partícula jamais deixa o plano. Por isso o gerador de configurações (Seção 8\) garante que as massas **não** sejam coplanares. Verifique isso na sua configuração e comente no relatório.

**1.5 Tabela de ordem observada.** Implemente Euler explícito, RK2 (Heun), RK4 e Velocity Verlet (Seção 5.2). Para cada um, com passo $\\Delta t$ e $\\Delta t/2$, compare o estado após uma órbita completa com a solução exata e calcule:

$$p\_{\\text{obs}} \\approx \\log\_2 \\frac{|\\mathbf y\_{\\Delta t}(T)-\\mathbf y\_{\\text{exata}}(T)|}{|\\mathbf y\_{\\Delta t/2}(T)-\\mathbf y\_{\\text{exata}}(T)|}$$

Esperado: $1, 2, 4, 2$. Faça a tabela para $e \= 0$, $e \= 0{,}5$ e $e \= 0{,}9$.

**1.6 O que dá errado em alta excentricidade.** Você vai observar que, para $e \= 0{,}9$, **todos** os métodos pioram. O motivo: o tempo dinâmico local é $t\_{\\text{din}}\\sim r^{3/2}$, e entre o afélio $r=a(1+e)$ e o periélio $r=a(1-e)$ ele varia por um fator

$$\\left(\\frac{1+e}{1-e}\\right)^{3/2} \\approx 83 \\quad\\text{para } e=0{,}9$$

Um passo fixo dimensionado para o afélio não resolve a passagem pelo periélio. Documente isso com gráficos — é a motivação concreta para a Parte 3\.

---

### Parte 2 — Por que um método de ordem 2 pode ser melhor que um de ordem 4

Aqui está o resultado mais contraintuitivo do trabalho.

**O método de Velocity Verlet.** Três linhas, uma única avaliação de força por passo:

$$\\mathbf v\_{n+1/2} \= \\mathbf v\_n \+ \\tfrac{\\Delta t}{2},\\mathbf g(\\mathbf r\_n)$$ $$\\mathbf r\_{n+1} \= \\mathbf r\_n \+ \\Delta t,\\mathbf v\_{n+1/2} \\tag{13}$$ $$\\mathbf v\_{n+1} \= \\mathbf v\_{n+1/2} \+ \\tfrac{\\Delta t}{2},\\mathbf g(\\mathbf r\_{n+1})$$

É de **ordem 2** — metade da ordem de RK4. E mesmo assim:

**2.1 Deriva secular de energia.** Com $\\gamma \= 0$, integre uma órbita circular por $10^4$ períodos com RK4 e com Verlet e plote $E(t)$.

> **Compare a custo computacional igual, não a passo igual.** RK4 usa 4 avaliações de força por passo, Verlet usa 1\. Portanto RK4 deve rodar com $\\Delta t\_{\\text{RK4}} \= 4,\\Delta t\_{\\text{Verlet}}$ para gastar o mesmo. Comparar a passo igual favorece RK4 injustamente por um fator 4, e é o erro mais comum nesta parte.

O que você vai ver: RK4 apresenta uma **deriva sistemática**, com $|\\Delta E|$ crescendo aproximadamente de forma linear no tempo. Verlet apresenta uma **oscilação de amplitude constante**, que não cresce nunca — nem depois de um milhão de períodos.

A explicação, em uma frase: Verlet é um **integrador simplético**, o que significa que ele resolve exatamente um problema ligeiramente diferente do nosso, cuja energia $\\tilde E \= E \+ O(\\Delta t^2)$ é conservada de verdade. Como $\\tilde E$ é constante e difere de $E$ por uma quantidade pequena e limitada, $E$ fica preso numa faixa estreita para sempre.

**2.2 Reversibilidade temporal.** Integre por um tempo $T$, inverta o sinal da velocidade, e integre de novo por $T$. Verlet volta ao ponto inicial a menos de erro de arredondamento (ele é exatamente reversível). RK4 não volta. Meça $|\\mathbf r\_{\\text{volta}}-\\mathbf r\_0|$ para os dois, em função de $\\Delta t$.

**2.3 A conclusão que você deve escrever com suas palavras.** Ordem de precisão e correção qualitativa são propriedades **independentes**. Um método de ordem alta pode destruir sistematicamente uma estrutura que a solução exata possui. Para integração curta, RK4 ganha. Para integração longa de sistemas conservativos, Verlet ganha. **Determine numericamente o tempo de cruzamento** para a sua configuração.

---

### Parte 3 — Passo adaptativo e detecção de eventos

**3.1 Controle de passo.** Implemente e compare pelo menos dois critérios:

$$\\Delta t\_n \= \\eta,\\min\_i \\frac{|\\mathbf r\_n \- \\mathbf R\_i|^{3/2}}{\\sqrt{M\_i}} \\qquad\\text{(tempo dinâmico local)}$$

$$\\Delta t\_n \= \\eta,\\frac{|\\mathbf v\_n|}{|\\mathbf a\_n|} \\qquad\\text{(cinemático)}$$

com $\\eta \\approx 0{,}01$–$0{,}05$. Opcionalmente, compare com o controle por erro embutido do RK45 (`scipy.integrate.solve_ivp`).

> **Armadilha honesta:** passo adaptativo **destrói** a propriedade simplética do Verlet. Se você juntar as Partes 2 e 3 sem pensar, vai perder a conservação de energia que acabou de demonstrar. Isso não é um bug — é uma incompatibilidade real entre as duas técnicas. Documente-a no diário de falhas e tome uma decisão justificada: ou Verlet com passo fixo pequeno, ou RK45 adaptativo aceitando a deriva (defensável aqui, porque com $\\gamma\>0$ a energia deve mesmo decrescer).

**3.2 Detecção de captura — root-finding dentro da EDO.** A partícula é capturada por $\\mathbf R\_i$ quando cruza a esfera de raio $R\_{\\text{cap}}$. Defina:

$$g\_i(t) \= |\\mathbf r(t)-\\mathbf R\_i|^2 \- R\_{\\text{cap}}^2$$

Testar $g\_i \< 0$ só nos pontos da malha **não basta**: em uma passagem rápida, um passo pode pular por cima da esfera inteira, e a partícula segue viagem, sendo capturada mais tarde por outro atrator. A cor do pixel sai errada.

O procedimento correto:

1. A cada passo, teste se $g\_i(t\_n)$ e $g\_i(t\_{n+1})$ têm sinais opostos, para algum $i$.  
2. Se sim, houve cruzamento dentro do passo. Localize o instante exato resolvendo $g\_i(t^\\ast)=0$ por **bissecção** ou **método de Brent** sobre a trajetória interpolada no intervalo $\[t\_n, t\_{n+1}\]$.  
3. Pare a integração em $t^\\ast$ e registre o atrator $i$.

**Entregável decisivo:** gere duas imagens de bacias, uma com teste ingênuo e outra com detecção de eventos, e **quantifique a fração de pixels diferentes**. Tipicamente são alguns por cento, concentrados exatamente na fronteira — a região mais delicada da figura. Este é o momento em que root-finding e integração de EDO aparecem acoplados, e é um resultado que só existe porque você rodou o código.

**3.3 Três formas de terminar.** Toda trajetória acaba de uma das três maneiras, e as três precisam estar implementadas:

| Condição | Cor do pixel |
| :---- | :---- |
| Capturada pelo atrator $i$ | cor $i$ |
| Escapou (critério 11\) | preto |
| $t \> t\_{\\max}$ sem resolver | cinza |

A fração de cinza é um diagnóstico: se for grande, o seu $\\tilde\\gamma$ está pequeno demais ou o $t\_{\\max}$ curto demais. Reporte-a sempre.

---

### Parte 4 — Caos: medindo o limite da própria simulação

**4.1 Expoente de Lyapunov.** Solte duas partículas de pontos separados por $|\\boldsymbol\\delta\_0| \= 10^{-10}$ e acompanhe a separação. Em região caótica ela cresce exponencialmente:

$$|\\boldsymbol\\delta(t)| \\approx |\\boldsymbol\\delta\_0|,e^{\\lambda t}$$

Você não pode simplesmente ajustar uma reta a $\\ln|\\boldsymbol\\delta|$, porque a separação satura no tamanho do sistema. Use o **algoritmo de Benettin**: a cada intervalo $\\tau$, reescale $\\boldsymbol\\delta$ de volta ao tamanho original mantendo a direção, e acumule:

$$\\lambda \\approx \\frac{1}{K\\tau}\\sum\_{k=1}^{K} \\ln\\frac{|\\boldsymbol\\delta(t\_k)|}{|\\boldsymbol\\delta\_0|} \\tag{14}$$

Faça isso para um ponto no interior de uma bacia (espere $\\lambda \\le 0$) e para um ponto perto da fronteira (espere $\\lambda \> 0$). Rode com $\\gamma \= 0$ nesta parte — o atrito mascara o caos.

**4.2 Horizonte de previsibilidade — a conta mais importante do trabalho.** Se o erro inicial é $\\delta\_0$ e cresce como $e^{\\lambda t}$, o tempo até ele atingir uma tolerância $\\Delta\_{\\text{tol}}$ é:

$$t\_{\\text{prev}} \= \\frac{1}{\\lambda}\\ln\\frac{\\Delta\_{\\text{tol}}}{\\delta\_0} \\tag{15}$$

Avalie com $\\delta\_0 \= \\epsilon\_{\\text{máquina}} \\approx 2\\times10^{-16}$ (o erro de arredondamento em ponto flutuante de dupla precisão, que você **não pode evitar**) e $\\Delta\_{\\text{tol}} \= 0{,}1$. Com um $\\lambda$ típico da ordem de $1$, dá $t\_{\\text{prev}} \\approx 36$ tempos dinâmicos.

Agora escreva no relatório as três consequências:

1. Além de $t\_{\\text{prev}}$, a sua trajetória calculada **não é** a trajetória do problema. É uma trajetória plausível do sistema, mas não a que parte da sua condição inicial.  
2. Um integrador de ordem maior **não resolve isso**, porque o problema não é o truncamento, é a amplificação do arredondamento.  
3. Passar para precisão quádrupla ($\\delta\_0\\sim10^{-32}$) **dobra** $t\_{\\text{prev}}$ e nada mais, porque a dependência é logarítmica. Ganhar um fator 10 no horizonte exigiria $\\delta\_0 \\sim 10^{-160}$.

**4.3 Então por que a imagem das bacias faz sentido?** Esta é a pergunta que fecha o trabalho, e vocês vão respondê-la com dados na Parte 5\. A resposta curta: trajetórias individuais são imprevisíveis, mas **estatísticas** são robustas. A área de cada bacia converge; a cor de um pixel específico na fronteira, não.

---

### Parte 5 — As bacias, e o que significa "convergiu"

Escolha um plano (por exemplo $z=0$), amostre uma grade $n\\times n$ de pontos de partida, solte cada partícula do repouso e pinte pela cor do destino.

**5.1 Convergência da imagem — dois números diferentes.** Gere a imagem com $\\Delta t$, $\\Delta t/2$, $\\Delta t/4$, $\\Delta t/8$. Para cada par consecutivo calcule:

$$\\Pi \= \\frac{\#{\\text{pixels que mudaram de cor}}}{n^2} \\qquad\\text{(convergência pontual)}$$

$$\\Delta\\mu \= \\max\_i \\big|\\mu\_i(\\Delta t) \- \\mu\_i(\\Delta t/2)\\big| \\qquad\\text{(convergência em medida)}$$

onde $\\mu\_i$ é a fração de área da bacia $i$.

**O que você vai encontrar:** $\\Delta\\mu$ cai limpo, na ordem do integrador. $\\Pi$ cai muito mais devagar e **estaciona num patamar não nulo**. Plote o mapa dos pixels que mudaram: eles desenham exatamente a fronteira entre bacias.

Escreva a interpretação com suas palavras. É a demonstração, com os seus próprios dados, de que existem **duas noções distintas de convergência** e de que elas podem discordar. Essa é a resposta à pergunta 4.3 e é o ponto mais alto do trabalho.

**5.2 Dependência das escolhas de modelo.** Repita a imagem para três valores de $\\tilde\\varepsilon$, três de $\\tilde\\gamma$ e dois de $R\_{\\text{cap}}$. Mostre que a figura muda qualitativamente. Separe explicitamente, em uma tabela:

| Parâmetro | Tipo | O que se exige dele |
| :---- | :---- | :---- |
| $\\Delta t$, tolerâncias, $n$ | numérico | convergência demonstrada |
| $\\tilde\\varepsilon$, $\\tilde\\gamma$, $R\_{\\text{cap}}$, $t\_{\\max}$ | modelo | escolha justificada \+ análise de sensibilidade |

**5.3 Trajetórias ilustrativas.** Escolha 4–6 pontos de partida interessantes (um em cada bacia, um bem perto da fronteira) e plote as trajetórias 3D completas, coloridas pelo destino, com as massas desenhadas como esferas coloridas. **Desenhe também o plano de corte**, como um retângulo semitransparente atravessando a cena, com os pontos de partida marcados sobre ele. Essa figura é o que torna visualmente óbvia a relação entre as duas dimensões do trabalho: o corte 2D é onde as partículas *nascem*, o espaço 3D é onde elas *andam*. Inclua o gráfico do resíduo $\\mathcal R(t)$ da identidade (10) ao lado de cada trajetória.

---

### Parte 6 — Fazer isso rodar (Engenharia Computacional)

Uma imagem $1000\\times1000$ com $\\sim10^4$ passos por trajetória são $\\sim10^{10}$ avaliações de força. Em Python puro, com laços, isso leva dias. Você precisa de duas coisas:

**6.1 Vetorização sobre partículas, não sobre o tempo.** Não integre uma partícula de cada vez. Guarde o estado como arrays de forma $(P,3)$ e avance **todas as $P$ partículas simultaneamente**, usando broadcasting para montar o array de forças de forma $(P,N,3)$. Este é o passo que torna o trabalho viável. Meça e reporte o ganho de velocidade.

**6.2 Máscara de partículas ativas.** As partículas terminam em tempos diferentes. Mantenha um vetor booleano de quem ainda está ativa e, periodicamente, compacte os arrays para não desperdiçar cálculo com partículas já resolvidas. Plote a fração ativa em função do tempo.

**Bônus (6.3) — refinamento adaptativo da imagem.** Calcule uma grade grossa e subdivida apenas as células cujos cantos têm cores diferentes (quadtree). Como a fronteira tem dimensão fractal $D \< 2$, o custo escala como $\\delta^{-D}$ em vez de $\\delta^{-2}$. Se você fizer o item T2 abaixo, verá que **a dimensão fractal que você mediu prevê o tempo de execução do seu próprio código**.

---

## 6\. Itens bônus (opcionais)

Só comece depois que o núcleo estiver completo e verificado.

**T1 — Massas em movimento.** Coloque as massas em órbitas circulares prescritas. A equação (7) vira não-autônoma e as bacias passam a depender da fase inicial $\\phi\_0$ das massas. Entregável natural: uma animação varrendo $\\phi\_0$.

**T2 — Dimensão fractal da fronteira.** Cubra os pixels de fronteira com caixas de lado $\\delta$ e conte:

$$N(\\delta)\\sim\\delta^{-D}\\ \\Longrightarrow\\ D \= \\text{inclinação de } \\ln N \\text{ vs } \\ln(1/\\delta)$$

Para uma curva suave $D=1$; aqui espera-se $D\\approx 1{,}3$–$1{,}8$. Discuta os limites de resolução que restringem a faixa útil do ajuste nos dois extremos.

**T3 — Expoente de incerteza.** Sorteie um ponto, perturbe-o por $\\delta$ e veja se a cor muda. A fração incerta escala como $f(\\delta)\\sim\\delta^\\alpha$ com $\\alpha \= 2 \- D$ num corte 2D. Com $\\alpha\\approx0{,}3$, reduzir a incerteza por 10 exige conhecer a condição inicial $10^{3{,}3}$ vezes melhor. Interprete.

**T4 — Verlet de quarta ordem.** Composição de Yoshida: aplique três passos de Verlet com passos $w\_1\\Delta t$, $w\_0\\Delta t$, $w\_1\\Delta t$, onde

$$w\_1 \= \\frac{1}{2-2^{1/3}}, \\qquad w\_0 \= \-2^{1/3}w\_1$$

O resultado é simplético **e** de quarta ordem, com 3 avaliações de força por passo. Verifique as duas propriedades.

**T5 — Animação interativa.** Trajetória com rastro que desvanece e leitura ao vivo do resíduo $\\mathcal R(t)$.

**T6 — Pilha de cortes paralelos.** Em vez de um único plano, gere a imagem em cinco cortes paralelos (por exemplo $z=-1;,-0{,}5;,0;,0{,}5;,1$) e mostre como as bacias se deformam ao longo do eixo. Custa 5× e não $1000\\times$, e não exige renderização volumétrica — é a forma barata de enxergar a estrutura tridimensional das bacias. Acompanhe com o gráfico de $\\mu\_i$ em função da altura do corte.

---

## 7\. Cronograma e checkpoints

Encontros de 15 minutos por grupo, semanais, **com código rodando na tela**. Não é apresentação, é acompanhamento.

| Semana | Entrega no checkpoint |
| :---- | :---- |
| 1 | Parte 0 completa (papel). Configuração do grupo conferida. |
| 2 | Campo gravitacional implementado e testado contra caso de uma massa. Euler e RK4 rodando. |
| 3 | Parte 1: órbitas de Kepler \+ tabela de ordem observada. |
| 4 | Parte 2: Verlet implementado, comparação de energia a custo igual. |
| 5 | Parte 3: detecção de eventos funcionando; primeira imagem de bacias (grade grossa, $200\\times200$). |
| 6 | Parte 6: versão vetorizada; imagem $1000\\times1000$. Parte 4 medida. |
| 7 | Parte 5: estudo de convergência completo ($\\Pi$ e $\\Delta\\mu$) e sensibilidade a $\\tilde\\varepsilon,\\tilde\\gamma$. |
| 8 | Entrega do relatório \+ defesa oral. |

---

## 8\. Configuração do seu grupo

Cada grupo recebe uma configuração diferente, gerada a partir da matrícula de um dos integrantes:

- número de massas $N$ (entre 3 e 5);  
- massas $M\_i$ (normalizadas, $\\sum M\_i \= 1$);  
- posições $\\mathbf R\_i$ em 3D;  
- valores de referência de $\\tilde\\gamma$ e $\\tilde\\varepsilon$;  
- orientação do plano de corte para a imagem.

As massas são geradas **explicitamente não-coplanares**: o gerador exige que o tetraedro formado por quatro delas tenha volume não nulo, isto é,

$$\\big|,(\\mathbf R\_2-\\mathbf R\_1)\\cdot\\big\[(\\mathbf R\_3-\\mathbf R\_1)\\times(\\mathbf R\_4-\\mathbf R\_1)\\big\],\\big| ;\>; V\_{\\min}$$

Isso garante que nenhum plano de corte seja invariante e que a dinâmica seja genuinamente tridimensional: as partículas saem do plano de partida, vagueiam pelo espaço e voltam. **Confira essa condição na sua configuração** — calcule o produto misto acima e reporte o valor. Se por acaso der zero ou quase zero, avise: a configuração precisa ser regerada.

**Sua imagem não é igual à de nenhum outro grupo**, e os números que você vai reportar ($\\lambda$, $D$, $\\mu\_i$, $\\Pi$) são todos específicos da sua configuração.

---

## 9\. O que entregar

1. **Relatório em PDF**, estruturado nas Partes 0 a 6\. Toda figura numerada, com legenda, e citada no texto.  
2. **Código-fonte**, organizado em módulos, com um `README` explicando como reproduzir cada figura do relatório.  
3. **Imagem final de bacias**, em alta resolução, convergida, com os parâmetros usados indicados na legenda.  
4. **Diário de falhas** (ver abaixo).  
5. **Apêndice de prompts de IA** (Seção 3, regra 2).  
6. **Defesa oral**, individual, de 20 minutos por grupo.

### O diário de falhas

Documente **no mínimo quatro** falhas numéricas que vocês encontraram, cada uma com quatro itens:

1. **Sintoma** — o que você observou (NaN, gráfico estranho, código lento, resultado que não converge).  
2. **Diagnóstico** — como você descobriu a causa. Qual teste isolou o problema?  
3. **Correção** — o que foi mudado.  
4. **Princípio geral** — o que isso ensina, além deste caso específico.

Esta é uma das partes mais valorizadas da nota. Um trabalho sem falhas documentadas ou não foi feito de verdade, ou perdeu o aprendizado pelo caminho. Falhas são o registro do trabalho intelectual.

---

## 10\. Avaliação

| Componente | Peso |
| :---- | :---- |
| Parte 0 (deduções no papel) | 10% |
| Parte 1 (verificação com Kepler \+ tabela de ordem) | 20% |
| Parte 2 (comparação de integradores: energia e reversibilidade) | 15% |
| Parte 3 (detecção de eventos e seu efeito medido na imagem) | 15% |
| Parte 5 (convergência: $\\Pi$ vs $\\Delta\\mu$, e sensibilidade a parâmetros) | 15% |
| Identidade de dissipação (10) usada como diagnóstico ao longo do trabalho | 5% |
| Diário de falhas \+ apêndice de prompts | 10% |
| Desempenho computacional (Parte 6\) \+ defesa oral | 10% |
| Bônus (T1–T5) | até \+15% |

Repare na distribuição: quase **70%** da nota está em análise, verificação e diagnóstico. A imagem bonita, sozinha, não vale quase nada — ela é fácil de gerar e difícil de justificar.

---

## 11\. Perguntas da defesa oral

Você deve conseguir responder qualquer uma destas, sem consultar nada:

1. Por que a massa da partícula de teste não aparece na equação do movimento?  
2. Por que o denominador tem expoente $3/2$ e não $1/2$?  
3. O que acontece com a sua imagem se $\\varepsilon \\to 0$? E se $\\varepsilon$ ficar maior que a distância entre as massas?  
4. Sem atrito, a partícula cai em algum lugar? Justifique.  
5. Por que Verlet, sendo de ordem 2, conserva energia melhor que RK4, de ordem 4?  
6. Um passo do seu integrador pulou por cima da esfera de captura. O que acontece com a cor do pixel? Como você evitou isso?  
7. Você mediu $\\lambda \= $ (seu valor). Por quanto tempo a sua simulação é confiável, ponto a ponto? Como esse número muda se você usar precisão quádrupla?  
8. Se trajetórias individuais são imprevisíveis, por que a sua imagem é reprodutível? Que quantidade converge e qual não converge?  
9. Qual dos seus parâmetros é numérico e qual é de modelo? Qual critério distingue os dois?  
10. Se todas as massas estivessem no mesmo plano e eu soltasse a partícula desse plano, o que aconteceria? Demonstre. Por que isso importa para a escolha da sua configuração?  
11. Vou mudar a posição de uma massa em 2%. Qual bacia cresce, e o que acontece com $\\Pi$? **Responda antes de rodar**, depois rode e compare.

---

## 12\. Dicas práticas

- **Comece com $N=1$ em 2D.** Só vá para 3D e $N=4$ quando o caso de uma massa estiver verificado contra Kepler.  
- **Use `float64` sempre.** Em `float32` o piso de arredondamento fica acima da estrutura fina que você está tentando resolver, e o estudo de convergência da Parte 5 perde o sentido.  
- **Fixe as sementes aleatórias** em qualquer coisa que use sorteio, para que suas figuras sejam reproduzíveis.  
- **Salve os resultados brutos em disco**, não só as figuras. Rodar de novo custa horas.  
- **Confira as unidades adimensionais o tempo todo.** Se algum número no seu código for maior que $10^3$ ou menor que $10^{-6}$, desconfie.  
- **Plote cedo e plote muito.** Uma trajetória desenhada vale por dez horas de leitura de código.  
- **Teste o campo gravitacional isoladamente**, antes de acoplá-lo ao integrador: calcule $\\mathbf g$ num ponto conhecido à mão e compare.

---

## 13\. Referências

- Burden, R. L.; Faires, J. D. *Análise Numérica* — métodos de passo único, controle de passo, raízes de equações.  
- Press, W. H. et al. *Numerical Recipes*, cap. 16–17 — integração de EDOs, eventos, problemas rígidos.  
- Hairer, E.; Lubich, C.; Wanner, G. *Geometric Numerical Integration*, cap. 1 — o experimento clássico Verlet vs RK4 (leitura opcional, mas o capítulo 1 é acessível).  
- Strogatz, S. *Nonlinear Dynamics and Chaos*, cap. 9–11 — caos, expoentes de Lyapunov, fractais. Livro escrito para engenheiros.  
- Ott, E.; Grebogi, C.; Yorke, J. — trabalhos originais sobre fronteiras fractais de bacias e o expoente de incerteza.

---

*Dúvidas sobre o enunciado: tragam no checkpoint semanal. Dúvidas sobre implementação: tentem primeiro, depois tragam o que já tentaram.*