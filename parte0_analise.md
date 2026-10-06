# Parte 0 - Analise no Papel (Deducoes Analiticas e Fundamentacao Teorica)

**Disciplina:** Calculo Numerico - UERJ  
**Professor:** Prof. Dr. Vahid Nikoofard  
**Tema:** Bacias de Atracao Gravitacionais em $\mathbb{R}^3$

---

## 0.1 Deducao da Forca Gravitacional Vetorial

Partindo da Lei da Gravitacao Universal de Newton em forma escalar:
$$F = G \frac{M m}{d^2}$$
onde $d = \|\mathbf{r} - \mathbf{R}\|$ e a distancia euclidiana entre o atrator de massa $M$ localizado em $\mathbf{R}$ e a particula de teste de massa $m$ localizada em $\mathbf{r}$.

Definindo o vetor deslocamento que aponta do atrator para a particula:
$$\mathbf{d} = \mathbf{r} - \mathbf{R}$$
O vetor unitario nessa direcao e sentido e:
$$\hat{\mathbf{u}} = \frac{\mathbf{d}}{\|\mathbf{d}\|} = \frac{\mathbf{r} - \mathbf{R}}{\|\mathbf{r} - \mathbf{R}\|}$$

Como a forca gravitacional e exclusivamente atrativa, o vetor forca $\mathbf{F}$ aponta no sentido oposto a $\hat{\mathbf{u}}$:
$$\mathbf{F} = - F \hat{\mathbf{u}} = - G \frac{M m}{\|\mathbf{r} - \mathbf{R}\|^2} \left( \frac{\mathbf{r} - \mathbf{R}}{\|\mathbf{r} - \mathbf{R}\|} \right)$$

Multiplicando os denominadores $\|\mathbf{r} - \mathbf{R}\|^2 \cdot \|\mathbf{r} - \mathbf{R}\| = \|\mathbf{r} - \mathbf{R}\|^3$:
$$\mathbf{F}(\mathbf{r}) = - G M m \frac{\mathbf{r} - \mathbf{R}}{\|\mathbf{r} - \mathbf{R}\|^3}$$

> **Origem do expoente 3:** O expoente 3 surge necessariamente no denominador da formulacao vetorial cartesiana porque a lei física e o inverso do quadrado da distancia ($d^2$), e a inclusao do vetor unitario $\frac{\mathbf{d}}{\|\mathbf{d}\|}$ para indicar a direcao introduz uma terceira potencia de $d$ no denominador.

Pela Segunda Lei de Newton para a particula de teste:
$$\mathbf{F} = m \mathbf{a} \implies m \ddot{\mathbf{r}} = - G M m \frac{\mathbf{r} - \mathbf{R}}{\|\mathbf{r} - \mathbf{R}\|^3}$$
A massa da particula de teste $m$ cancela-se em ambos os lados (Principio da Equivalencia), resultando no campo de aceleracao:
$$\mathbf{g}(\mathbf{r}) = - G M \frac{\mathbf{r} - \mathbf{R}}{\|\mathbf{r} - \mathbf{R}\|^3}$$

---

## 0.2 Potencial Amaciado e Campo Gravitacional

Para eliminar a singularidade em $\mathbf{r} \to \mathbf{R}_i$, adota-se o potencial amaciado de Plummer com parametro $\varepsilon > 0$:
$$\Phi(\mathbf{r}) = - \sum_{i=1}^N \frac{G M_i}{\sqrt{\|\mathbf{r} - \mathbf{R}_i\|^2 + \varepsilon^2}}$$

Escrevendo em coordenadas cartesianas $\mathbf{r} = (x, y, z)$ e $\mathbf{R}_i = (X_i, Y_i, Z_i)$:
$$\|\mathbf{r} - \mathbf{R}_i\|^2 + \varepsilon^2 = (x - X_i)^2 + (y - Y_i)^2 + (z - Z_i)^2 + \varepsilon^2$$

Calculando a derivada parcial em relacao a $x$:
$$\frac{\partial \Phi}{\partial x} = - \sum_{i=1}^N G M_i \frac{\partial}{\partial x} \left[ (x - X_i)^2 + (y - Y_i)^2 + (z - Z_i)^2 + \varepsilon^2 \right]^{-1/2}$$
Pela regra da cadeia:
$$\frac{\partial \Phi}{\partial x} = - \sum_{i=1}^N G M_i \left( -\frac{1}{2} \right) \left[ \|\mathbf{r} - \mathbf{R}_i\|^2 + \varepsilon^2 \right]^{-3/2} \cdot 2(x - X_i) = \sum_{i=1}^N \frac{G M_i (x - X_i)}{\left( \|\mathbf{r} - \mathbf{R}_i\|^2 + \varepsilon^2 \right)^{3/2}}$$

De forma identica para $y$ e $z$:
$$\nabla \Phi(\mathbf{r}) = \sum_{i=1}^N \frac{G M_i (\mathbf{r} - \mathbf{R}_i)}{\left( \|\mathbf{r} - \mathbf{R}_i\|^2 + \varepsilon^2 \right)^{3/2}}$$

O campo de forca por unidade de massa (campo gravitacional) e o gradiente negativo do potencial:
$$\mathbf{g}(\mathbf{r}) = - \nabla \Phi(\mathbf{r}) = - \sum_{i=1}^N \frac{G M_i (\mathbf{r} - \mathbf{R}_i)}{\left( \|\mathbf{r} - \mathbf{R}_i\|^2 + \varepsilon^2 \right)^{3/2}}$$
Isso demonstra rigorosamente que $\mathbf{g} = -\nabla\Phi$.

---

## 0.3 Adimensionalizacao Completa das Equacoes

A equacao do movimento com amortecimento viscoso (arrasto linear) e:
$$\frac{d^2\mathbf{r}}{dt^2} = - \sum_{i=1}^N \frac{G M_i (\mathbf{r} - \mathbf{R}_i)}{\left( \|\mathbf{r} - \mathbf{R}_i\|^2 + \varepsilon^2 \right)^{3/2}} - \gamma \frac{d\mathbf{r}}{dt}$$

Definem-se as mudancas de variaveis adimensionais (denotadas com til $\sim$):
- Posicao: $\mathbf{r} = L \tilde{\mathbf{r}}$, com $\mathbf{R}_i = L \tilde{\mathbf{R}}_i$
- Parametro de amaciamento: $\varepsilon = L \tilde{\varepsilon} \implies \tilde{\varepsilon} = \frac{\varepsilon}{L}$
- Tempo: $t = T_c \tilde{t}$
- Massas: $M_i = M_{\text{tot}} \tilde{M}_i$, onde $M_{\text{tot}} = \sum_{i=1}^N M_i \implies \sum_{i=1}^N \tilde{M}_i = 1$

Pela regra da cadeia para derivadas temporais:
$$\frac{d\mathbf{r}}{dt} = \frac{d(L\tilde{\mathbf{r}})}{d(T_c\tilde{t})} = \frac{L}{T_c} \frac{d\tilde{\mathbf{r}}}{d\tilde{t}}$$
$$\frac{d^2\mathbf{r}}{dt^2} = \frac{L}{T_c^2} \frac{d^2\tilde{\mathbf{r}}}{d\tilde{t}^2}$$

Substituindo na equacao diferencial:
$$\frac{L}{T_c^2} \frac{d^2\tilde{\mathbf{r}}}{d\tilde{t}^2} = - \sum_{i=1}^N \frac{G (M_{\text{tot}} \tilde{M}_i) L (\tilde{\mathbf{r}} - \tilde{\mathbf{R}}_i)}{\left[ L^2 \|\tilde{\mathbf{r}} - \tilde{\mathbf{R}}_i\|^2 + L^2 \tilde{\varepsilon}^2 \right]^{3/2}} - \gamma \frac{L}{T_c} \frac{d\tilde{\mathbf{r}}}{d\tilde{t}}$$

Fatorando $L^3$ do denominador:
$$\left[ L^2 \left( \|\tilde{\mathbf{r}} - \tilde{\mathbf{R}}_i\|^2 + \tilde{\varepsilon}^2 \right) \right]^{3/2} = L^3 \left( \|\tilde{\mathbf{r}} - \tilde{\mathbf{R}}_i\|^2 + \tilde{\varepsilon}^2 \right)^{3/2}$$
Assim:
$$\frac{L}{T_c^2} \frac{d^2\tilde{\mathbf{r}}}{d\tilde{t}^2} = - \frac{G M_{\text{tot}} L}{L^3} \sum_{i=1}^N \frac{\tilde{M}_i (\tilde{\mathbf{r}} - \tilde{\mathbf{R}}_i)}{\left( \|\tilde{\mathbf{r}} - \tilde{\mathbf{R}}_i\|^2 + \tilde{\varepsilon}^2 \right)^{3/2}} - \gamma \frac{L}{T_c} \frac{d\tilde{\mathbf{r}}}{d\tilde{t}}$$

Multiplicando toda a equacao por $\frac{T_c^2}{L}$:
$$\frac{d^2\tilde{\mathbf{r}}}{d\tilde{t}^2} = - \left( \frac{G M_{\text{tot}} T_c^2}{L^3} \right) \sum_{i=1}^N \frac{\tilde{M}_i (\tilde{\mathbf{r}} - \tilde{\mathbf{R}}_i)}{\left( \|\tilde{\mathbf{r}} - \tilde{\mathbf{R}}_i\|^2 + \tilde{\varepsilon}^2 \right)^{3/2}} - (\gamma T_c) \frac{d\tilde{\mathbf{r}}}{d\tilde{t}}$$

Para que a equacao fique adimensional e o coeficiente gravitacional seja exatamente $1$, escolhe-se a escala natural de tempo:
$$\frac{G M_{\text{tot}} T_c^2}{L^3} = 1 \implies T_c = \sqrt{\frac{L^3}{G M_{\text{tot}}}}$$

Definindo o amortecimento adimensional:
$$\tilde{\gamma} = \gamma T_c$$

Obtem-se a forma adimensional final utilizada no codigo:
$$\frac{d^2\tilde{\mathbf{r}}}{d\tilde{t}^2} = - \sum_{i=1}^N \frac{\tilde{M}_i (\tilde{\mathbf{r}} - \tilde{\mathbf{R}}_i)}{\left( \|\tilde{\mathbf{r}} - \tilde{\mathbf{R}}_i\|^2 + \tilde{\varepsilon}^2 \right)^{3/2}} - \tilde{\gamma} \frac{d\tilde{\mathbf{r}}}{d\tilde{t}}$$
com $G=1$, $M_{\text{tot}}=1$ e todas as grandezas em unidades adimensionais canônicas.

---

## 0.4 Identidade de Dissipacao e Teorema Trabalho-Energia

A energia mecanica total por unidade de massa e:
$$E(t) = \frac{1}{2} \|\mathbf{v}(t)\|^2 + \Phi(\mathbf{r}(t))$$

Derivando $E(t)$ em relacao ao tempo:
$$\frac{dE}{dt} = \mathbf{v} \cdot \frac{d\mathbf{v}}{dt} + \nabla\Phi(\mathbf{r}) \cdot \frac{d\mathbf{r}}{dt} = \mathbf{v} \cdot \dot{\mathbf{v}} + \nabla\Phi(\mathbf{r}) \cdot \mathbf{v}$$

Substituindo a equacao de movimento $\dot{\mathbf{v}} = \mathbf{g}(\mathbf{r}) - \gamma \mathbf{v} = -\nabla\Phi(\mathbf{r}) - \gamma \mathbf{v}$:
$$\frac{dE}{dt} = \mathbf{v} \cdot (-\nabla\Phi(\mathbf{r}) - \gamma \mathbf{v}) + \nabla\Phi(\mathbf{r}) \cdot \mathbf{v} = - \mathbf{v} \cdot \nabla\Phi(\mathbf{r}) - \gamma \|\mathbf{v}\|^2 + \mathbf{v} \cdot \nabla\Phi(\mathbf{r}) = -\gamma \|\mathbf{v}\|^2$$

Integrando de $0$ a $t$:
$$E(t) - E(0) = -\gamma \int_0^t \|\mathbf{v}(s)\|^2 \, ds \implies E(t) - E(0) + \gamma \int_0^t \|\mathbf{v}(s)\|^2 \, ds = 0$$

Define-se o **residuo numerico de dissipacao**:
$$\mathcal{R}(t) = \left| E(t) - E(0) + \gamma \int_0^t \|\mathbf{v}(s)\|^2 \, ds \right|$$
Em uma integracao numerica exata, $\mathcal{R}(t) \equiv 0$. O valor de $\mathcal{R}(t)$ e o diagnostico central da qualidade e da consistencia da integracao temporal com arrasto.

---

## 0.5 Criterio Triplo de Escape

Para distancias muito grandes das massas ($\|\mathbf{r}\| \gg \|\mathbf{R}_i\|$), o potencial gravitacional comporta-se como o de uma massa pontual unitaria na origem:
$$\Phi(\mathbf{r}) \approx -\frac{1}{\|\mathbf{r}\|}$$
A energia e aproximadamente:
$$E \approx \frac{1}{2}\|\mathbf{v}\|^2 - \frac{1}{\|\mathbf{r}\|}$$

Para que uma particula escape definitivamente do campo atrativo em direcao ao infinito, tres condicoes sao simultaneamente necessarias e suficientes:
1. **Energia Mecanica Positiva ($E > 0$):** Garante velocidade de excesso hiperbolico $v_\infty = \sqrt{2E} > 0$, de modo que a particula nao esteja em orbita eliptica ligada.
2. **Velocidade Radial Afastando-se ($\mathbf{r} \cdot \mathbf{v} > 0$):** O produto escalar positivo garante que a distancia radial esta aumentando ($\frac{d}{dt}\|\mathbf{r}\|^2 = 2 \mathbf{r}\cdot\mathbf{v} > 0$), evitando classificar particulas em aproximacao (passagem no periastro).
3. **Distancia Superior ao Raio de Escape ($\|\mathbf{r}\| > R_{\text{esc}}$):** Garante que a particula ja superou a regiao de forte acoplamento caotico multi-corpos ($R_{\text{esc}} \gg \max \|\mathbf{R}_i\|$, tipicamente $R_{\text{esc}} = 20$).

---

## 0.6 Estimativa de Escala e Tempo de Queda Livre Radial

Para estimar a escala temporal do sistema, considera-se a queda livre radial de repouso ($v_0 = 0$) a partir de uma distancia $r_0$ em direcao a uma massa pontual unitaria na origem.

Pela conservacao da energia:
$$\frac{1}{2}\left(\frac{dr}{dt}\right)^2 - \frac{1}{r} = -\frac{1}{r_0} \implies \frac{dr}{dt} = - \sqrt{2 \left(\frac{1}{r} - \frac{1}{r_0}\right)} = - \sqrt{\frac{2(r_0 - r)}{r_0 r}}$$

Separando variaveis e integrando de $r = r_0$ ate $r = 0$:
$$t_{\text{queda}} = \int_0^{r_0} \sqrt{\frac{r_0 r}{2(r_0 - r)}} \, dr$$
Fazendo a substituicao trigonometrica $r = r_0 \cos^2\theta$, com $dr = -2 r_0 \cos\theta \sin\theta \, d\theta$:
$$t_{\text{queda}} = \sqrt{\frac{r_0^3}{2}} \int_0^{\pi/2} 2\cos^2\theta \, d\theta = \sqrt{\frac{r_0^3}{2}} \int_0^{\pi/2} (1 + \cos 2\theta) \, d\theta = \frac{\pi}{2\sqrt{2}} r_0^{3/2}$$

Para uma particula solta na distancia adimensional de corte $r_0 = 2.0$:
$$t_{\text{queda}}(2) = \frac{\pi}{2\sqrt{2}} (2)^{3/2} = \frac{\pi}{2\sqrt{2}} \cdot 2\sqrt{2} = \pi \approx 3.1416$$

> **Justificativa de $t_{\text{max}} = 100$:** Como o tempo de queda caracteristico e de ordem $\sim 3$, o tempo maximo de simulacao $t_{\text{max}} = 100$ cobre mais de 30 tempos dinamicos de travessia do sistema. Particulas que nao foram capturadas ou nao escaparam nesse intervalo ou estao em trajetorias quasi-estaveis altamente complexas na fronteira caotica ou requerem aumento de $\gamma$.

---

## 0.7 Parametros da Configuracao do Grupo

Executando o script `parte0_calculos.py` para a configuracao oficial de 4 atratores:

```
Massas:
M = [0.35, 0.25, 0.20, 0.20],  soma M_tot = 1.0

Posicoes 3D:
R_0 = [-1.2,  0.0,  0.0]
R_1 = [ 1.0,  1.1,  0.3]
R_2 = [ 0.5, -1.4, -0.2]
R_3 = [-0.4,  1.5, -0.7]
```

- **Distancia media entre atratores:** $L_{\text{medio}} \approx 2.3282$
- **Produto misto do tetraedro formado pelos 4 atratores:**
  $$V_{\text{misto}} = (\mathbf{R}_1 - \mathbf{R}_0) \cdot \left[ (\mathbf{R}_2 - \mathbf{R}_0) \times (\mathbf{R}_3 - \mathbf{R}_0) \right] = 5.0500$$
  Como $V_{\text{misto}} = 5.05 > 0$, as quatro massas sao **estritamente nao-coplanares**. O plano de corte da imagem ($z=0$) nao e um plano invariante da dinamica, garantindo que as trajetorias sejam genuinamente tridimensionais.
- **Parametros Adimensionais de Modelo:**
  - $\tilde{\varepsilon} = 0.18$: Suficientemente pequeno para manter o campo gravitacional proximo do newtoniano puro, mas grande o bastante para evitar singularidade e passos infinitesimais perto dos centros.
  - $\tilde{\gamma} = 0.15$: Amortecimento moderado que permite a presenca de multiplas oscilações e caos transiente antes do decaimento em uma das bacias.
  - $R_{\text{cap}} = 0.22$: Raio da esfera de captura em torno de cada atrator ($R_{\text{cap}} > \varepsilon$).
  - $R_{\text{esc}} = 20.0$: Raio da esfera de escape.

---

## 0.8 Discussao da Hipotese de Massas Fixas

A aproximacao de que as massas atratoras permanecem estaticas no espaco apoia-se no **Problema Restrito dos $N+1$ Corpos**:
- A massa da particula de teste e infinitesimal em relacao as massas dos atratores: $m \ll M_i$. Portanto, a forca que a particula exerce sobre os atratores e desprezivel e nao perturba suas posicoes.
- No modelo fisico ideal, massas reais orbitariam em torno do centro de massa comum segundo solucoes de $N$ corpos. Fixar os atratores equivale a considerar um referencial onde forcas externas de sustentacao equilibram as atracoes mutuas (ou um potencial efetivo medio em escalas onde os tempos de relaxamento dos atratores sao ordens de magnitude maiores que a orbita da particula de teste).
