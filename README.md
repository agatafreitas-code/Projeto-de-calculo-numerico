# Bacias de Atracao Gravitacionais

## Ambiente

```powershell
python -m pip install -r requirements.txt
```

## Interface

```powershell
streamlit run app.py
```

A interface possui abas para as Partes 0 a 6 e Itens Bônus. Os campos da barra lateral permitem alterar massas, posicoes 3D, dominio do corte, `eps`, `gamma`, `R_cap`, `R_esc`, `tmax`, `dt`, tolerancias e resolucao.

## Execucoes por etapa

```powershell
python parte1_verificacao.py --dt 0.01
python parte2_integradores.py --periods 20 --dt 0.02
python parte3_eventos.py --x 0 --y 0 --tmax 100
python parte4_caos.py --x -2.0 --y -2.0 --boundary-x 0.2 --boundary-y 0.2 --intervals 100
python parte5_bacias.py --resolution 32
python parte6_desempenho.py --particles 10000 --repetitions 20
python bonus_itens.py --test-all
```

## Dados do relatorio

Na aba Parte 1, use os botoes de download para salvar as tabelas de orbitas circulares, orbitas elipticas, refinamento `dt` e plano invariante.

Na aba Parte 3, salve a figura da trajetoria e o grafico do residuo de dissipacao.

Na aba Parte 5, execute a convergencia em `dt`, a comparacao entre teste ingenuo e eventos e a sensibilidade de `eps`, `gamma` e `R_cap`.

Na aba Parte 6, execute a bacia vetorizada e registre tempo total, formato dos arrays e fracao de particulas ativas.

## Reprodutibilidade

Use `float64` e mantenha a configuracao de massas, posicoes, parametros e resolucao registrada nas tabelas e legendas. Resultados do relatorio devem ser acompanhados dos dados brutos correspondentes.
