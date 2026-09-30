# Relatório do Projeto de Machine Learning

**Classificação binária da remuneração de profissionais júnior na área de tecnologia**

---

## 1. Problema e objetivo do projeto

O início de carreira em tecnologia costuma gerar muita dúvida sobre o que realmente influencia o salário: a linguagem que a pessoa usa, o regime de trabalho, o porte da empresa ou o país em que atua. Anúncios de vaga e relatos isolados não mostram um padrão claro, especialmente para quem ainda tem poucos anos de experiência.

Este projeto investiga a seguinte questão: **é possível classificar a remuneração de profissionais júnior de TI como alta ou baixa a partir de características de cargo, stack, localização e contexto de trabalho?**

O objetivo é treinar um modelo de **classificação binária** que preveja se o salário anual (convertido em dólares) de um profissional em início de carreira tende a ficar **acima** ou **abaixo** da mediana do próprio conjunto de treino. As classes foram definidas assim de propósito:

- **Alta:** salário maior ou igual à mediana calculada **somente no treino**;
- **Baixa:** salário menor que essa mediana.

Usar a mediana do treino evita vazamento de informação do teste e equilibra as duas classes, o que facilita a avaliação com acurácia e matriz de confusão — métricas pedidas na disciplina.

O recorte é específico: profissionais empregados ou autônomos, com até **três anos** de experiência profissional, em cargos de desenvolvimento de software, dados, mobile e infraestrutura. O trabalho não busca prever o valor exato em dólares (regressão), e sim um sinal de mercado útil para quem está começando: o perfil tende a ser mais ou menos valorizado.

---

## 2. Descrição do dataset e justificativa da escolha

Foi utilizado o arquivo `results.csv` da **Stack Overflow Developer Survey 2025**, pesquisa anual respondida por profissionais de tecnologia em vários países. A base original contém **49.191 respostas** e **172 colunas**, volume adequado para análise exploratória e modelagem.

### Por que essa base

- **Não foi usada em aula.** Os conjuntos da disciplina (Churn, Iris, Credit, Cars e transações para associação) são outros; este dataset atende à regra de unicidade.
- **É público e reconhecido** no mercado de TI, o que facilita explicar a origem dos dados na apresentação.
- **Tem variável salarial comparável entre países** (`ConvertedCompYearly`), já convertida para dólares anuais.
- **Descreve o perfil técnico** (cargo, linguagens, frameworks, remoto, porte da empresa, escolaridade e país), alinhado ao problema escolhido.
- **Permite filtrar juniores** pela variável `WorkExp`, em vez de misturar plenos e seniores no mesmo modelo.

### Variáveis utilizadas

Nem todas as 172 colunas entram no modelo. Foram selecionadas as que têm relação direta com o problema:

| Variável original | Papel no projeto |
|---|---|
| `WorkExp` | Filtro de júnior (até 3 anos) e feature numérica |
| `YearsCode` | Tempo de código (incluindo estudos/hobby) |
| `ConvertedCompYearly` | Base para o alvo Alta/Baixa |
| `DevType` | Cargo, depois agrupado em categorias macro |
| `EdLevel` | Escolaridade |
| `Age` | Faixa etária |
| `Employment` | Filtro (empregado ou freelancer) |
| `ICorPM` | Filtro (exclui gestores) |
| `RemoteWork` | Regime de trabalho |
| `OrgSize` | Porte da empresa |
| `Country` | País e região de mercado |
| `Industry` | Setor da empresa |
| `LanguageHaveWorkedWith` | Linguagens usadas |
| `WebframeHaveWorkedWith` | Frameworks web usados |

A justificativa prática é simples: salário de júnior no mercado global de TI depende muito de **onde a pessoa trabalha**, **como trabalha** e **com qual stack**, mais do que de dezenas de perguntas da pesquisa sobre opinião em IA ou uso do Stack Overflow.

---

## 3. Análise exploratória e pré-processamento

O fluxo seguiu o padrão da disciplina: carregar os dados, olhar nulos e distribuições, limpar, transformar e só então treinar o modelo.

### 3.1 Situação inicial dos dados

Na base bruta havia muitos valores ausentes, o que é esperado em pesquisa voluntária. Exemplos aproximados nas colunas usadas:

- `ConvertedCompYearly`: 25.244 nulos (muita gente não informa salário);
- `RemoteWork` e `OrgSize`: cerca de 15 mil nulos;
- `LanguageHaveWorkedWith` e `WebframeHaveWorkedWith`: listas vazias em milhares de respostas;
- o texto `"NA"` foi padronizado para nulo real, porque o arquivo não vinha só com células vazias.

Também existiam **inconsistências graves** para um recorte júnior: salário de **US$ 1** em países de alto custo, salário de **quase US$ 10 milhões** para quem tinha 1 ano de experiência, e `YearsCode` igual a **100**.

### 3.2 Filtros de escopo (juniores de TI)

Foram mantidos apenas:

1. Empregados ou autônomos/freelancers;
2. Cargos de tecnologia alinhados ao tema (desenvolvimento, dados, mobile, infra/segurança);
3. **Contribuidores individuais** (gestores foram removidos);
4. **Experiência profissional de 0 a 3 anos**;
5. Salário anual convertido informado e positivo.

Cargos fora do recorte (estudante, aposentado, C-level, founder, gerente de projeto/produto, pesquisador acadêmico, “Other”, etc.) foram excluídos para reduzir ruído.

Depois desses filtros restaram **2.440** registros. Em seguida o salário foi limpo.

### 3.3 Limpeza de salário e de anos de código

- **Piso:** removidos **113** registros com salário abaixo de **US$ 1.000** ao ano (valores irreais).
- **Teto:** outliers altos pelo método **IQR** (limite = Q3 + 3×IQR ≈ **US$ 213.303**). Removidos **24** casos extremos.
- **YearsCode:** nulos preenchidos com a mediana do recorte júnior; valores acima de 30 anos (incompatíveis com o perfil) substituídos pela mediana; se o tempo de código era menor que o tempo de trabalho, o valor foi alinhado.

A amostra final para análise e modelo ficou com **2.303 profissionais**, quantidade ainda razoável para classificação.

Nulos em variáveis categóricas **não foram preenchidos com a moda**. Preencher país com “Estados Unidos”, por exemplo, enviesaria o alvo, porque país é o fator que mais pesa no salário. O preenchimento usado foi a categoria **“Não informado”**.

### 3.4 Engenharia de recursos

- **Cargos macro:** dezenas de nomenclaturas viraram `Software Developer`, `Data Professional`, `Mobile Developer`, `Infra / DevOps / Security`, `Arquitetura / Gestão Técnica` e `Outros`.
- **Regime:** remoto, híbrido, presencial ou não informado.
- **Porte da empresa:** pequena, média, grande, muito grande ou não informado.
- **País:** os 20 mais frequentes foram mantidos; os demais viraram “Outros”.
- **Região:** agrupamento por mercado/custo de vida (América do Norte, Europa Ocidental, Europa Oriental, América Latina, Ásia do Sul, etc.).
- **Stack:** linguagens e frameworks viraram colunas 0/1 (possui ou não), evitando fingir ordem entre tecnologias. Exemplos: JavaScript, TypeScript, Python, Java, SQL, React, Node.js, Spring Boot, Django.

Não foi aplicada normalização/padronização numérica porque o algoritmo escolhido (Random Forest) não exige escala entre as variáveis.

### 3.5 O que a exploração mostrou

Na amostra final:

| Estatística | WorkExp (anos) | YearsCode (anos) | Salário anual (USD) |
|---|---|---|---|
| Média | 2,18 | 7,01 | 42.270 |
| Mediana | 2,00 | 6,00 | 35.220 |
| Mínimo | 1 | 1 | 1.000 |
| Máximo | 3 | 28 | 200.356 |

Distribuição de cargos macro:

- Software Developer: **1.793**
- Data Professional: **250**
- Infra / DevOps / Security: **134**
- Mobile Developer: **71**
- Arquitetura / Gestão Técnica: **36**
- Outros: **19**

A maioria do recorte é desenvolvedor de software, o que reflete a pesquisa do Stack Overflow.

**Correlação** (após a limpeza):

- `WorkExp` × salário: **0,10** (relação fraca — esperado, porque todos são júnior);
- `YearsCode` × salário: **0,28** (relação moderada: quem programa há mais tempo, mesmo com pouco tempo de emprego, tende a ganhar mais);
- `WorkExp` × `YearsCode`: **0,17**.

Os gráficos gerados no código (barras de cargo e regime, boxplot e histograma de salário, boxplot de salário por anos de experiência e heatmap de correlação) mostram salário assimétrico à direita: muita gente na faixa mais baixa/intermediária e uma cauda de salários altos, típica de mercados como Estados Unidos.

A leitura prática da EDA é: **dentro do grupo júnior, país/região explicam mais o salário do que “ter 2 ou 3 anos de carteira”.**

---

## 4. Modelo escolhido e processo de treinamento

### 4.1 Por que classificação e por que Random Forest

O problema foi modelado como **classificação** porque a pergunta do projeto é a faixa relativa (Alta ou Baixa), não o valor contínuo. Random Forest foi escolhido por três motivos:

1. Lida bem com **muitas variáveis categóricas** depois do One-Hot Encoding (país, região, escolaridade, cargo, stack);
2. É menos sensível a escala e a relações não lineares do que uma regressão logística simples;
3. Permite **interpretar importância das variáveis**, o que ajuda na discussão dos resultados em sala.

### 4.2 Variável alvo

1. Os dados foram separados em **70% treino** e **30% teste**.
2. A mediana salarial foi calculada **apenas no treino**: **US$ 35.162**.
3. Treino e teste receberam o rótulo Alta/Baixa com essa mesma mediana.

O split foi **estratificado** só para equilibrar as classes no teste. O rótulo oficial continua sendo o da mediana do treino, para não “olhar” o teste na definição do alvo.

Tamanhos obtidos:

- Treino: **1.612** registros (806 Alta e 806 Baixa);
- Teste: **691** registros (346 Alta e 345 Baixa).

### 4.3 Transformação e treino

As categóricas passaram por **One-Hot Encoding** (`handle_unknown="ignore"`), para não criar hierarquia falsa (por exemplo, fingir que “Brasil” é maior que “Índia”). As numéricas e as flags de tecnologia seguiram em paralelo, dentro de um `Pipeline` do scikit-learn.

Hiperparâmetros principais do `RandomForestClassifier`:

- `n_estimators=300`
- `max_depth=14`
- `min_samples_leaf=3`
- `random_state=0` (reprodutibilidade)

O fluxo de treino é o mesmo das aulas: `fit` no conjunto de treino e `predict` no teste.

---

## 5. Resultados e análise da performance

### 5.1 Métricas no conjunto de teste

| Métrica | Resultado |
|---|---|
| **Acurácia** | **87,84%** |
| Precisão — Alta | 88,76% |
| Precisão — Baixa | 86,97% |
| Recall — Alta | 86,71% |
| Recall — Baixa | 88,99% |
| F1 — Alta | 0,877 |
| F1 — Baixa | 0,880 |

**Matriz de confusão** (linhas = valor real; colunas = previsto):

|  | Previsto Baixa | Previsto Alta |
|---|---|---|
| **Real Baixa** | 307 | 38 |
| **Real Alta** | 46 | 300 |

Em 691 profissionais do teste, o modelo errou **84** casos e acertou **607**. Um classificador aleatório, com classes equilibradas, acertaria cerca de 50%. O ganho é relevante.

As duas classes têm desempenho parecido: o modelo não “chuta” só Alta ou só Baixa. Isso importa porque o alvo foi construído na mediana, então as classes já nasceram balanceadas.

### 5.2 Interpretação

As variáveis mais importantes do Random Forest foram, nesta ordem aproximada:

1. **Região — América do Norte**
2. **Região — Europa Ocidental**
3. **País — Estados Unidos**
4. **País — Outros**
5. **Região — Ásia do Sul**
6. **Anos de código (`YearsCode`)**
7. Em seguida: Europa Oriental, Índia, regime híbrido, Alemanha, América Latina, Ucrânia, porte da empresa, escolaridade.

Essa ordem é coerente com o mercado: um júnior nos Estados Unidos ou na Europa Ocidental tende a cair em **Alta**; um júnior na Índia, na Europa Oriental ou na América Latina tende a cair em **Baixa**, mesmo com stack parecida. O tempo de código pesa mais do que os 1–3 anos de emprego, o que combina com a correlação vista na EDA.

Stack e cargo entram no modelo, mas com peso menor. Para júnior, **localização e região salarial** dominam. Isso não significa que linguagem não importe na vida real; significa que, nesta base global, a diferença entre trabalhar nos EUA ou na Índia é maior do que a diferença entre usar Python ou Java.

### 5.3 Limitações

- A pesquisa é **autodeclarada**: salário e cargo podem ter erro de preenchimento, mesmo após a limpeza.
- Quem responde o Stack Overflow **não representa todo o mercado** (há viés para quem usa a plataforma).
- O recorte júnior (até 3 anos) reduz a amostra de 49 mil para cerca de 2,3 mil.
- A classe Alta/Baixa é **relativa à mediana do treino**, não a um piso nacional (por exemplo, salário mínimo brasileiro).
- País “Outros” mistura mercados diferentes, o que ainda gera um pouco de ruído.

### 5.4 Melhorias possíveis

- Comparar Random Forest com **Gradient Boosting** ou regressão logística, para ver se a acurácia se mantém.
- Trocar o alvo binário por **faixas** (baixo / médio / alto) ou por regressão do valor em dólares.
- Incluir poder de compra (salário ajustado por país) em vez de dólar nominal.
- Ajustar o corte de júnior para 2 anos e medir o impacto.
- Validação cruzada além de um único split 70/30.

---

## 6. Conclusão

O projeto cumpre o que foi pedido na atividade: problema definido, dataset próprio e com volume adequado, exploração com gráficos e tratamento de dados, modelo de classificação com treino/teste e avaliação por acurácia e matriz de confusão.

Com dados coerentes de profissionais júnior, o Random Forest atingiu **87,84% de acurácia** no teste. O padrão mais forte encontrado é o de **mercado geográfico**: a valorização inicial em TI depende sobretudo de região/país, e em segundo lugar do tempo de prática em programação.

O código está no arquivo `projeto.py` e a base utilizada é o `results.csv`.

---

## Referências

- Stack Overflow. *Developer Survey 2025*. Dados em `results.csv`.
- Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 2011.
- Material e práticas da disciplina (análise exploratória, tratamento de nulos/outliers, `train_test_split`, acurácia e matriz de confusão).
