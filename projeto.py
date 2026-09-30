# -*- coding: utf-8 -*-
"""
Projeto de Machine Learning - Classificação Binária de Remuneração Júnior (TI)

Objetivo: prever se a remuneração de profissionais em início de carreira
na área de tecnologia é "Alta" ou "Baixa", com base na mediana salarial
calculada exclusivamente no conjunto de treino.

Dataset: Stack Overflow Developer Survey 2025 (results.csv)
"""

# =============================================================================
# Importações (mesmo ecossistema usado em aula)
# =============================================================================
import pandas as pd  # Manipulação e análise de dados
import numpy as np  # Cálculos e arrays
import statistics as sts  # Mediana, desvio padrão etc.
import seaborn as sns  # Gráficos estatísticos
import matplotlib.pyplot as plt  # Gráficos

from sklearn.model_selection import train_test_split  # Treino e teste
from sklearn.preprocessing import OneHotEncoder  # Encoding sem hierarquia falsa
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # Modelo robusto a categóricas
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# =============================================================================
# 1. Carregamento dos dados
# =============================================================================
dataset = pd.read_csv("results.csv", low_memory=False)

print("Shape inicial:", dataset.shape)
print(dataset.head())

# Colunas relevantes para o problema
colunas_uteis = [
    "MainBranch",
    "Age",
    "EdLevel",
    "Employment",
    "WorkExp",
    "YearsCode",
    "DevType",
    "OrgSize",
    "RemoteWork",
    "Country",
    "Industry",
    "ICorPM",
    "LanguageHaveWorkedWith",
    "WebframeHaveWorkedWith",
    "ConvertedCompYearly",
]

dados = dataset[colunas_uteis].copy()
# Stack Overflow usa "NA" textual — padronizar como nulo real
dados = dados.replace("NA", np.nan)
print("\nColunas selecionadas:", dados.columns.tolist())
print(dados.shape)

# =============================================================================
# 2. Funções auxiliares de limpeza e engenharia de recursos
# =============================================================================

def converter_anos(valor):
    """Converte YearsCode / WorkExp (texto ou número) para float."""
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip()
    if texto in ("", "NA", "nan", "None"):
        return np.nan
    if "Less than" in texto:
        return 0.5
    if "More than" in texto:
        return 51.0
    try:
        return float(texto)
    except ValueError:
        return np.nan


def agrupar_cargo(dev_type):
    """
    Agrupa dezenas de nomenclaturas de cargo em categorias macro,
    reduzindo ruído nos dados.
    """
    if pd.isna(dev_type):
        return "Outros"
    cargo = str(dev_type).lower()

    if any(x in cargo for x in [
        "data scientist", "data engineer", "data or business analyst",
        "ai/ml", "applied scientist", "developer, ai",
        "database administrator",
    ]):
        return "Data Professional"

    if "mobile" in cargo:
        return "Mobile Developer"

    if any(x in cargo for x in [
        "devops", "cloud infrastructure", "system administrator",
        "security", "infosec", "sre",
    ]):
        return "Infra / DevOps / Security"

    if any(x in cargo for x in [
        "back-end", "front-end", "full-stack", "desktop",
        "embedded", "game", "qa or test",
    ]):
        return "Software Developer"

    if any(x in cargo for x in ["architect", "engineering manager"]):
        return "Arquitetura / Gestão Técnica"

    return "Outros"


def simplificar_remoto(valor):
    """Padroniza regime de trabalho em categorias claras."""
    if pd.isna(valor):
        return "Não informado"
    texto = str(valor).lower()
    if "remote" in texto and "hybrid" not in texto:
        return "Remoto"
    if "hybrid" in texto or "flexible" in texto or "your choice" in texto:
        return "Híbrido"
    if "in-person" in texto:
        return "Presencial"
    return "Não informado"


def simplificar_porte(valor):
    """Agrupa porte da empresa em faixas macro."""
    if pd.isna(valor):
        return "Não informado"
    texto = str(valor)
    if "Just me" in texto or "Less than 20" in texto:
        return "Pequena (até 19)"
    if "20 to 99" in texto or "100 to 499" in texto:
        return "Média (20-499)"
    if "500 to 999" in texto or "1,000 to 4,999" in texto:
        return "Grande (500-4999)"
    if "5,000" in texto or "10,000" in texto:
        return "Muito grande (5000+)"
    return "Não informado"


def agrupar_regiao(pais):
    """Agrupa países em regiões de custo de vida / mercado de TI."""
    if pd.isna(pais) or str(pais) == "Não informado":
        return "Não informado"
    nome = str(pais)
    grupos = {
        "América do Norte": [
            "United States of America", "Canada", "Mexico",
        ],
        "Europa Ocidental": [
            "Germany", "United Kingdom of Great Britain and Northern Ireland",
            "France", "Netherlands", "Switzerland", "Sweden", "Austria",
            "Belgium", "Denmark", "Norway", "Finland", "Ireland", "Italy",
            "Spain", "Portugal", "Luxembourg",
        ],
        "Europa Oriental": [
            "Ukraine", "Poland", "Romania", "Czech Republic", "Czechia",
            "Hungary", "Bulgaria", "Slovakia", "Croatia", "Serbia",
            "Lithuania", "Latvia", "Estonia", "Belarus", "Republic of Moldova",
            "Bosnia and Herzegovina", "North Macedonia", "Albania",
        ],
        "América Latina": [
            "Brazil", "Argentina", "Colombia", "Chile", "Peru", "Uruguay",
            "Ecuador", "Venezuela, Bolivarian Republic of...", "Costa Rica",
            "Guatemala", "Bolivia", "Paraguay",
        ],
        "Ásia do Sul": [
            "India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal",
        ],
        "Ásia Oriental": [
            "China", "Japan", "South Korea", "Taiwan", "Hong Kong (S.A.R.)",
            "Viet Nam", "Indonesia", "Thailand", "Malaysia", "Philippines",
            "Singapore",
        ],
        "Oceania": ["Australia", "New Zealand"],
        "Oriente Médio / África": [
            "Israel", "Turkey", "Iran, Islamic Republic of...", "Egypt",
            "South Africa", "Nigeria", "Kenya", "Morocco", "Tunisia",
            "United Arab Emirates", "Saudi Arabia", "Ghana",
        ],
    }
    for regiao, paises in grupos.items():
        if nome in paises:
            return regiao
    return "Outros"


def possui_tecnologia(celula, tecnologia):
    """Verifica se a tecnologia aparece na lista separada por ';'."""
    if pd.isna(celula):
        return 0
    itens = [item.strip().lower() for item in str(celula).split(";")]
    return int(tecnologia.lower() in itens)


# Linguagens e frameworks mais relevantes para o mercado júnior
LINGUAGENS_TOP = [
    "JavaScript",
    "TypeScript",
    "Python",
    "Java",
    "C#",
    "SQL",
    "HTML/CSS",
    "Go",
    "PHP",
    "Kotlin",
    "Swift",
    "C++",
    "Rust",
    "Bash/Shell (all shells)",
]

FRAMEWORKS_TOP = [
    "React",
    "Node.js",
    "Angular",
    "Vue.js",
    "Next.js",
    "Django",
    "Flask",
    "Spring Boot",
    ".NET",
    "ASP.NET CORE",
    "Laravel",
    "Express",
]

# =============================================================================
# 3. Pré-processamento e filtros (foco em juniores)
# =============================================================================

# Converter anos de experiência e de código
dados["WorkExp"] = dados["WorkExp"].apply(converter_anos)
dados["YearsCode"] = dados["YearsCode"].apply(converter_anos)

# Salário anual convertido (USD)
dados["ConvertedCompYearly"] = pd.to_numeric(
    dados["ConvertedCompYearly"], errors="coerce"
)

print("\nValores nulos antes do tratamento:")
print(dados.isnull().sum())

# Manter apenas profissionais empregados / freelancers
dados = dados[dados["Employment"].isin([
    "Employed",
    "Independent contractor, freelancer, or self-employed",
])]

# Remover cargos fora do escopo de TI ativa / início de carreira
cargos_fora_escopo = [
    "Student",
    "Retired",
    "Senior executive (C-suite, VP, etc.)",
    "Founder, technology or otherwise",
    "Project manager",
    "Product manager",
    "Engineering manager",
    "Academic researcher",
    "Financial analyst or engineer",
    "UX, Research Ops or UI design professional",
    "Other (please specify):",
]
dados = dados[~dados["DevType"].isin(cargos_fora_escopo)]
dados = dados[dados["DevType"].notna()]

# Pessoas gestoras com poucos anos distorcem o recorte júnior
dados = dados[dados["ICorPM"].fillna("Individual contributor") != "People manager"]

# Filtrar início de carreira: até 3 anos de experiência profissional
MAX_EXP_JUNIOR = 3
dados = dados[
    (dados["WorkExp"].notna())
    & (dados["WorkExp"] >= 0)
    & (dados["WorkExp"] <= MAX_EXP_JUNIOR)
]

# Exigir salário informado e positivo
dados = dados[
    (dados["ConvertedCompYearly"].notna())
    & (dados["ConvertedCompYearly"] > 0)
]
print("\nShape após filtro de juniores com salário:", dados.shape)

# Salários irreais (ex.: US$ 1 em países de alto custo) — piso anual coerente
SALARIO_MINIMO = 1000
n_antes = len(dados)
dados = dados[dados["ConvertedCompYearly"] >= SALARIO_MINIMO]
print(f"Removidos por salário < US$ {SALARIO_MINIMO}: {n_antes - len(dados)}")

# Outliers altos: IQR (mais estável que média + 2 desvios quando há milhões)
q1_salario = dados["ConvertedCompYearly"].quantile(0.25)
q3_salario = dados["ConvertedCompYearly"].quantile(0.75)
iqr_salario = q3_salario - q1_salario
limite_superior = q3_salario + (3 * iqr_salario)
print(f"\nQ1 salarial: {q1_salario:.2f}")
print(f"Q3 salarial: {q3_salario:.2f}")
print(f"IQR: {iqr_salario:.2f}")
print(f"Limite superior (Q3 + 3*IQR): {limite_superior:.2f}")

n_antes = len(dados)
dados = dados[dados["ConvertedCompYearly"] <= limite_superior]
print(f"Removidos por salário acima do IQR: {n_antes - len(dados)}")
print("Shape após tratamento de salário:", dados.shape)

# YearsCode: nulos e valores incompatíveis com júnior / com WorkExp
mediana_anos = sts.median(dados["YearsCode"].dropna())
dados.loc[dados["YearsCode"].isna(), "YearsCode"] = mediana_anos
# 100 anos de código em um júnior é erro de preenchimento
dados.loc[dados["YearsCode"] > 30, "YearsCode"] = mediana_anos
# Não faz sentido ter menos anos de código do que anos de trabalho na área
incoerente_anos = dados["YearsCode"] < dados["WorkExp"]
dados.loc[incoerente_anos, "YearsCode"] = dados.loc[incoerente_anos, "WorkExp"]

# Categóricas: não usar moda (evita empurrar nulos para EUA / empresa média)
for coluna in ["EdLevel", "RemoteWork", "OrgSize", "Country", "Industry", "Age"]:
    dados[coluna] = dados[coluna].fillna("Não informado")

# Engenharia de recursos: cargo macro, remoto e porte
dados["CargoMacro"] = dados["DevType"].apply(agrupar_cargo)
dados["RegimeTrabalho"] = dados["RemoteWork"].apply(simplificar_remoto)
dados["PorteEmpresa"] = dados["OrgSize"].apply(simplificar_porte)

# Reduzir cardinalidade de Country / Industry (top categorias + Outros)
TOP_PAISES = dados["Country"].value_counts().head(20).index.tolist()
dados["Pais"] = dados["Country"].apply(
    lambda x: x if x in TOP_PAISES else "Outros"
)
dados["Regiao"] = dados["Country"].apply(agrupar_regiao)

TOP_INDUSTRIAS = dados["Industry"].value_counts().head(10).index.tolist()
dados["Industria"] = dados["Industry"].apply(
    lambda x: x if x in TOP_INDUSTRIAS else "Outros"
)

# One-hot manual das tecnologias (lista multi-escolha do Stack Overflow)
for linguagem in LINGUAGENS_TOP:
    nome_col = "Lang_" + linguagem.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    dados[nome_col] = dados["LanguageHaveWorkedWith"].apply(
        lambda x, tech=linguagem: possui_tecnologia(x, tech)
    )

for framework in FRAMEWORKS_TOP:
    nome_col = "Fw_" + framework.replace(".", "_").replace(" ", "_").replace("/", "_")
    dados[nome_col] = dados["WebframeHaveWorkedWith"].apply(
        lambda x, tech=framework: possui_tecnologia(x, tech)
    )

print("\nDistribuição de cargos macro:")
print(dados["CargoMacro"].value_counts())

print("\nValores nulos após tratamento:")
print(dados.isnull().sum().sort_values(ascending=False).head(15))

# =============================================================================
# 4. Análise exploratória dos dados (EDA)
# =============================================================================
print("\n===== ANÁLISE EXPLORATÓRIA =====")
print(dados[["WorkExp", "YearsCode", "ConvertedCompYearly"]].describe())

# Distribuição de cargos
agrupado_cargo = dados.groupby(["CargoMacro"]).size()
print("\nCargos macro:")
print(agrupado_cargo)
agrupado_cargo.plot.bar(figsize=(10, 5), title="Distribuição de Cargos Macro (Juniores)")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

# Regime de trabalho
agrupado_remoto = dados.groupby(["RegimeTrabalho"]).size()
agrupado_remoto.plot.bar(figsize=(8, 4), title="Regime de Trabalho")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

# Boxplot e histograma do salário
plt.figure(figsize=(8, 4))
sns.boxplot(x=dados["ConvertedCompYearly"])
plt.title("Boxplot - Salário Anual (USD) - Juniores")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(dados["ConvertedCompYearly"], kde=True)
plt.title("Histograma - Salário Anual (USD) - Juniores")
plt.xlabel("ConvertedCompYearly")
plt.tight_layout()
plt.show()

# Experiência vs salário
plt.figure(figsize=(8, 4))
sns.boxplot(data=dados, x="WorkExp", y="ConvertedCompYearly")
plt.title("Salário por Anos de Experiência (Júnior)")
plt.tight_layout()
plt.show()

# Correlação entre variáveis numéricas
corr = dados[["WorkExp", "YearsCode", "ConvertedCompYearly"]].corr()
print("\nCorrelação:")
print(corr)
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="Blues")
plt.title("Correlação - Variáveis Numéricas")
plt.tight_layout()
plt.show()

# =============================================================================
# 5. Preparação do modelo (variável alvo binária)
# =============================================================================

# Features usadas no modelo
cols_linguagens = [c for c in dados.columns if c.startswith("Lang_")]
cols_frameworks = [c for c in dados.columns if c.startswith("Fw_")]

features_categoricas = [
    "EdLevel",
    "Age",
    "CargoMacro",
    "RegimeTrabalho",
    "PorteEmpresa",
    "Pais",
    "Regiao",
    "Industria",
]
features_numericas = ["WorkExp", "YearsCode"] + cols_linguagens + cols_frameworks

X = dados[features_categoricas + features_numericas].copy()
y_salario = dados["ConvertedCompYearly"].copy()

# Split 70/30 estratificado só para equilibrar o teste (rótulo final usa a mediana do treino)
alvo_temporario = np.where(
    y_salario >= sts.median(y_salario), "Alta", "Baixa"
)
X_treinamento, X_teste, y_salario_treino, y_salario_teste = train_test_split(
    X,
    y_salario,
    test_size=0.3,
    random_state=0,
    stratify=alvo_temporario,
)

# IMPORTANTE: mediana calculada APENAS no conjunto de treino
mediana_treino = sts.median(y_salario_treino)
print(f"\nMediana salarial do TREINO (USD): {mediana_treino:.2f}")

# Variável alvo binária: Alta (>= mediana) ou Baixa (< mediana)
y_treinamento = np.where(y_salario_treino >= mediana_treino, "Alta", "Baixa")
y_teste = np.where(y_salario_teste >= mediana_treino, "Alta", "Baixa")

print("Distribuição alvo (treino):")
print(pd.Series(y_treinamento).value_counts())
print("Distribuição alvo (teste):")
print(pd.Series(y_teste).value_counts())

# =============================================================================
# 6. Treinamento — Random Forest + One-Hot Encoding
# =============================================================================

# Compatibilidade: sklearn novo usa sparse_output; versões antigas usam sparse
try:
    encoder_categorico = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder_categorico = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessador = ColumnTransformer(
    transformers=[
        ("cat", encoder_categorico, features_categoricas),
        ("num", "passthrough", features_numericas),
    ]
)

modelo = Pipeline(
    steps=[
        ("preprocessamento", preprocessador),
        (
            "classificador",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=14,
                min_samples_leaf=3,
                random_state=0,
                n_jobs=-1,
            ),
        ),
    ]
)

modelo.fit(X_treinamento, y_treinamento)
print("\nModelo Random Forest treinado com sucesso.")

# =============================================================================
# 7. Avaliação do modelo
# =============================================================================
previsoes = modelo.predict(X_teste)

acertos = accuracy_score(y_teste, previsoes)
print(f"\nAcurácia: {acertos:.4f}")

confusao = confusion_matrix(y_teste, previsoes, labels=["Baixa", "Alta"])
print("\nMatriz de confusão:")
print(confusao)

print("\nRelatório de classificação:")
print(classification_report(y_teste, previsoes, digits=4))

plt.figure(figsize=(8, 6))
sns.heatmap(
    confusao,
    annot=True,
    cmap="Greens",
    cbar=False,
    fmt="d",
    xticklabels=["Baixa", "Alta"],
    yticklabels=["Baixa", "Alta"],
)
plt.title("Matriz de Confusão - Remuneração Júnior (Alta x Baixa)")
plt.xlabel("Previsão")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# Importância das features (após One-Hot)
ohe = modelo.named_steps["preprocessamento"].named_transformers_["cat"]
try:
    nomes_cat = ohe.get_feature_names_out(features_categoricas).tolist()
except AttributeError:
    nomes_cat = ohe.get_feature_names(features_categoricas).tolist()
nomes_features = nomes_cat + features_numericas
importancias = modelo.named_steps["classificador"].feature_importances_

ranking = (
    pd.DataFrame({"Feature": nomes_features, "Importancia": importancias})
    .sort_values("Importancia", ascending=False)
    .head(20)
)
print("\nTop 20 features mais importantes:")
print(ranking)

plt.figure(figsize=(10, 6))
sns.barplot(data=ranking, x="Importancia", y="Feature")
plt.title("Top 20 Features Mais Importantes - Random Forest")
plt.tight_layout()
plt.show()

# =============================================================================
# 8. Interpretação rápida / possíveis melhorias
# =============================================================================
print(
    """
===== INTERPRETAÇÃO =====
- O modelo classifica juniores em remuneração Alta ou Baixa com base na mediana do treino.
- Features como País, Porte da Empresa, Regime de Trabalho, Cargo e Stack tendem a
  explicar boa parte da valorização inicial no mercado de TI.
- Melhorias possíveis: comparar com outros algoritmos (ex.: Gradient Boosting),
  balancear classes se necessário, incluir mais tecnologias ou ajustar o corte
  de experiência júnior (WorkExp <= 2).
"""
)
