import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from modeling.modeling_treatments import converter_para_numerico, truncar_contagens
from report import functions as f

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
 
# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
ANOS_TREINO = [2023, 2024]
ANO_TESTE = 2025
 
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
RANDOM_STATE = 0
PESO_CORTE = 2500  # baixo peso: PESO < 2500 g
TARGET = "PESO_BAIXO"
 
NUMERICAS = ["IDADEPAI", "IDADEMAE"]
CATEGORICAS = [
    "ESTCIVMAE", "ESCMAE", "RACACORMAE", "QTDFILVIVO", "QTDFILMORT",
    "QTDGESTANT", "QTDPARTNOR", "QTDPARTCES", "GRAVIDEZ",
]
COLUNAS_TRUNCAR = ["QTDFILVIVO", "QTDFILMORT", "QTDGESTANT", "QTDPARTNOR", "QTDPARTCES"]
TRUNC_MAX = 10.0
 

dfs = {}
for ano in ANOS_TREINO + [ANO_TESTE]:
    arquivo = DATA_DIR / f"sinasc_{ano}.parquet"
 
    if not arquivo.exists():
        f.download_sinasc_data(ano, salvar_parquet=str(arquivo))
 
    df = pl.read_parquet(arquivo)
    df = (
        df.pipe(f.remove_unused_features)
          .pipe(f.create_features)
          .pipe(f.optimize_dtypes)
          .pipe(f.set_ignored_to_null)
    )
    dfs[ano] = df
    print(f"{ano}: {df.shape}")
 
df_treino = pl.concat([dfs[a] for a in ANOS_TREINO], how="diagonal_relaxed")
df_teste = dfs[ANO_TESTE]
 
print("\nTreino/validação:", df_treino.shape)
print("Teste:", df_teste.shape)
print(df_treino.head())
 


FEATURES = NUMERICAS + CATEGORICAS
faltando = [c for c in FEATURES if c not in df_treino.columns]
if faltando:
    print(f"\n[AVISO] Variáveis ausentes após o pipeline (ignoradas): {faltando}")
features = [c for c in FEATURES if c in df_treino.columns]
numericas = [c for c in NUMERICAS if c in features]
categoricas = [c for c in CATEGORICAS if c in features]
truncar = [c for c in COLUNAS_TRUNCAR if c in features]
outras_cat = [c for c in categoricas if c not in truncar]
 
dados = {}
for nome, df in [("treino", df_treino), ("teste", df_teste)]:
    df = (
        df.with_columns(pl.col("PESO").cast(pl.Float64, strict=False))
          .filter(pl.col("PESO") > 0)
          .with_columns((pl.col("PESO") < PESO_CORTE).cast(pl.Int8).alias(TARGET))
    )
    pdf = df.select(features + [TARGET]).to_pandas()
 
    dados[nome] = (pdf[features], pdf[TARGET])
 
X, y = dados["treino"]
X_test, y_test = dados["teste"]
 
print(f"\nX treino+val: {X.shape} | prevalência de baixo peso: {y.mean():.2%}")
print(f"X teste:      {X_test.shape} | prevalência de baixo peso: {y_test.mean():.2%}")
print("\nValores ausentes (%):")
print((X.isna().mean() * 100).round(2))
 


X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTreino: {X_train.shape[0]:,} | Validação: {X_val.shape[0]:,}")
 
# Pipeline: pré-processamento + LightGBM

tratamento_tipos = FunctionTransformer(
    converter_para_numerico, feature_names_out="one-to-one"
)
 
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
])
 
contagens_transformer = Pipeline(steps=[
    ("truncation", FunctionTransformer(
        truncar_contagens, kw_args={"maximo": TRUNC_MAX}, feature_names_out="one-to-one"
    )),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False)),
])
 
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False)),
])
 
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numericas),
    ("contagens", contagens_transformer, truncar),
    ("cat", categorical_transformer, outras_cat),
])
 
 
pipe = Pipeline(steps=[
    ("tipos", tratamento_tipos),
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(verbose=-1, random_state=RANDOM_STATE)),
])
 
pipe.fit(X_train, y_train)
print("\nTreino concluído.")
 
nomes = pipe.named_steps["preprocessor"].get_feature_names_out()
imp = pd.Series(pipe.named_steps["classifier"].feature_importances_, index=nomes)
print("\nTop 15 variáveis por importância:")
print(imp.sort_values(ascending=False).head(15))
 
 
p_val = pipe.predict_proba(X_val)[:, 1]
p_test = pipe.predict_proba(X_test)[:, 1]
 
limiares = np.linspace(p_val.min(), p_val.max(), num=100)
f1s = [f1_score(y_val, (p_val > t).astype(int)) for t in limiares]
limiar = float(limiares[int(np.argmax(f1s))])
 
# Métricas
cortes = [0, 0.3, 0.6, 0.8, 1.0001]
rotulos = ["Baixo [0-30%[", "Moderado [30-60%[", "Alto [60-80%[", "Muito alto [80-100%]"]
 
resultados = {}
for nome, y_true, y_prob in [
    ("VALIDAÇÃO (2023-2024)", y_val, p_val),
    (f"TESTE FINAL ({ANO_TESTE})", y_test, p_test),
]:
    y_pred = (y_prob > limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
 
    m = {
        "n": int(len(y_true)),
        "prevalencia": float(np.mean(y_true)),
        "limiar": limiar,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "especificidade": float(tn / (tn + fp)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    resultados[nome] = m
 
    print(f"\n{'=' * 70}\n{nome}  (n={m['n']:,}, prevalência={m['prevalencia']:.2%})\n{'=' * 70}")
    print(f"Limiar: {limiar:.4f}")
    print(f"ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | Brier: {m['brier']:.4f}")
    print(f"Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | "
          f"F1: {m['f1']:.4f} | Especificidade: {m['especificidade']:.4f}")
    print("\n" + classification_report(y_true, y_pred, digits=3))
    print("Matriz de confusão [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_true, y_pred))
 
    faixas = pd.DataFrame({"y": np.asarray(y_true), "p": y_prob})
    faixas["faixa"] = pd.cut(faixas["p"], bins=cortes, labels=rotulos, right=False)
    tab = (
        faixas.groupby("faixa", observed=False)
              .agg(Total=("y", "size"), BaixoPeso=("y", "sum"))
              .reset_index()
    )
    tab["Percentual"] = (100 * tab["BaixoPeso"] / tab["Total"]).round(1)
    print("\nFaixas de risco:")
    print(tab.to_string(index=False))
 
print("\nComparação validação x teste:")
print(pd.DataFrame(resultados).T)
 
 
# Salvar modelo e métricas
joblib.dump(
    {"pipeline": pipe, "limiar": limiar, "features": features},
    OUTPUT_DIR / "model_lbw.joblib",
)
with open(OUTPUT_DIR / "metricas.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, indent=2, ensure_ascii=False)
 
print(f"\nModelo e métricas salvos em {OUTPUT_DIR.resolve()}")
 