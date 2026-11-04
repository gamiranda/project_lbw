"""
Tratamentos de dados usados dentro do pipeline do modelo (FunctionTransformer).
 
Ficam em um módulo próprio (e não no script de treino) para que o modelo salvo com
joblib possa ser carregado em qualquer outro script, desde que este arquivo esteja
no PYTHONPATH.
"""
 
import pandas as pd
from pandas.api.types import is_numeric_dtype
 
 
def converter_para_numerico(X: pd.DataFrame) -> pd.DataFrame:
    """Converte todas as colunas para float64; valores não numéricos viram NaN."""
    X = X.copy()
    for c in X.columns:
        s = X[c]
        if not is_numeric_dtype(s):
            s = pd.to_numeric(s.astype("object"), errors="coerce")
        X[c] = s.astype("float64")
    return X
 
 
def truncar_contagens(X: pd.DataFrame, maximo: float = 10.0) -> pd.DataFrame:
    """Limita (trunca) os valores das colunas em `maximo`; NaN é preservado."""
    return X.clip(upper=maximo)