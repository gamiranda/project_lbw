import io
import zipfile
from urllib.request import urlopen

import plotnine as pn
import polars as pl


def download_sinasc_data(ano: int, salvar_parquet: str | None = None) -> pl.DataFrame | None:
    
    url = f"https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINASC/csv/SINASC_{ano}_csv.zip"

    with urlopen(url) as resp:
        conteudo = resp.read()

    with zipfile.ZipFile(io.BytesIO(conteudo)) as zf:
        nome_csv = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(nome_csv) as f:
            df = pl.read_csv(
                f.read(),
                separator=";",
                encoding="utf8-lossy",
                infer_schema=False, 
            )

    if salvar_parquet:
        df.write_parquet(salvar_parquet)
        return None

    return df


columns_to_remove = [
    "contador", "ORIGEM", "NUMEROLOTE", "VERSAOSIST", "DTCADASTRO", "DTRECEBIM",
    "DTRECORIGA", "DIFDATA", "OPORT_DN", "DTDECLARAC", "TPFUNCRESP", "TPDOCRESP",
    "STDNEPIDEM", "STDNNOVA", "CODPAISRES", "NATURALMAE", "CODMUNNATU", "CODUFNATU",
    "SERIESCMAE", "DTNASCMAE", "DTULTMENST", "TPMETESTIM", "CODOCUPMAE", "CODANOMAL",
    "HORANASC","ESCMAE2010","ESCMAE","APGAR1","APGAR5","STCESPARTO"
]

def remove_unused_features(df: pl.DataFrame, colunas: list[str] = columns_to_remove) -> pl.DataFrame:
    
    return df.drop(colunas, strict=False)


def create_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:

    UF_BY_CODE = {
        12: "AC", 27: "AL", 16: "AP", 13: "AM", 29: "BA", 23: "CE", 53: "DF",
        32: "ES", 52: "GO", 21: "MA", 51: "MT", 50: "MS", 31: "MG", 15: "PA",
        25: "PB", 41: "PR", 26: "PE", 22: "PI", 24: "RN", 43: "RS", 33: "RJ",
        11: "RO", 14: "RR", 42: "SC", 35: "SP", 28: "SE", 17: "TO",
    }

    _REGIONS = {
        "North": [11, 12, 13, 14, 15, 16, 17],
        "Northeast": [21, 22, 23, 24, 25, 26, 27, 28, 29],
        "Southeast": [31, 32, 33, 35],
        "South": [41, 42, 43],
        "Midwest": [50, 51, 52, 53],
    }
    REGIONS_BY_CODE = {cod: reg for reg, cods in _REGIONS.items() for cod in cods}

    return (
        df.with_columns(
            pl.col("DTNASC").cast(pl.Utf8).str.pad_start(8, "0").str.to_date("%d%m%Y", strict=False).alias("DTNASC_padded"),
            pl.col("CODMUNRES").cast(pl.Utf8).str.slice(0, 2).cast(pl.Int32, strict=False).alias("STATE_CODE"),
        ).with_columns(
            pl.col("STATE_CODE").replace_strict(UF_BY_CODE, default=None, return_dtype=pl.Utf8).alias("STATE"),
            pl.col("STATE_CODE").replace_strict(REGIONS_BY_CODE, default=None, return_dtype=pl.Utf8).alias("REGION"),
            #pl.col("DTNASC_padded").dt.year().alias("YEAR"),
            #pl.col("DTNASC_padded").dt.month().alias("MONTH"),
            pl.col("DTNASC_padded").dt.day().alias("DAY"),
            (pl.col("DTNASC_padded").dt.weekday() % 7 + 1).alias("DAY_WEEK"),
            pl.col("DTNASC_padded").dt.quarter().alias("QUARTER"),
            pl.col("DTNASC_padded").dt.strftime("%Y-%m").alias("YEAR_MONTH"),
            pl.when(pl.col("PESO").cast(pl.Int32, strict=False) <= 2500).then(1).otherwise(0).alias("LOWBIRTHWEIGHT"),
        ).drop("STATE_CODE", "DTNASC")
    )


def optimize_dtypes(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    
    DTYPES = {
        "IDADEMAE": pl.UInt8, "IDADEPAI": pl.UInt8,
        "QTDFILVIVO": pl.UInt8, "QTDFILMORT": pl.UInt8, "QTDGESTANT": pl.UInt8,
        "QTDPARTNOR": pl.UInt8, "QTDPARTCES": pl.UInt8,
        "SEMAGESTAC": pl.UInt8, "CONSPRENAT": pl.UInt8, "MESPRENAT": pl.UInt8,
        "PESO": pl.UInt16,

        "SEXO": pl.UInt8, "RACACOR": pl.UInt8, "RACACORMAE": pl.UInt8,
        "ESTCIVMAE": pl.UInt8, "ESCMAEAGR1": pl.UInt8,
        "GRAVIDEZ": pl.UInt8, "GESTACAO": pl.UInt8, "PARTO": pl.UInt8,
        "CONSULTAS": pl.UInt8, "LOCNASC": pl.UInt8,
        "TPAPRESENT": pl.UInt8, "STTRABPART": pl.UInt8,
        "TPNASCASSI": pl.UInt8, "TPROBSON": pl.UInt8,
        "KOTELCHUCK": pl.UInt8, "IDANOMAL": pl.UInt8, "PARIDADE": pl.UInt8,
        
        "CODMUNRES": pl.Categorical, "CODMUNNASC": pl.Categorical, "CODESTAB": pl.Categorical,
          
        "YEAR": pl.Int16, "MONTH": pl.UInt8, "DAY": pl.UInt8,
        "DAY_WEEK": pl.UInt8, "QUARTER": pl.UInt8, "LOWBIRTHWEIGHT": pl.UInt8,
        "YEAR_MONTH": pl.Categorical, "STATE": pl.Categorical, "REGION": pl.Categorical,
    }
    

    schema = df.collect_schema()
    exprs = []

    for c, t in DTYPES.items():
        if c not in schema:
            continue

        col = pl.col(c)

        # strings: remove espaços e converte vazio em null
        if schema[c] == pl.String:
            col = col.str.strip_chars()
            col = pl.when(col == "").then(None).otherwise(col)

        exprs.append(col.cast(t).alias(c))

    return df.with_columns(exprs)
    

PALETA = ["#374e55", "#df8f44", "#00a1d5", "#b24745", "#79af97", "#6a6599", "#80796b"]

REGIAO_PT = {
    "North": "Norte",
    "Northeast": "Nordeste",
    "Midwest": "Centro-Oeste",
    "Southeast": "Sudeste",
    "South": "Sul",
}
ORDEM_REGIAO = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

ORDEM_FAIXA_IDADEMAE = ["Até 14 anos", "15 a 19 anos", "20 a 34 anos", "35 anos ou mais"]
ORDEM_FAIXA_PESO = [
    "Menos de 1.500 g", "1.500 a 2.499 g", "2.500 a 2.999 g",
    "3.000 a 3.999 g", "4.000 g ou mais",
]

MAPA_RACACOR = {1: "Branca", 2: "Preta", 3: "Amarela", 4: "Parda", 5: "Indígena", 9: "Ignorado"}
ORDEM_RACACOR = ["Branca", "Preta", "Amarela", "Parda", "Indígena", "Ignorado"]


MAPA_ESCOLARIDADE_AGR = {
    0: "Até fundamental I", 1: "Até fundamental I", 2: "Até fundamental I", 10: "Até fundamental I",
    3: "Fundamental II", 4: "Fundamental II", 11: "Fundamental II",
    5: "Ensino médio", 6: "Ensino médio", 12: "Ensino médio",
    7: "Superior", 8: "Superior",
    9: "Ignorado",
}
ORDEM_ESCOLARIDADE_AGR = ["Até fundamental I", "Fundamental II", "Ensino médio", "Superior", "Ignorado"]


IGNORADOS_NUMERICOS = {
    "IDADEMAE": 99, "IDADEPAI": 99,
    "QTDFILVIVO": 99, "QTDFILMORT": 99, "QTDGESTANT": 99,
    "QTDPARTNOR": 99, "QTDPARTCES": 99,
    "SEMAGESTAC": 99, "CONSPRENAT": 99, "MESPRENAT": 99,
    "PESO": 9999,
}


def fmt_int(x) -> str:
    return f"{x:,.0f}".replace(",", ".")


def fmt_num(x, casas: int = 1) -> str:
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x, casas: int = 1) -> str:
    return fmt_num(x, casas) + "%"


def set_ignored_to_null(df: pl.DataFrame, ignorados: dict = IGNORADOS_NUMERICOS) -> pl.DataFrame:
    
    cols = df.collect_schema().names()
    return df.with_columns(
        pl.when(pl.col(c) == v).then(None).otherwise(pl.col(c)).alias(c)
        for c, v in ignorados.items() if c in cols
    )


def rotular(df, variavel, mapa, nome_novo, valor_nulo=9):
    
    ignorado = mapa.get(valor_nulo, "Ignorado")
    return df.with_columns(
        pl.col(variavel).cast(pl.Int64, strict=False).fill_null(valor_nulo)
        .replace_strict(mapa, default=ignorado, return_dtype=pl.Utf8)
        .alias(nome_novo)
    )


def _ind(condicao: pl.Expr, valido: pl.Expr) -> pl.Expr:
    
    return pl.when(valido).then(condicao.cast(pl.UInt8)).otherwise(None)


def create_report_features(df: pl.DataFrame) -> pl.DataFrame:
    
    idade, peso = pl.col("IDADEMAE"), pl.col("PESO")

    df = df.with_columns(
        pl.col("REGION").cast(pl.Utf8).replace_strict(REGIAO_PT, default=None, return_dtype=pl.Utf8).alias("REGIAO"),
        pl.col("STATE").cast(pl.Utf8).alias("UF"),
        pl.when(idade <= 14).then(pl.lit(ORDEM_FAIXA_IDADEMAE[0]))
        .when(idade <= 19).then(pl.lit(ORDEM_FAIXA_IDADEMAE[1]))
        .when(idade <= 34).then(pl.lit(ORDEM_FAIXA_IDADEMAE[2]))
        .when(idade >= 35).then(pl.lit(ORDEM_FAIXA_IDADEMAE[3]))
        .otherwise(None).alias("FAIXA_IDADEMAE"),
        pl.when(peso < 1500).then(pl.lit(ORDEM_FAIXA_PESO[0]))
        .when(peso < 2500).then(pl.lit(ORDEM_FAIXA_PESO[1]))
        .when(peso < 3000).then(pl.lit(ORDEM_FAIXA_PESO[2]))
        .when(peso < 4000).then(pl.lit(ORDEM_FAIXA_PESO[3]))
        .when(peso >= 4000).then(pl.lit(ORDEM_FAIXA_PESO[4]))
        .otherwise(None).alias("FAIXA_PESO"),
        # indicadores 0/1 (nulo = sem informação válida)
        _ind(pl.col("PARTO") == 2, pl.col("PARTO").is_in([1, 2])).alias("IND_CESAREA"),
        _ind(pl.col("GESTACAO") <= 4, pl.col("GESTACAO").is_in([1, 2, 3, 4, 5, 6])).alias("IND_PREMATURO"),
        _ind(peso < 2500, peso.is_not_null()).alias("IND_BAIXO_PESO"),
        _ind(idade <= 19, idade.is_not_null()).alias("IND_MAE_ADOLESCENTE"),
        _ind(idade >= 35, idade.is_not_null()).alias("IND_MAE_35MAIS"),
        _ind(pl.col("GRAVIDEZ") >= 2, pl.col("GRAVIDEZ").is_in([1, 2, 3])).alias("IND_MULTIPLA"),
        _ind(pl.col("CONSULTAS") == 4, pl.col("CONSULTAS").is_in([1, 2, 3, 4])).alias("IND_PRENATAL_7MAIS"),
        _ind(pl.col("KOTELCHUCK") >= 4, pl.col("KOTELCHUCK").is_in([1, 2, 3, 4, 5])).alias("IND_PRENATAL_ADEQUADO"),
        _ind(pl.col("LOCNASC") == 1, pl.col("LOCNASC").is_in([1, 2, 3, 4, 5])).alias("IND_HOSPITAL"),
        _ind(pl.col("IDANOMAL") == 1, pl.col("IDANOMAL").is_in([1, 2])).alias("IND_ANOMALIA"),
        _ind(
            pl.col("CODMUNRES").cast(pl.Utf8) != pl.col("CODMUNNASC").cast(pl.Utf8),
            pl.col("CODMUNRES").is_not_null() & pl.col("CODMUNNASC").is_not_null(),
        ).alias("IND_PARTO_FORA_MUN_RESIDENCIA"),
    )
    df = rotular(df, "RACACORMAE", MAPA_RACACOR, "RACACOR_MAE")
    df = rotular(df, "ESCMAEAGR1", MAPA_ESCOLARIDADE_AGR, "ESCOLARIDADE_MAE")
    return df


def cap_count(df, variavel, teto, valor_nulo=9):
    
    return df.with_columns(
        pl.when(pl.col(variavel).is_null()).then(valor_nulo)
        .when(pl.col(variavel) >= teto).then(teto)
        .otherwise(pl.col(variavel).cast(pl.Int64))
        .alias(variavel)
    )


def _fmt_celula(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return fmt_int(v)
    if isinstance(v, float):
        return fmt_num(v, 1)
    return str(v).replace("|", "\\|")


def df_to_markdown(df: pl.DataFrame, legenda: str | None = None, direita: list[str] | None = None) -> str:
    
    cols = df.columns
    direita = set(direita or [])
    alinhar_dir = [df.schema[c].is_numeric() or c in direita for c in cols]
    linhas = [[_fmt_celula(v) for v in row] for row in df.iter_rows()]
    larg = [min(max([len(c)] + [len(l[i]) for l in linhas]), 60) for i, c in enumerate(cols)]

    cab = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(
        ("-" * (max(w, 3) - 1) + ":") if d else (":" + "-" * (max(w, 3) - 1))
        for w, d in zip(larg, alinhar_dir)
    ) + " |"
    corpo = ["| " + " | ".join(l) + " |" for l in linhas]
    md = "\n".join([cab, sep, *corpo])
    if legenda:
        md += f"\n\n: {legenda}\n"
    return md


def tabela(df: pl.DataFrame, legenda: str | None = None, direita: list[str] | None = None):
    
    from IPython.display import Markdown
    return Markdown(df_to_markdown(df, legenda=legenda, direita=direita))


def freq_table(df, variavel, mapa=None, rotulo="Categoria", valor_nulo=9, ordem=None, ordenar_por_freq=False):
    
    if mapa is None:
        expr = pl.col(variavel).cast(pl.Utf8).fill_null("Ignorado")
    else:
        ignorado = mapa.get(valor_nulo, "Ignorado")
        expr = (
            pl.col(variavel).cast(pl.Int64, strict=False).fill_null(valor_nulo)
            .replace_strict(mapa, default=ignorado, return_dtype=pl.Utf8)
        )
    out = df.select(expr.alias(rotulo)).group_by(rotulo).len().rename({"len": "Nascimentos"})
    out = out.with_columns((pl.col("Nascimentos") / pl.col("Nascimentos").sum() * 100).alias("%"))

    if ordenar_por_freq or (ordem is None and mapa is None):
        return out.sort("Nascimentos", descending=True)
    if ordem is None:
        ordem = list(dict.fromkeys(mapa.values()))
    idx = {r: i for i, r in enumerate(ordem)}
    return (
        out.with_columns(pl.col(rotulo).replace_strict(idx, default=len(idx), return_dtype=pl.Int64).alias("_o"))
        .sort(["_o", "Nascimentos"], descending=[False, True])
        .drop("_o")
    )


def taxa_global(df, indicador: str):
    
    s = df[indicador].drop_nulls().cast(pl.Float64)
    return (s.mean() * 100 if len(s) else None), len(s)


def indicadores_por_grupo(df, grupo: str, indicadores: dict, ordem: list | None = None, rotulo_grupo: str | None = None):
    
    base = df.filter(pl.col(grupo).is_not_null())
    out = base.group_by(grupo).agg(
        pl.len().alias("Nascimentos"),
        *[(pl.col(col).cast(pl.Float64).mean() * 100).alias(nome) for nome, col in indicadores.items()],
    )
    out = out.with_columns((pl.col("Nascimentos") / df.height * 100).alias("% do total"))
    out = out.select([grupo, "Nascimentos", "% do total", *indicadores.keys()])
    if ordem:
        idx = {r: i for i, r in enumerate(ordem)}
        out = (
            out.with_columns(pl.col(grupo).replace_strict(idx, default=len(idx), return_dtype=pl.Int64).alias("_o"))
            .sort("_o").drop("_o")
        )
    else:
        out = out.sort("Nascimentos", descending=True)
    return out.rename({grupo: rotulo_grupo}) if rotulo_grupo else out


def resumo_numerico(df, colunas: dict) -> pl.DataFrame:
    
    def _f(v):
        if v is None:
            return "–"
        return fmt_num(v, 0) if float(v).is_integer() else fmt_num(v, 1)

    linhas = []
    for rotulo, col in colunas.items():
        s = df[col].drop_nulls().cast(pl.Float64)
        if len(s) == 0:
            continue
        linhas.append({
            "Variável": rotulo,
            "Registros válidos": len(s),
            "Média": fmt_num(s.mean(), 1),
            "Mediana": _f(s.median()),
            "P25": _f(s.quantile(0.25, interpolation="nearest")),
            "P75": _f(s.quantile(0.75, interpolation="nearest")),
            "Mín.": _f(s.min()),
            "Máx.": _f(s.max()),
        })
    return pl.DataFrame(linhas)


def tabela_robson(df) -> pl.DataFrame:
    
    descricao = {
        1: "Nulípara, único, cefálico, ≥ 37 sem, trabalho de parto espontâneo",
        2: "Nulípara, único, cefálico, ≥ 37 sem, induzido ou cesárea antes do trabalho de parto",
        3: "Multípara sem cesárea anterior, único, cefálico, ≥ 37 sem, espontâneo",
        4: "Multípara sem cesárea anterior, único, cefálico, ≥ 37 sem, induzido ou cesárea antes do trabalho de parto",
        5: "Multípara com cesárea anterior, único, cefálico, ≥ 37 sem",
        6: "Nulípara, apresentação pélvica",
        7: "Multípara, apresentação pélvica (inclui cesárea anterior)",
        8: "Gestação múltipla (inclui cesárea anterior)",
        9: "Situação transversa ou oblíqua (inclui cesárea anterior)",
        10: "Único, cefálico, < 37 sem (inclui cesárea anterior)",
    }
    base = df.filter(pl.col("TPROBSON").is_not_null()).with_columns(pl.col("TPROBSON").cast(pl.Int64))
    n_total = base.height
    n_ces_total = base["IND_CESAREA"].cast(pl.Float64).sum()
    out = base.group_by("TPROBSON").agg(
        pl.len().alias("Nascimentos"),
        pl.col("IND_CESAREA").cast(pl.Float64).sum().alias("_ces"),
        (pl.col("IND_CESAREA").cast(pl.Float64).mean() * 100).alias("Taxa de cesárea (%)"),
    ).sort("TPROBSON")
    return out.select(
        pl.col("TPROBSON").alias("Grupo"),
        pl.col("TPROBSON").replace_strict(descricao, default="Não classificável", return_dtype=pl.Utf8).alias("Descrição"),
        "Nascimentos",
        (pl.col("Nascimentos") / n_total * 100).alias("% dos nascimentos"),
        "Taxa de cesárea (%)",
        (pl.col("_ces") / n_ces_total * 100).alias("% das cesáreas"),
    ).with_columns(pl.col("Grupo").cast(pl.Utf8))


def _tema(figsize=(7, 4), **kwargs):
    return pn.theme_minimal() + pn.theme(figure_size=figsize, **kwargs)


def _rotulos_milhar(valores):
    return [fmt_int(v) for v in valores]


def plot_bars(df, variavel, titulo, xlabel, mapa=None, ordem=None, valor_nulo=9,
              tamanho_texto=9, rotacao=0, ordenar_por_freq=False, horizontal=False,
              figsize=(7, 4), cor=PALETA[0]):
    """Barras com a contagem de cada categoria. Mesma interface de plot_bars_pct."""
    tab = freq_table(df, variavel, mapa, "categoria", valor_nulo, ordem, ordenar_por_freq)
    cats = tab["categoria"].to_list()
    tab = tab.with_columns(pl.Series("rotulo", [fmt_int(v) for v in tab["Nascimentos"]]))
    topo = tab["Nascimentos"].max()

    p = (
        pn.ggplot(tab, pn.aes(x="categoria", y="Nascimentos"))
        + pn.geom_col(fill=cor)
        + pn.labs(title=titulo, x=xlabel, y="Frequência")
        + pn.scale_y_continuous(limits=(0, topo * 1.15), expand=(0, 0), labels=_rotulos_milhar)
    )
    if horizontal:
        p = (
            p
            + pn.geom_text(pn.aes(label="rotulo"), ha="left", nudge_y=topo * 0.01, size=tamanho_texto)
            + pn.scale_x_discrete(limits=cats[::-1])
            + pn.coord_flip()
            + _tema(figsize, legend_position="none")
        )
    else:
        p = (
            p
            + pn.geom_text(pn.aes(label="rotulo"), va="bottom", nudge_y=topo * 0.01, size=tamanho_texto)
            + pn.scale_x_discrete(limits=cats)
            + _tema(figsize, legend_position="none",
                    axis_text_x=pn.element_text(rotation=rotacao, ha="right" if rotacao else "center"))
        )
    return p

def plot_bars_pct(df, variavel, titulo, xlabel, mapa=None, ordem=None, valor_nulo=9,
                  tamanho_texto=9, rotacao=0, ordenar_por_freq=False, horizontal=False,
                  figsize=(7, 4), cor=PALETA[0]):
                  
    tab = freq_table(df, variavel, mapa, "categoria", valor_nulo, ordem, ordenar_por_freq)
    cats = tab["categoria"].to_list()
    tab = tab.with_columns(pl.Series("rotulo", [fmt_pct(v) for v in tab["%"]]))
    topo = tab["%"].max()

    p = (
        pn.ggplot(tab, pn.aes(x="categoria", y="%"))
        + pn.geom_col(fill=cor)
        + pn.labs(title=titulo, x=xlabel, y="Proporção dos nascimentos (%)")
        + pn.scale_y_continuous(limits=(0, topo * 1.15), expand=(0, 0))
    )
    if horizontal:
        p = (
            p
            + pn.geom_text(pn.aes(label="rotulo"), ha="left", nudge_y=topo * 0.01, size=tamanho_texto)
            + pn.scale_x_discrete(limits=cats[::-1])
            + pn.coord_flip()
            + _tema(figsize, legend_position="none")
        )
    else:
        p = (
            p
            + pn.geom_text(pn.aes(label="rotulo"), va="bottom", nudge_y=topo * 0.01, size=tamanho_texto)
            + pn.scale_x_discrete(limits=cats)
            + _tema(figsize, legend_position="none",
                    axis_text_x=pn.element_text(rotation=rotacao, ha="right" if rotacao else "center"))
        )
    return p


def plot_hist(df, variavel, titulo, xlabel, bins=30, binwidth=None, xlim=None,
              linhas=None, cor=PALETA[0], figsize=(7, 3.8), log_x=False):
              
    d = df.select(pl.col(variavel).cast(pl.Float64)).drop_nulls()
    geom = (
        pn.geom_histogram(binwidth=binwidth, fill=cor, color="white")
        if binwidth else pn.geom_histogram(bins=bins, fill=cor, color="white")
    )
    p = (
        pn.ggplot(d, pn.aes(x=variavel))
        + geom
        + pn.labs(title=titulo, x=xlabel, y="Frequência")
        + pn.scale_y_continuous(labels=_rotulos_milhar)
        + _tema(figsize)
    )
    if log_x:
        p += pn.scale_x_log10()
    if xlim:
        p += pn.coord_cartesian(xlim=xlim)
    for x in linhas or []:
        p += pn.geom_vline(xintercept=x, linetype="dashed", color=PALETA[3])
    return p


def plot_taxa(tab, grupo, coluna, titulo, rotulo_eixo, referencia=None, horizontal=True,
              ordenar=True, tamanho_texto=8, figsize=(7, 5), cor=PALETA[0]):
              
    if ordenar:
        tab = tab.sort(coluna, descending=True)
    cats = tab[grupo].to_list()
    tab = tab.with_columns(pl.Series("rotulo", [fmt_pct(v) for v in tab[coluna]]))
    topo = tab[coluna].max()

    p = (
        pn.ggplot(tab, pn.aes(x=grupo, y=coluna))
        + pn.geom_col(fill=cor)
        + pn.labs(title=titulo, x="", y=rotulo_eixo)
        + pn.scale_y_continuous(limits=(0, topo * 1.15), expand=(0, 0))
    )
    if referencia is not None:
        p += pn.geom_hline(yintercept=referencia, linetype="dashed", color=PALETA[1])
    if horizontal:
        p = (
            p
            + pn.geom_text(pn.aes(label="rotulo"), ha="left", nudge_y=topo * 0.01, size=tamanho_texto)
            + pn.scale_x_discrete(limits=cats[::-1])
            + pn.coord_flip()
        )
    else:
        p = (
            p
            + pn.geom_text(pn.aes(label="rotulo"), va="bottom", nudge_y=topo * 0.01, size=tamanho_texto)
            + pn.scale_x_discrete(limits=cats)
        )
    return p + _tema(figsize, legend_position="none")


def plot_stacked_pct(df, grupo, categoria, titulo, xlabel, legenda, ordem_grupo=None,
                     ordem_categoria=None, figsize=(7, 4)):
                     
    d = (
        df.select(
            pl.col(grupo).cast(pl.Utf8).alias("g"),
            pl.col(categoria).cast(pl.Utf8).fill_null("Ignorado").alias("c"),
        )
        .drop_nulls("g")
        .group_by("g", "c").len()
    )
    ordem_grupo = ordem_grupo or d.group_by("g").agg(pl.col("len").sum()).sort("len", descending=True)["g"].to_list()
    ordem_categoria = ordem_categoria or sorted(d["c"].unique().to_list())
    cores = [PALETA[i % len(PALETA)] for i in range(len(ordem_categoria))]

    return (
        pn.ggplot(d, pn.aes(x="g", y="len", fill="c"))
        + pn.geom_col(position="fill", color="white", size=0.2)
        + pn.scale_x_discrete(limits=ordem_grupo)
        + pn.scale_fill_manual(values=cores, limits=ordem_categoria, name=legenda)
        + pn.scale_y_continuous(labels=lambda ls: [f"{v * 100:.0f}%" for v in ls], expand=(0, 0))
        + pn.labs(title=titulo, x=xlabel, y="Proporção dentro do grupo")
        + _tema(figsize, legend_position="bottom")
    )


def plot_line_mensal(df, ano, titulo="Nascimentos por mês", figsize=(7, 3.6)):
    
    meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    m = (
        df.select(pl.col("YEAR_MONTH").cast(pl.Utf8).alias("mes"))
        .drop_nulls()
        .filter(pl.col("mes").str.starts_with(str(ano)))
        .group_by("mes").len().sort("mes")
        .with_columns(pl.col("mes").str.slice(5, 2).cast(pl.Int64).alias("n_mes"))
        .with_columns(pl.col("n_mes").replace_strict({i + 1: n for i, n in enumerate(meses)}, return_dtype=pl.Utf8).alias("rotulo_mes"))
    )
    return (
        pn.ggplot(m, pn.aes(x="rotulo_mes", y="len", group=1))
        + pn.geom_line(color=PALETA[0], size=1)
        + pn.geom_point(color=PALETA[0], size=2.5)
        + pn.scale_x_discrete(limits=m["rotulo_mes"].to_list())
        + pn.scale_y_continuous(labels=_rotulos_milhar)
        + pn.expand_limits(y=0)
        + pn.labs(title=titulo, x="Mês de nascimento", y="Nascimentos")
        + _tema(figsize)
    )