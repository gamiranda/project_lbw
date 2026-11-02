import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

# ----------------------------- Configuração -----------------------------
REPORTS_DIR = Path(__file__).parent / "reports"
MANIFEST = REPORTS_DIR / "manifest.json"
PADRAO = re.compile(r"relatorio_sinasc_(\d{4})\.pdf$")
REPO_URL = "https://github.com/gamiranda/project_lbw"  

st.set_page_config(page_title="Relatórios SINASC", page_icon="👶", layout="centered")


# ----------------------------- Funções -----------------------------------
def listar_relatorios() -> dict[int, Path]:
    """Anos disponíveis (mais recente primeiro), a partir dos PDFs que existem em reports/."""
    encontrados = {}
    for p in REPORTS_DIR.glob("relatorio_sinasc_*.pdf"):
        m = PADRAO.search(p.name)
        if m:
            encontrados[int(m.group(1))] = p
    return dict(sorted(encontrados.items(), reverse=True))


def ler_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


@st.cache_data(show_spinner=False)
def ler_pdf(caminho: str, mtime: float) -> bytes:
    # mtime entra só como chave de cache: se o arquivo mudar, o cache é refeito
    return Path(caminho).read_bytes()


# ----------------------------- Interface ---------------------------------
st.title("Relatório anual de nascidos vivos")
st.caption("Fonte: SINASC / OpenDATASUS · Ministério da Saúde")

relatorios = listar_relatorios()
if not relatorios:
    st.warning("Nenhum relatório disponível no momento.")
    st.stop()

ano = st.selectbox("Selecione o ano", list(relatorios))
caminho = relatorios[ano]

pdf = ler_pdf(str(caminho), caminho.stat().st_mtime)
info = ler_manifest().get(str(ano), {})

col1, col2 = st.columns(2)
col1.metric("Ano de referência", ano)
col2.metric("Tamanho do arquivo", f"{len(pdf) / 1e6:.1f} MB".replace(".", ","))

if "gerado_em" in info:
    gerado = datetime.fromisoformat(info["gerado_em"])
    st.caption(f"Relatório gerado em {gerado:%d/%m/%Y}.")

st.download_button(
    label="📄 Baixar PDF",
    data=pdf,
    file_name=caminho.name,
    mime="application/pdf",
    type="primary",
)

with st.expander("Sobre este relatório"):
    st.markdown(
        """
- Os dados vêm da Declaração de Nascido Vivo (DNV), publicada no OpenDATASUS.
- Códigos de "ignorado" das variáveis numéricas (99 e 9999) são tratados como ausentes.
- As taxas são calculadas apenas sobre os registros com informação válida.
- Análises por região e UF usam o local de residência da mãe.
- Dados do ano mais recente podem ser preliminares e ser revisados pelo Ministério da Saúde.
        """
    )

st.divider()
st.caption(f"Desenvolvido por Gabriel Miranda · [Código no GitHub]({REPO_URL})")