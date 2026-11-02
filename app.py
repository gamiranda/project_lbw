import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st

# ----------------------------- Configuração -----------------------------
QMD = Path("RelatorioAnual.qmd")
OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)

ANOS = list(range(datetime.now().year - 1, 2014, -1))  # ajuste conforme os anos disponíveis
SUBTITULO = "Sistema de Informações sobre Nascidos Vivos (SINASC) – {ano}"

st.set_page_config(page_title="Relatórios SINASC", page_icon="👶", layout="centered")


# ----------------------------- Funções -----------------------------------
def pdf_path(ano: int) -> Path:
    return OUT_DIR / f"relatorio_sinasc_{ano}.pdf"


def render_report(ano: int) -> Path:
    """Renderiza o .qmd para o ano escolhido e devolve o caminho do PDF."""
    destino = pdf_path(ano)
    cmd = [
        "quarto", "render", str(QMD),
        "--to", "pdf",
        "-P", f"ano:{ano}",
        "-M", f"subtitle:{SUBTITULO.format(ano=ano)}",
        "--output", destino.name,
    ]
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        raise RuntimeError(resultado.stderr or resultado.stdout)

    # o Quarto grava o PDF ao lado do .qmd; movemos para reports/
    gerado = QMD.parent / destino.name
    if gerado.exists():
        gerado.replace(destino)
    return destino


@st.cache_data(show_spinner=False)
def get_pdf_bytes(ano: int) -> bytes:
    destino = pdf_path(ano)
    if not destino.exists():
        render_report(ano)
    return destino.read_bytes()


# ----------------------------- Interface ---------------------------------
st.title("Relatório anual de nascidos vivos")
st.caption("Fonte: SINASC / OpenDATASUS")

ano = st.selectbox("Selecione o ano", ANOS)

if st.button("Gerar relatório", type="primary"):
    with st.spinner(f"Gerando relatório de {ano}... isso pode levar alguns minutos."):
        try:
            st.session_state["pdf"] = get_pdf_bytes(ano)
            st.session_state["ano_pdf"] = ano
        except Exception as e:
            st.error("Falha ao renderizar o relatório.")
            st.code(str(e))

if "pdf" in st.session_state:
    st.success(f"Relatório de {st.session_state['ano_pdf']} pronto!")
    st.download_button(
        label="📄 Baixar PDF",
        data=st.session_state["pdf"],
        file_name=f"relatorio_sinasc_{st.session_state['ano_pdf']}.pdf",
        mime="application/pdf",
    )