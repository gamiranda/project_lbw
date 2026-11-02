"""Gera os PDFs anuais do relatório SINASC e grava em reports/.

Uso (a partir da raiz do repositório):

    python scripts/build_reports.py                  # gera só os anos que ainda não têm PDF
    python scripts/build_reports.py --anos 2022 2023 # gera apenas esses anos (se ainda não existirem)
    python scripts/build_reports.py --anos 2024 --force   # regera mesmo que o PDF já exista
    python scripts/build_reports.py --force          # regera todos os anos

Requisitos: Quarto + LaTeX (TinyTeX) instalados e as dependências de
requirements-build.txt.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------- Configuração -----------------------------
ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"      
OUT_DIR = ROOT / "reports"        
CACHE_DIR = ROOT / "data"         
MANIFEST = OUT_DIR / "manifest.json"
QMD = "RelatorioAnual.qmd"

ANO_MIN = 2019
SUBTITULO = "Sistema de Informações sobre Nascidos Vivos (SINASC) – {ano}"
URL_DADOS = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINASC/csv/SINASC_{ano}_csv.zip"


# ----------------------------- Funções -----------------------------------
def pdf_path(ano: int) -> Path:
    return OUT_DIR / f"relatorio_sinasc_{ano}.pdf"


def dados_disponiveis(ano: int) -> bool:
    """Confere (sem baixar) se o OpenDATASUS já publicou o arquivo do ano."""
    req = urllib.request.Request(URL_DADOS.format(ano=ano), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30):
            return True
    except urllib.error.HTTPError as e:
        # o S3 costuma responder 403 (e não 404) para objetos que não existem
        if e.code in (403, 404):
            return False
        raise


def render(ano: int) -> Path:
    """Renderiza o .qmd para o ano e devolve o caminho do PDF em reports/."""
    nome_pdf = pdf_path(ano).name
    cache = CACHE_DIR / f"sinasc_{ano}.parquet"

    with tempfile.TemporaryDirectory(prefix=f"sinasc_{ano}_") as tmp_str:
        tmp = Path(tmp_str)

        # Cópia isolada: o Quarto grava arquivos intermediários ao lado do .qmd
        # e o .qmd lê/grava sinasc_{ano}.parquet no diretório de trabalho.
        shutil.copytree(
            REPORT_DIR, tmp, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".quarto", "*_files", "__pycache__", "*.parquet", "*.pdf", "*.tex", "*.html"
            ),
        )
        if cache.exists():  # reaproveita o download de execuções anteriores
            shutil.copy2(cache, tmp / cache.name)

        cmd = [
            "quarto", "render", QMD,
            "--to", "pdf",
            "-P", f"ano:{ano}",
            "-M", f"subtitle:{SUBTITULO.format(ano=ano)}",
            "--output", nome_pdf,
        ]
        res = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        if res.returncode != 0:
            log = (res.stderr or res.stdout).strip().splitlines()
            raise RuntimeError("quarto render falhou:\n" + "\n".join(log[-40:]))

        gerado = tmp / nome_pdf
        if not gerado.exists():
            raise FileNotFoundError(f"O Quarto terminou sem erro, mas {nome_pdf} não foi criado.")

        # Grava em arquivo temporário e renomeia: nunca deixa um PDF pela metade.
        OUT_DIR.mkdir(exist_ok=True)
        destino = pdf_path(ano)
        parcial = destino.with_suffix(".pdf.tmp")
        shutil.copy2(gerado, parcial)
        parcial.replace(destino)

        baixado = tmp / cache.name
        if baixado.exists() and not cache.exists():
            CACHE_DIR.mkdir(exist_ok=True)
            shutil.copy2(baixado, cache)

    return destino


def atualizar_manifest(ano: int, pdf: Path) -> None:
    """Registra quando cada PDF foi gerado (o app mostra essa data)."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest[str(ano)] = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tamanho_bytes": pdf.stat().st_size,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera os relatórios anuais do SINASC em PDF.")
    parser.add_argument("--anos", nargs="+", type=int,
                        help="Anos a gerar (padrão: de %d até o ano passado)." % ANO_MIN)
    parser.add_argument("--force", action="store_true",
                        help="Regera o PDF mesmo que ele já exista.")
    args = parser.parse_args()

    if shutil.which("quarto") is None:
        print("ERRO: o comando 'quarto' não foi encontrado no PATH.", file=sys.stderr)
        return 1

    anos = sorted(set(args.anos or range(ANO_MIN, datetime.now().year)))
    pendentes = [a for a in anos if args.force or not pdf_path(a).exists()]
    ja_existem = [a for a in anos if a not in pendentes]

    if ja_existem:
        print(f"Já existem (use --force para regerar): {', '.join(map(str, ja_existem))}")
    if not pendentes:
        print("Nada a fazer.")
        return 0

    gerados, indisponiveis, falhas = [], [], []
    for ano in pendentes:
        print(f"\n=== {ano} ===", flush=True)
        try:
            if not dados_disponiveis(ano):
                print(f"Dados de {ano} ainda não publicados no OpenDATASUS; pulando.")
                indisponiveis.append(ano)
                continue
            inicio = time.time()
            pdf = render(ano)
            atualizar_manifest(ano, pdf)
            print(f"OK: {pdf.name} ({pdf.stat().st_size / 1e6:.1f} MB, {time.time() - inicio:.0f}s)")
            gerados.append(ano)
        except Exception as e:  # um ano com erro não deve impedir os demais
            print(f"FALHOU: {e}", file=sys.stderr)
            falhas.append(ano)

    print("\n--- Resumo ---")
    print(f"Gerados: {gerados or '-'}")
    print(f"Sem dados publicados: {indisponiveis or '-'}")
    print(f"Falhas: {falhas or '-'}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
