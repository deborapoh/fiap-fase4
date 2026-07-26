#!/usr/bin/env bash
# Baixa os datasets publicos usados no Tech Challenge Fase 4.
#
# Todos os downloads suportam retomada (-C -), o que importa porque o
# PhysioNet costuma entregar poucos MB/s e conexoes caem no meio.
# Rode o script novamente para continuar de onde parou.
#
# Uso: ./scripts/download_datasets.sh [keraal|bidmc|mimic|todos]

set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINO="$RAIZ/data/raw"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

baixar() {
  local url="$1" saida="$2"
  mkdir -p "$(dirname "$saida")"
  echo ">> $(basename "$saida")"
  # --retry cobre quedas de conexao; -C - retoma arquivo parcial.
  # --no-progress-meter porque a barra do curl polui o log quando o script
  # roda em background (o PhysioNet e lento e gera milhares de linhas).
  curl -L -A "$UA" -C - --retry 5 --retry-delay 5 --retry-all-errors \
       --no-progress-meter -o "$saida" "$url" || {
    # curl sai com 33 quando o servidor nao aceita retomada e o arquivo ja esta completo
    if [ "$?" = "33" ] && [ -s "$saida" ]; then
      echo "   ja completo"
    else
      return 1
    fi
  }
  echo "   $(du -h "$saida" | cut -f1)"
}

keraal() {
  # Videos RGB de reabilitacao com anotacao medica de erros.
  # Group2A = 6 adultos saudaveis, anotados. Group1A = 6 pacientes com
  # lombalgia cronica, anotados. Juntos permitem comparar saudavel x paciente.
  local base="http://keraal.enstb.org/data"
  baixar "$base/group2A.tar.xz"           "$DESTINO/keraal/group2A.tar.xz"
  baixar "$base/group1A.tar.xz"           "$DESTINO/keraal/group1A.tar.xz"
  baixar "$base/pose_model.pth"           "$DESTINO/keraal/pose_model.pth"
  baixar "$base/readme_files_format.txt"  "$DESTINO/keraal/readme_files_format.txt"
}

bidmc() {
  # Sinais vitais de UTI: HR, frequencia respiratoria e SpO2 a 1 Hz.
  baixar "https://physionet.org/static/published-projects/bidmc/bidmc-ppg-and-respiration-dataset-1.0.0.zip" \
         "$DESTINO/bidmc/bidmc-1.0.0.zip"
}

mimic() {
  # Demo aberto do MIMIC-IV: 100 pacientes, inclui prescriptions e pharmacy.
  baixar "https://physionet.org/static/published-projects/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2.zip" \
         "$DESTINO/mimic-iv-demo/mimic-iv-demo-2.2.zip"
}

case "${1:-todos}" in
  keraal) keraal ;;
  bidmc)  bidmc ;;
  mimic)  mimic ;;
  todos)  keraal; bidmc; mimic ;;
  *) echo "uso: $0 [keraal|bidmc|mimic|todos]" >&2; exit 1 ;;
esac

echo
echo "Concluido. Conteudo de $DESTINO:"
du -sh "$DESTINO"/* 2>/dev/null || true
