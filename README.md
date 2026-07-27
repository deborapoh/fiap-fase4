# Tech Challenge — Fase 4

Sistema de monitoramento hospitalar multimodal (áudio · vídeo · vitais · prescrições).

- Relatório: [`reports/relatorio_tecnico.md`](reports/relatorio_tecnico.md)
- Datasets: [`docs/datasets.md`](docs/datasets.md)
- Demo em nuvem: https://huggingface.co/spaces/deborapoh/fiap-fase4-monitoramento
- Demo local: `streamlit run app/streamlit_app.py`

```bash
source .venv/bin/activate
python scripts/analisar_audio.py
python scripts/detectar_anomalias.py
python scripts/analisar_video.py
python scripts/fundir_risco.py
streamlit run app/streamlit_app.py
```

Checklist do que é **obrigatório** neste trabalho.

## Entregas técnicas

### 1. Análise de Vídeo

- Processar vídeos clínicos (ex.: sessões de fisioterapia ou cirurgias gravadas)
- Detectar movimentos ou eventos fora do padrão esperado, utilizando:
  - OpenPose para análise postural
  - YOLOv8 para detecção de objetos e áreas críticas
- Gerar relatórios automáticos indicando desvios ou falhas no procedimento

### 2. Análise de Áudio

- Processar áudios de consultas médicas
- Detectar alterações vocais indicativas de condições médicas (ex.: cansaço, dificuldades respiratórias)
- Transcrever com faster-whisper (Whisper)
- Identificar termos críticos (spaCy EntityRuler) e sentimento (distilbert)

### 3. Detecção de Anomalias

- Aplicar técnicas de detecção de anomalias em:
  - Séries temporais de sinais vitais (batimentos, pressão arterial, oxigenação)
  - Evolução de prescrições (alterações inesperadas no tratamento)
  - Padrões de movimentação do paciente durante a internação
- Gerar alertas automáticos para a equipe médica com base nas anomalias detectadas

## Entregáveis

### Repositório Git

- Código-fonte completo da solução
- Relatório técnico com:
  - Descrição do fluxo multimodal
  - Modelos aplicados em cada tipo de dado
  - Resultados obtidos e exemplos de anomalias detectadas

### Vídeo (até 15 minutos)

- Upload no YouTube ou Vimeo (público ou não listado)
- Demonstração do processamento multimodal:
  - Exemplo prático da análise de áudio e vídeo
  - Detecção e resposta a anomalias
  - Integração com Hugging Face Spaces
  - Fluxo final do alerta à equipe médica
