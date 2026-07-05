# FIAP Fase 4 — Monitoramento Multimodal Clínico

Sistema multimodal para o Tech Challenge da Fase 4: análise de vídeo (OpenPose KERAAL + YOLOv8), áudio de consulta (Azure Speech + Text Analytics + librosa), detecção de anomalias em vitais/prescrições/evolução clínica/movimentação, e fusão com alertas via FastAPI.

## Requisitos

- Python 3.10+
- Conta Azure com **Speech Services** e **Language Service** (região `brazilsouth` recomendada)
- Dataset KERAAL em `data/keraal/` (vídeos + OpenPose JSON)
- 3 áudios de consulta em `data/audio/` (já incluídos)

## Setup rápido (15 min)

```bash
# 1. Ambiente virtual
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Credenciais Azure
cp config/azure.example.env .env
# Edite .env com suas chaves

# 3. Dados sintéticos
python scripts/generate_synthetic.py

# 4. Demo completa (sem YOLO = mais rápido)
python scripts/run_demo.py --no-yolo
```

## Azure — o que configurar no `.env`

```env
AZURE_SPEECH_KEY=sua_chave
AZURE_SPEECH_REGION=brazilsouth
AZURE_LANGUAGE_KEY=sua_chave
AZURE_LANGUAGE_ENDPOINT=https://SEU-RECURSO.cognitiveservices.azure.com/
```

> **Dica:** Speech e Language podem usar a mesma chave se criou um recurso multi-serviço Cognitive Services.

Sem Azure configurado, a demo roda em **modo offline** com placeholders de transcrição e heurísticas locais.

## API FastAPI

Use o Python do `.venv` (evita pegar uvicorn/Python do sistema):

```bash
source .venv/bin/activate
python -m uvicorn src.api.main:app --reload --port 8000
```

Endpoints:

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Health check |
| GET | `/analyze/demo?run_yolo=false` | Demo com dados locais |
| POST | `/analyze` | Upload de vídeo/áudio ou dados de amostra |

Exemplo:

```bash
curl http://localhost:8000/analyze/demo?run_yolo=false | python -m json.tool
```

## Estrutura

```
src/
├── video/       # OpenPose KERAAL + YOLOv8
├── audio/       # Azure STT + Text Analytics + librosa
├── anomaly/     # Vitais, prescrições, notas, movimentação
├── fusion/      # Orchestrator + risk scorer
└── api/         # FastAPI
scripts/
├── generate_synthetic.py
├── run_demo.py
└── download_keraal.py
config/
├── critical_terms_pt.txt
└── azure.example.env
data/
├── keraal/      # vídeos + openpose (baixar separadamente)
├── audio/       # consultas gravadas
└── synthetic/   # vitais, prescrições, notas
```

## Mapeamento PDF → código

| Requisito | Implementação |
|-----------|---------------|
| OpenPose | Parser JSON KERAAL em `src/video/openpose_loader.py` |
| YOLOv8 | `ultralytics` em `src/video/yolo_detect.py` |
| Azure STT | `src/audio/azure_stt.py` |
| Azure Text Analytics | `src/audio/azure_text.py` |
| Fadiga/disartria | librosa F0 + energia em `src/audio/vocal_features.py` |
| Anomalias vitais | Isolation Forest em `src/anomaly/vitals.py` |
| Prescrições | Regras JSON em `src/anomaly/prescriptions.py` |
| Evolução clínica | Regras texto em `src/anomaly/clinical_notes.py` |
| Movimentação | Velocidade joints OpenPose em `src/anomaly/movement.py` |
| Fusão + alertas | `src/fusion/` + FastAPI |

## Vídeo demo (gravação)

1. Rodar `python scripts/run_demo.py` e mostrar o JSON de alerta
2. Subir API e chamar `/analyze/demo`
3. (Opcional) Mostrar Azure transcrevendo um áudio ao vivo

## Relatório

Ver `docs/relatorio.md` para estrutura sugerida do documento entregável.
