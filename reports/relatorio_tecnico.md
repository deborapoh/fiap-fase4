# Relatório técnico — Tech Challenge Fase 4

Sistema de monitoramento hospitalar multimodal  
FIAP — 8IADT · Fase 4

## 1. Objetivo

Construir um fluxo que analisa vídeo clínico, áudio de fala, séries de sinais
vitais e evolução de prescrições; detecta anomalias; funde os sinais num escore
de risco; e apresenta alerta à equipe médica. A demonstração em nuvem roda num
Hugging Face Space.

## 2. Fluxo multimodal

```
TORGO (áudio) ──► Whisper + Praat + sentimento/termos ──► escore_criticidade
Keraal (vídeo) ─► OpenPose JSON + YOLOv8 + classificador ─► escore_risco
BIDMC (vitais) ─► IsolationForest + LSTM AE + regras ───► escore_risco
MIMIC (rx) ─────► regras auditáveis de prescrições ─────► escore_risco
                         │
                         ▼
              paciente sintético (fusão ponderada)
                         │
                         ▼
           fila de alertas (Streamlit / HF Space)
```

As quatro fontes públicas **não compartilham a mesma pessoa**. A fusão monta
pacientes sintéticos (amostras emparelhadas por semente) e declara isso
explicitamente. Inventar um join clínico entre TORGO, BIDMC, Keraal e MIMIC
seria desonesto.

## 3. Modelos por tipo de dado

### 3.1 Vídeo (Keraal)

| Peça | Papel |
|---|---|
| Anotações Anvil (UTF-16) | Rótulo dos dois médicos; consenso quando A=B, senão o mais grave |
| OpenPose (JSONs do dataset) | Ângulos de cotovelo, ombro, joelho e inclinação de tronco |
| Perfil G2A | Distância z-score ao grupo saudável, por exercício |
| Regressão logística | Probabilidade de erro postural, treinada no consenso G1A |
| YOLOv8n | Pessoa/objetos em 5 frames por vídeo (áreas críticas) |

Validação cruzada estratificada (5 folds) no consenso G1A (n=189):

| Métrica | Valor |
|---|---|
| AUC | 0,832 |
| Precisão | 0,802 |
| Revocação | 0,770 |
| F1 | 0,786 |
| Acurácia | 0,778 |

Taxa de acordo entre médicos: 78,4%. Discordância resolvida para o rótulo mais
grave (Incorrect > Incomplete > Motionless > Correct).

### 3.2 Áudio (TORGO)

| Peça | Papel |
|---|---|
| faster-whisper large-v3 | Transcrição de áudio |
| Praat / parselmouth | jitter, shimmer, HNR, f0, pausa, taxa de fala |
| distilbert SST-2 | Sentimento (proxy de desconforto) |
| spaCy EntityRuler | Termos críticos por dicionário curado |

Amostra pareada por frase, microfone e sexo (40+40). Mann-Whitney:

| Métrica | Variação | p |
|---|---|---|
| WER | +837% | <0,0001 |
| logprob Whisper | −49,7% | 0,0022 |
| proporção de pausa | +35,3% | 0,0014 |

Jitter/shimmer/HNR saem invertidos sobre frase inteira (são métricas de vogal
sustentada); a ressalva está na metodologia e essas três não sustentam a
conclusão sozinhas.

### 3.3 Sinais vitais (BIDMC)

| Peça | Papel |
|---|---|
| Isolation Forest | Anomalia em estatísticas de janela 30 s |
| Autoencoder LSTM | Anomalia de forma temporal |
| Regras clínicas | Hipoxemia, bradi/taquicardia, apneia, sensor congelado |

BIDMC não tem rótulo de evento: validação por **anomalia injetada** (44 eventos
em 22 pacientes de teste). Combinados: **42/44 detectados** (95%), atraso
mediano 14 s, 2 alertas fora de evento.

| Detector | AUC | Precisão | Revocação |
|---|---|---|---|
| Isolation Forest | 0,903 | 1,000 | 0,519 |
| Autoencoder LSTM | 0,841 | 0,787 | 0,638 |
| Regras clínicas | 0,764 | 0,643 | 0,682 |

### 3.4 Prescrições (MIMIC-IV Demo)

Cinco regras auditáveis sobre 18.087 ordens: inconsistência temporal, salto de
dose (≥5× em ≤48 h), dose atípica (z robusto ≥5), escalonamento de via
(oral→parenteral ≤24 h), rajada de prescrições (p99 em janela de 6 h).
1.714 eventos sinalizados; 686 em medicamentos de alto risco (ISMP).

Exemplos: midazolam 100 mg → 0,5 mg em 8 h; insulina 75 U contra mediana 5;
KCl oral → IV em 6 h.

### 3.5 Fusão

Pesos: áudio 0,20 · vitais 0,30 · vídeo 0,30 · prescrições 0,20.  
Alerta se risco ≥ 0,55, ou uma frente ≥ 0,75, ou ≥2 frentes em alerta.
Severidade alta se risco ≥ 0,75 ou ≥2 frentes.

## 4. Resultados e exemplos de anomalias

### Vídeo
- 301 gravações processadas; 161 com alerta após YOLO.
- Top desvios são majoritariamente `Incorrect` segundo os médicos
  (ex.: G1A-CTK-R2-Brest-008, escore 0,70).
- YOLO: pessoa em 85,4% dos frames amostrados.

### Vitais (séries reais, sem injeção)
- 27 alertas em 53 pacientes / 425 min.
- Pacientes 49 e 32: hipoxemia sustentada nos 8 min; paciente 13: bradipneia
  + sinal congelado.

### Prescrições
Ver seção 3.4.

### Fusão (20 pacientes sintéticos, semente 42)
- 15 alertas; 7 severidade alta; escore médio 0,34.
- Ex.: SYN-020 risco 0,67 (vitais 0,67 · vídeo 0,81 · prescrições 0,75).

## 5. Camada de nuvem

A solução usa Hugging Face como serviço gerenciado em nuvem:

1. **Modelos Hugging Face** locais — Whisper (faster-whisper) para transcrição
   e distilbert para sentimento.
2. **Hugging Face Space** — dashboard da fila de alertas e da fusão.

URL: https://huggingface.co/spaces/deborapoh/fiap-fase4-monitoramento

Nota: em 2026 o hospedamento gratuito do Hugging Face cobre Spaces
**static**; Gradio/Docker/Streamlit na CPU gratuita passaram a exigir plano
PRO. Por isso o Space público é um dashboard estático gerado dos CSVs, e o
app Streamlit completo (`app/streamlit_app.py`) roda localmente para a demo
gravada. Ambos consomem os mesmos resultados.

## 6. Desvios de escopo

Decisões de engenharia, não omissão silenciosa:

1. **Nuvem via Hugging Face** — Spaces para a demo pública; Whisper e
   distilbert para áudio e sentimento.
2. **KIMORE → Keraal** — KIMORE indisponível (403/404); Keraal traz vídeo RGB,
   OpenPose pronto e anotação médica temporal.
3. **OpenPose não compilado** — JSONs do Keraal atendem o requisito; pose nova
   via YOLOv8-pose se necessário.
4. **scispacy descartado** — conflito numpy<2 vs OpenCV; TORGO sem vocabulário
   clínico; EntityRuler com dicionário curado.
5. **Python 3.11** — 3.14 incompatível com torch/ultralytics/faster-whisper.
6. **Amostra TORGO pareada por frase** — amostragem sequencial enviesava sexo,
   microfone e locutor.

Decisões de metodologia (não são desvio de escopo, mas constam aqui por
rastreabilidade):

- BIDMC: validação por anomalia injetada; split por paciente; limiar fora do
  ajuste; sinal congelado com persistência de ~55 s.
- Keraal: discordância A/B → rótulo mais grave; classificador supervisionado
  no consenso (há rótulo verdadeiro).
- Fusão: paciente sintético declarado.

## 7. Como reproduzir

```bash
cd fiap-fase4
source .venv/bin/activate
python scripts/analisar_audio.py          # ~7 min
python scripts/detectar_anomalias.py      # ~30 s
python scripts/analisar_video.py          # ~2 min (com YOLO)
python scripts/fundir_risco.py
streamlit run app/streamlit_app.py
```

## 8. Entregável pendente do autor

Vídeo de apresentação (até 15 min) no YouTube/Vimeo — demonstrando as três
frentes, a fusão, o Space e o fluxo de alerta. Fora do escopo deste relatório.
