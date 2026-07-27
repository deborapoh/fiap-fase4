# Relatorio tecnico — Tech Challenge Fase 4

Sistema de monitoramento hospitalar multimodal  
FIAP — 8IADT · Fase 4

## 1. Objetivo

Construir um fluxo que analisa video clinico, audio de fala, series de sinais
vitais e evolucao de prescricoes; detecta anomalias; funde os sinais num escore
de risco; e apresenta alerta a equipe medica. A demonstracao em nuvem roda num
Hugging Face Space.

## 2. Fluxo multimodal

```
TORGO (audio) ──► Whisper + Praat + sentimento/termos ──► escore_criticidade
Keraal (video) ─► OpenPose JSON + YOLOv8 + classificador ─► escore_risco
BIDMC (vitais) ─► IsolationForest + LSTM AE + regras ───► escore_risco
MIMIC (rx) ─────► regras auditaveis de prescricoes ─────► escore_risco
                         │
                         ▼
              paciente sintetico (fusao ponderada)
                         │
                         ▼
           fila de alertas (Streamlit / HF Space)
```

As quatro fontes publicas **nao compartilham a mesma pessoa**. A fusao monta
pacientes sinteticos (amostras emparelhadas por semente) e declara isso
explicitamente. Inventar um join clinico entre TORGO, BIDMC, Keraal e MIMIC
seria desonesto.

## 3. Modelos por tipo de dado

### 3.1 Video (Keraal)

| Peça | Papel |
|---|---|
| Anotacoes Anvil (UTF-16) | Rotulo dos dois medicos; consenso quando A=B, senao o mais grave |
| OpenPose (JSONs do dataset) | Angulos de cotovelo, ombro, joelho e inclinacao de tronco |
| Perfil G2A | Distancia z-score ao grupo saudavel, por exercicio |
| Regressao logistica | Probabilidade de erro postural, treinada no consenso G1A |
| YOLOv8n | Pessoa/objetos em 5 frames por video (areas criticas) |

Validacao cruzada estratificada (5 folds) no consenso G1A (n=189):

| Metrica | Valor |
|---|---|
| AUC | 0,832 |
| Precisao | 0,802 |
| Revocacao | 0,770 |
| F1 | 0,786 |
| Acuracia | 0,778 |

Taxa de acordo entre medicos: 78,4%. Discordancia resolvida para o rotulo mais
grave (Incorrect > Incomplete > Motionless > Correct).

### 3.2 Audio (TORGO)

| Peça | Papel |
|---|---|
| faster-whisper large-v3 | Transcricao (substitui Azure Speech to Text) |
| Praat / parselmouth | jitter, shimmer, HNR, f0, pausa, taxa de fala |
| distilbert SST-2 | Sentimento (proxy de desconforto) |
| spaCy EntityRuler | Termos criticos por dicionario curado |

Amostra pareada por frase, microfone e sexo (40+40). Mann-Whitney:

| Metrica | Variacao | p |
|---|---|---|
| WER | +837% | <0,0001 |
| logprob Whisper | −49,7% | 0,0022 |
| proporcao de pausa | +35,3% | 0,0014 |

Jitter/shimmer/HNR saem invertidos sobre frase inteira (sao metricas de vogal
sustentada); a ressalva esta na metodologia e essas tres nao sustentam a
conclusao sozinhas.

### 3.3 Sinais vitais (BIDMC)

| Peça | Papel |
|---|---|
| Isolation Forest | Anomalia em estatisticas de janela 30 s |
| Autoencoder LSTM | Anomalia de forma temporal |
| Regras clinicas | Hipoxemia, bradi/taquicardia, apnea, sensor congelado |

BIDMC nao tem rotulo de evento: validacao por **anomalia injetada** (44 eventos
em 22 pacientes de teste). Combinados: **42/44 detectados** (95%), atraso
mediano 14 s, 2 alertas fora de evento.

| Detector | AUC | Precisao | Revocacao |
|---|---|---|---|
| Isolation Forest | 0,903 | 1,000 | 0,519 |
| Autoencoder LSTM | 0,841 | 0,787 | 0,638 |
| Regras clinicas | 0,764 | 0,643 | 0,682 |

### 3.4 Prescricoes (MIMIC-IV Demo)

Cinco regras auditaveis sobre 18.087 ordens: inconsistencia temporal, salto de
dose (≥5× em ≤48 h), dose atipica (z robusto ≥5), escalonamento de via
(oral→parenteral ≤24 h), rajada de prescricoes (p99 em janela de 6 h).
1.714 eventos sinalizados; 686 em medicamentos de alto risco (ISMP).

Exemplos: midazolam 100 mg → 0,5 mg em 8 h; insulina 75 U contra mediana 5;
KCl oral → IV em 6 h.

### 3.5 Fusao

Pesos: audio 0,20 · vitais 0,30 · video 0,30 · prescricoes 0,20.  
Alerta se risco ≥ 0,55, ou uma frente ≥ 0,75, ou ≥2 frentes em alerta.
Severidade alta se risco ≥ 0,75 ou ≥2 frentes.

## 4. Resultados e exemplos de anomalias

### Video
- 301 gravacoes processadas; 161 com alerta apos YOLO.
- Top desvios sao majoritariamente `Incorrect` segundo os medicos
  (ex.: G1A-CTK-R2-Brest-008, escore 0,70).
- YOLO: pessoa em 85,4% dos frames amostrados.

### Vitais (series reais, sem injecao)
- 27 alertas em 53 pacientes / 425 min.
- Pacientes 49 e 32: hipoxemia sustentada nos 8 min; paciente 13: bradipneia
  + sinal congelado.

### Prescricoes
Ver secao 3.4.

### Fusao (20 pacientes sinteticos, semente 42)
- 15 alertas; 7 severidade alta; escore medio 0,34.
- Ex.: SYN-020 risco 0,67 (vitais 0,67 · video 0,81 · prescricoes 0,75).

## 5. Camada de nuvem

O enunciado cita Azure Speech to Text e Azure Text Analytics. A solucao usa:

1. **Modelos Hugging Face** locais (Whisper, distilbert) no lugar dos dois
   servicos Azure de NLP/speech.
2. **Hugging Face Space** com dashboard da fila de alertas e da fusao —
   servico gerenciado em nuvem, alinhado ao Objetivo do enunciado
   (“servicos gerenciados em nuvem, **como** Azure”).

URL: https://huggingface.co/spaces/deborapoh/fiap-fase4-monitoramento

Nota: em 2026 o hospedamento gratuito do Hugging Face cobre Spaces
**static**; Gradio/Docker/Streamlit na CPU gratuita passaram a exigir plano
PRO. Por isso o Space publico e um dashboard estatico gerado dos CSVs, e o
app Streamlit completo (`app/streamlit_app.py`) roda localmente para a demo
gravada. Ambos consomem os mesmos resultados.

## 6. Desvios de escopo

Decisoes de engenharia, nao omissao silenciosa:

1. **Azure → Hugging Face** — barreira de cartao de credito; Space fecha o
   requisito de servico gerenciado.
2. **KIMORE → Keraal** — KIMORE indisponivel (403/404); Keraal traz video RGB,
   OpenPose pronto e anotacao medica temporal.
3. **OpenPose nao compilado** — JSONs do Keraal atendem o requisito; pose nova
   via YOLOv8-pose se necessario.
4. **scispacy descartado** — conflito numpy<2 vs OpenCV; TORGO sem vocabulario
   clinico; EntityRuler com dicionario curado.
5. **Python 3.11** — 3.14 incompativel com torch/ultralytics/faster-whisper.
6. **Amostra TORGO pareada por frase** — amostragem sequencial enviesava sexo,
   microfone e locutor.

Decisoes de metodologia (nao sao desvio de escopo, mas constam aqui por
rastreabilidade):

- BIDMC: validacao por anomalia injetada; split por paciente; limiar fora do
  ajuste; sinal congelado com persistencia de ~55 s.
- Keraal: discordancia A/B → rotulo mais grave; classificador supervisionado
  no consenso (ha rotulo verdadeiro).
- Fusao: paciente sintetico declarado.

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

## 8. Entregavel pendente do autor

Video de apresentacao (ate 15 min) no YouTube/Vimeo — demonstrando as tres
frentes, a fusao, o Space e o fluxo de alerta. Fora do escopo deste relatorio.
