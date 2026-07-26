# Handoff: estado do projeto e como continuar

Documento de continuidade do Tech Challenge Fase 4. Leia junto com
[datasets.md](datasets.md), que detalha as fontes de dados.

## Ponto de retomada

Ultima sessao: 26/07, noite. O que foi feito nela:

- Pipeline de anomalias implementado e rodado ponta a ponta nas duas fontes:
  series de sinais vitais do BIDMC e evolucao de prescricoes do MIMIC-IV Demo.
- Duas das tres frentes obrigatorias (audio e anomalias) estao fechadas.

Nao ha pendencia mecanica aberta. Os commits continuam **apenas locais**, ainda
nao enviados ao remote; o `git status` mostra quantos. Rode `git push` quando
quiser publicar, atento a questao das duas contas GitHub descrita mais abaixo.

Nada em `data/processed/` e versionado (`data/` esta no `.gitignore`), entao
numa maquina nova e preciso refazer as duas execucoes:
`python scripts/analisar_audio.py`, cerca de 7 minutos, e
`python scripts/detectar_anomalias.py`, cerca de 30 segundos.

**Proximo passo: o pipeline de video** (`src/video/`), o unico dos tres
obrigatorios que falta, e o mais caro: 4,7 GB de MP4 e YOLOv8. Comece pelo
parser das anotacoes `.anvil`, pelo motivo detalhado na secao do que falta
fazer.

## O trabalho

Sistema de monitoramento hospitalar multimodal. Os requisitos obrigatorios
estao no [README.md](../README.md) e vem do PDF
`8IADT - Fase 4 - Tech challenge ultimo.pdf`, que fica no diretorio pai do
repositorio (fora do versionamento).

Tres frentes obrigatorias: analise de video (postura e objetos), analise de
audio (transcricao, termos criticos, sentimento) e deteccao de anomalias
(sinais vitais, prescricoes). Mais fusao multimodal e alerta a equipe medica.

Entregaveis: repositorio com codigo, relatorio tecnico e video de ate 15 min.

## Decisoes de arquitetura ja tomadas

Estas decisoes foram discutidas e aprovadas. **Nao reabrir sem motivo novo.**
Todas precisam constar da secao de desvios de escopo do relatorio tecnico.

### 1. Azure foi substituido por Hugging Face

O enunciado exige, nominalmente, Azure Speech to Text e Azure Text Analytics.
A usuaria decidiu nao usar Azure. Isso significa que dois bullets obrigatorios
nao serao cumpridos ao pe da letra.

A brecha usada: a secao Objetivo do enunciado pede "servicos gerenciados em
nuvem, **como** Azure Cognitive Services", onde o Azure e exemplo. Mantendo
algum servico gerenciado em nuvem no fluxo, o objetivo e atendido.

Escolhemos Hugging Face (Inference API para servir o modelo de audio, Spaces
para hospedar a demo) porque tem tier gratuito e **nao exige cartao de
credito**, que era o obstaculo. Google Cloud foi considerado e descartado:
Speech-to-Text e Natural Language API tem tier gratuito permanente, mas o
Google exige conta de faturamento vinculada, o mesmo obstaculo do Azure.

### 2. KIMORE foi substituido por Keraal

O KIMORE ficou indisponivel: dominio do laboratorio retorna 403 inteiro,
SharePoint do artigo retorna 404, e o mirror no Zenodo tem apenas CSVs de
features, sem video. Detalhes em [datasets.md](datasets.md).

O Keraal e melhor para este trabalho: video RGB, pacientes clinicos reais de
lombalgia, anotacao de erro por dois medicos com localizacao temporal, e
**esqueletos OpenPose ja calculados**.

### 3. OpenPose nao sera compilado

O enunciado pede OpenPose. Ele nao tem build oficial para Apple Silicon e
exige compilar Caffe. Como o Keraal ja traz os keypoints do OpenPose em
formato COCO, **consumimos esses JSONs diretamente** e o requisito e atendido
sem compilar nada. Para os frames onde precisarmos de pose nova, usar
YOLOv8-pose (ja instalado via ultralytics).

### 4. scispacy foi descartado

Dois motivos. O tecnico: exige numpy<2, o que conflita com o opencv.
O relevante: as transcricoes do TORGO nao tem vocabulario clinico (sao frases
do TIMIT, digitos, alfabeto radiofonico), entao um NER biomedico nao acharia
nada. Termos criticos saem de dicionario curado com `EntityRuler` do spaCy,
que ainda e mais facil de justificar no relatorio. Analise clinica de texto de
verdade fica com os dados do MIMIC.

### 5. Python 3.11, nao 3.14

O sistema tem Python 3.14.5, incompativel com torch, ultralytics e
faster-whisper. A venv usa `/opt/homebrew/bin/python3.11` (3.11.9). Ha quatro
instalacoes de 3.11 na maquina; **sempre usar o caminho absoluto** do
Homebrew.

### 6. Amostra do TORGO e pareada por frase

A amostragem por ordem de arquivo produzia uma comparacao invalida (um unico
locutor por grupo, sexos diferentes, microfones diferentes) e as metricas saiam
com o sinal trocado. O amostrador atual pareia por frase e controla microfone,
sexo e locutor. O racional completo esta em [datasets.md](datasets.md); vale
para a secao de metodologia do relatorio.

## Estado do ambiente

```bash
cd /Users/debs/Personal/ia_para_devs/fiap-fase4
source .venv/bin/activate    # Python 3.11.9
```

Dependencias em [requirements.txt](../requirements.txt), todas instaladas e
com imports verificados. `pip check` limpo. Destaques: torch 2.13 com **MPS
disponivel** (aceleracao Apple Silicon), faster-whisper 1.2.1,
ultralytics 8.4.107, opencv 5.0, spacy 3.8.14 com `en_core_web_sm`,
praat-parselmouth 0.4.7 e jiwer 4.0.

O `.env` e lido por [src/common/config.py](../src/common/config.py). Modulo que
depende de variavel de ambiente deve importar as constantes de la, e nao chamar
`os.getenv` direto, senao o valor do arquivo e ignorado. O cache de modelos
(`HF_HOME`) aponta para `models/hf`, que ja tem os pesos do Whisper large-v3, do
Whisper small, do distilbert de sentimento e os shards do TORGO: 5,1 GB, fora
do versionamento.

### Duas armadilhas do ambiente

**Encoding.** O editor grava arquivos novos em UTF-16, que o Python rejeita com
"source code string cannot contain null bytes". Depois de criar ou editar
arquivo pelo agente, rode `python scripts/normalizar_encoding.py`. Com
`--verificar` ele so aponta, sem alterar, o que serve para hook de commit.

Ha um falso alarme parecido no caminho inverso: a ferramenta de leitura do
agente exibe como texto ilegivel os arquivos que comecam com aspas triplas,
mesmo estando em UTF-8 correto. Antes de sair reescrevendo, confirme com `file`
ou com o proprio `normalizar_encoding.py --verificar`; se acusarem UTF-8, o
arquivo esta bom e basta ler pelo terminal.

**Bus error no transformers.** O `from_pretrained` deixa os pesos como vistas
do arquivo `.safetensors` mapeado em memoria, e nesta combinacao de torch com
Apple Silicon a multiplicacao de matrizes sobre essas vistas derruba o processo
com bus error (SIGBUS). Ler os pesos funciona; so o kernel de GEMM falha. A
saida e copiar os tensores com `clone` logo apos carregar, como esta em
`carregar_classificador_sentimento`. `low_cpu_mem_usage=False` e `dtype`
explicito nao resolvem.

### Git

Remote: `git@github-deborapoh:deborapoh/fiap-fase4.git`

Atencao: a maquina tem duas contas GitHub. A chave padrao autentica como
`deboraoliveira-hub`, que **nao tem permissao** neste repositorio. O remote
usa o alias `github-deborapoh` do `~/.ssh/config`, que autentica como
`deborapoh`. Se o push falhar com "Permission denied", verifique o remote.

## Dados baixados

Tudo em `data/raw/`, ignorado pelo git. Reproduzivel com
`./scripts/download_datasets.sh todos` e `python scripts/download_torgo.py`.

| Frente | Dataset | Estado |
|---|---|---|
| Video | Keraal grupos 1A e 2A | 4,7 GB. 299 videos MP4, 301 JSONs OpenPose, 301 BlazePose, 51 Kinect, 302 anotacoes por medico (2 medicos) |
| Audio | TORGO via Hugging Face | 80 WAVs pareados (40 frases, cada uma dita pelos dois grupos) + `manifesto.csv` |
| Sinais vitais | BIDMC | Extraido, 417 MB. 53 pacientes, `bidmc_csv/bidmc_##_Numerics.csv` com HR, PULSE, RESP e SpO2 a 1 Hz |
| Prescricoes | MIMIC-IV Demo | Extraido, 130 MB. 22 tabelas em `hosp/` (inclui `prescriptions`) e 9 em `icu/` |

Ha tambem `kimore_ex1_apenas_esqueleto.zip` (37 MB), resquicio do mirror do
Zenodo. Contem apenas features de esqueleto do exercicio 1 com scores
clinicos. Pode servir de validacao extra ou ser descartado.

### Formato do Keraal

Mesmo identificador em seis pastas paralelas:

```
videos/      G2A-Anon-CTK-S1-Brest-029.mp4    MJPEG 480x360, 10 fps
openpose/    G2A-OP-CTK-S1-Brest-029.json     keypoints COCO por frame
blazepose/   G2A-BP-CTK-S1-Brest-029.json     33 keypoints 3D
kinect/      G2A-Kinect-CTK-S1-Brest-029.txt  posicoes/orientacoes 3D
annotatorA/  G2A-CTK-S1-Brest-029.anvil       XML UTF-16, medico A
annotatorB/  G2A-CTK-S1-Brest-029.anvil       XML UTF-16, medico B
```

Exercicios: CTK (esconder o rosto), ELK (alongamento lateral), RTK (rotacao
de tronco). Grupo 1A e paciente com lombalgia, 2A e saudavel: essa comparacao
sustenta a deteccao de "movimento fora do padrao".

Anotacoes tem tres trilhas: `Global evaluation` (Correct, Incorrect,
Incomplete, Motionless), `Global error` e `Temporal error`, as duas ultimas
com severidade, tipo de erro e parte do corpo. Os XMLs sao **UTF-16**.

## Pipeline de audio: pronto

Roda com `python scripts/analisar_audio.py` (cerca de 7 minutos nos 80 audios
com o large-v3 em CPU) e escreve `data/processed/audio_metricas.csv`. Aceita
`--modelo` e `--limite N` para teste rapido.

| Modulo | O que faz |
|---|---|
| `src/audio/transcricao.py` | faster-whisper, no lugar do Azure Speech to Text. Guarda tambem a logprob media dos segmentos, que cai quando o modelo tem dificuldade de reconhecer a fala |
| `src/audio/metricas_vocais.py` | jitter, shimmer, HNR, f0, pausa e taxa de fala pelo Praat (parselmouth) |
| `src/audio/analise_texto.py` | sentimento com distilbert e termos criticos por dicionario curado no `EntityRuler`, no lugar do Azure Text Analytics |

O detector de termos trata negacao ("no pain" nao vira alerta) e contracao
("can't breathe" casa como expressao unica). Como o TORGO nao tem vocabulario
clinico, ele e exercitado pelas frases de `EXEMPLOS_CLINICOS`, no proprio
modulo.

### O que sai no CSV

Uma linha por audio, em `data/processed/audio_metricas.csv`. A fusao vai
consumir daqui, entao vale conhecer as colunas:

| Grupo de colunas | Colunas |
|---|---|
| Identificacao | `arquivo`, `grupo`, `genero`, `falante`, `duracao_s` |
| Acustica | `f0_media_hz`, `f0_desvio_hz`, `jitter_local`, `shimmer_local`, `hnr_db`, `proporcao_pausa`, `taxa_fala_silabas_s` |
| Transcricao | `referencia`, `transcricao`, `wer`, `logprob_media`, `proporcao_sem_fala` |
| Texto | `sentimento`, `sentimento_confianca`, `escore_criticidade`, `severidade_maxima`, `categorias`, `termos_criticos`, `n_termos_criticos`, `n_termos_negados` |

O `escore_criticidade` ja vem normalizado de 0 a 1 e e o candidato natural a
entrar como componente de audio no score de risco da fusao.

### Resultados, grupo disartrico contra controle

Comparacao com Mann-Whitney, 40 audios por grupo:

| Metrica | Variacao | p |
|---|---|---|
| WER | +837% (0,242 contra 0,026) | <0,0001 |
| logprob media do Whisper | -49,7% | 0,0022 |
| proporcao de pausa | +35,3% | 0,0014 |
| taxa de fala | -21,1% | 0,196 |
| jitter / shimmer | -22% | 0,004 |
| HNR | +40,5% | <0,0001 |

Os quatro primeiros vao na direcao esperada e sustentam o requisito de detectar
alteracao vocal: a fala disartrica quebra o reconhecedor, reduz a confianca dele
e tem mais pausa.

Jitter, shimmer e HNR aparecem invertidos, e **isso nao e erro de calculo**.
Essas tres medidas sao validadas em vogal sustentada; sobre frase inteira elas
acompanham o estilo de fala. Nos proprios dados a taxa de fala se correlaciona
com o jitter (rho de 0,28 a 0,36) e, invertida, com o HNR (rho de -0,23 a
-0,41), inclusive dentro de um mesmo grupo. Como o controle fala mais rapido,
mede pior. Ou o relatorio traz essa ressalva, ou essas tres metricas ficam de
fora da conclusao. O refinamento possivel e medi-las so em trecho de fonacao
sustentada.

## Pipeline de anomalias: pronto

Roda com `python scripts/detectar_anomalias.py`, cerca de 30 segundos em CPU.
Aceita `--sem-autoencoder` (so Isolation Forest e regras, uns 10 segundos),
`--epocas` e `--semente`. Cobre as duas fontes tabulares numa execucao so.

| Modulo | O que faz |
|---|---|
| `src/anomaly/sinais_vitais.py` | carga do BIDMC, interpolacao dos faltantes, janelas deslizantes de 30 s com passo de 5 s e estatisticas por janela |
| `src/anomaly/modelos.py` | Isolation Forest sobre as estatisticas e autoencoder LSTM sobre a sequencia, ambos com limiar por quantil |
| `src/anomaly/regras.py` | limiares clinicos de alarme, que nomeiam o evento e servem de baseline |
| `src/anomaly/injecao.py` | cinco tipos de anomalia sintetica, usados para validar |
| `src/anomaly/validacao.py` | metricas por janela e por evento |
| `src/anomaly/alertas.py` | junta janelas vizinhas em um alerta com inicio, fim, motivo e gravidade |
| `src/anomaly/prescricoes.py` | cinco regras sobre a evolucao das prescricoes do MIMIC |

### Decisoes desta frente

Vao na metodologia do relatorio, nao nos desvios de escopo.

1. **Validacao por anomalia injetada.** O BIDMC nao tem rotulo de evento. Os
   pacientes de teste recebem eventos sinteticos de forma conhecida
   (dessaturacao, bradicardia, taquicardia, apneia, falha de sensor) e o
   detector e medido contra eles. Isso mede reacao a desvio plausivel e taxa de
   alarme; **nao** mede acuracia clinica em evento espontaneo real.
2. **Divisao por paciente, nao por janela.** Janelas do mesmo paciente sao
   parecidas; divididas ao acaso, o modelo veria em treino quem vai avaliar.
3. **Limiar calibrado fora do ajuste.** 23 pacientes ajustam o modelo, outros 8
   fixam o limiar no quantil 0,99. Calibrado nos proprios dados de ajuste, ele
   sai apertado e a taxa de alarme estoura em paciente novo.
4. **Faltantes interpolados ate 5 s** (413 valores em 102 mil leituras); buraco
   maior descarta a janela, 67 de 4.823. Preencher com media ensinaria o
   detector a tratar perda de sinal como normal.
5. **Sinal congelado exige persistencia.** Canal isolado constante nao e
   evidencia: o SpO2 fica parado por 30 s em 49% das janelas normais. Mesmo
   olhando os dois canais do mesmo aparelho (oximetro da PULSE e SpO2, ECG da
   HR e RESP), sao 5%. So vira alerta com 6 janelas seguidas, cerca de 55 s, o
   que leva os 63 trechos candidatos do BIDMC a 18.
6. **Escore da internacao pelo pior evento ou pela densidade diaria.** Somar
   todos os eventos saturava 140 das 211 internacoes em 1,0: internacao longa
   acumula evento por tempo de exposicao, nao por gravidade.

### Resultados nos sinais vitais

Validacao com 44 eventos injetados em 22 pacientes de teste:

| Detector | AUC | Precisao | Revocacao | Janelas falsas por hora |
|---|---|---|---|---|
| Isolation Forest | 0,903 | 1,000 | 0,519 | 0 |
| autoencoder LSTM | 0,841 | 0,787 | 0,638 | 65 |
| regras clinicas | 0,764 | 0,643 | 0,682 | 144 |

Combinados em alerta, **42 dos 44 eventos sao detectados** (95%), com atraso
mediano de 14 s e 2 alertas fora de evento. O atraso conta do inicio do evento
ate o fechamento da primeira janela que o pega, e o piso e o proprio tamanho
da janela.

As janelas falsas por hora sao pessimistas de proposito: janela sinalizada
logo depois de um evento conta como falso positivo ali, mas na saida real e
absorvida pelo alerta do proprio evento. O custo que a equipe sente sao os 2
alertas fora de evento em 22 pacientes.

Os tres detectores se complementam, e a tabela por tipo mostra onde cada um e
cego. O autoencoder pega apneia (0,98) e dessaturacao (0,94) por janela, que
sao eventos de forma temporal; a floresta vai melhor em taquicardia; e **falha
de sensor so as regras pegam** (0,62 contra 0,02 e 0,00), porque sinal parado
e justamente o que reduz a variacao que os dois modelos usam. Por evento, a
combinacao pega todos os eventos de todos os tipos; so falha de sensor fica
abaixo, em 78%.

Nas series reais, sem injecao: 27 alertas em 53 pacientes, 425 minutos de
monitoramento. Tres pacientes chegam a escore 1,0, e valem como exemplo no
relatorio: o 49 e o 32 tem hipoxemia sustentada pelos 8 minutos inteiros, e o
13 combina bradipneia com sinal congelado.

### Resultados nas prescricoes

18.087 ordens de 250 internacoes, 1.714 eventos sinalizados. Por tipo:

| Regra | Eventos | O que e |
|---|---|---|
| `inconsistencia_temporal` | 753 | termino anterior ao inicio, ordem revista ou cancelada |
| `salto_de_dose` | 565 | mudanca de 5x ou mais entre ordens consecutivas em ate 48 h |
| `dose_atipica` | 280 | z robusto >= 5 contra a distribuicao da coorte |
| `escalonamento_de_via` | 97 | oral para parenteral no mesmo medicamento em ate 24 h |
| `rajada_de_prescricoes` | 19 | medicamentos novos em 6 h acima do percentil 99 |

686 dos 1.714 caem em medicamento de alto risco da lista do ISMP, o que agrava
a severidade. Exemplos bons para o relatorio: midazolam de 100 mg para 0,5 mg
em 8 h, insulina 75 unidades contra mediana de 5 na coorte, cloreto de potassio
de oral para intravenoso em 6 h.

### O que sai em `data/processed/`

| Arquivo | Conteudo |
|---|---|
| `anomalias_vitais.csv` | um alerta por linha: paciente, inicio, fim, tipo, severidade, escore, detectores |
| `anomalias_vitais_resumo.csv` | um paciente por linha, com `escore_risco` para a fusao |
| `anomalias_validacao.csv` | evento injetado a evento injetado, com o que foi detectado |
| `anomalias_prescricoes.csv` | um evento por linha, com o detalhe legivel do que mudou |
| `anomalias_prescricoes_resumo.csv` | uma internacao por linha, com `escore_risco` para a fusao |

Os dois `_resumo` sao a interface com a fusao multimodal. Trazem
`escore_risco` de 0 a 1 na mesma escala do `escore_criticidade` do audio, o que
foi proposital: os pesos de severidade e a saturacao vem de `regras.py` e sao
os mesmos de `analise_texto.py`.

## O que falta fazer

### Fase 2, o pipeline de video

**Video** (`src/video/`): consumir os JSONs OpenPose do Keraal para angulos
articulares, rodar YOLOv8 nos MP4 para objetos e areas criticas, comparar
execucao de paciente contra saudavel e gerar relatorio automatico de desvios.
Validar contra as anotacoes dos medicos (que dao o rotulo verdadeiro, o erro
e a janela temporal).

Comece pelo parser das anotacoes `.anvil`, nao pelo modelo: e ele que define o
alvo. Sao **XML em UTF-16** (`open(..., encoding='utf-16')`), um par de
arquivos por gravacao, um medico em cada. Onde os dois discordam, ou se usa
apenas o consenso, ou se registra a discordancia como incerteza; e uma escolha
que precisa aparecer no relatorio.

Falta tambem o terceiro item do bullet de anomalias do enunciado, "padroes de
movimentacao do paciente durante a internacao", que depende do video: os dois
primeiros (sinais vitais e prescricoes) ja estao entregues.

### Fase 3

Fusao (`src/fusion/`) consolidando os tres sinais em score de risco por
paciente, com regra de alerta. Duas das tres entradas ja existem no formato
certo: `audio_metricas.csv` traz `escore_criticidade` e os dois
`anomalias_*_resumo.csv` trazem `escore_risco`, todos de 0 a 1 e com a mesma
escala de severidade. Falta decidir como as tres frentes se ligam, ja que cada
uma tem uma chave diferente de paciente (locutor do TORGO, paciente do BIDMC,
internacao do MIMIC) e nenhuma delas se refere a mesma pessoa. A saida honesta
e montar um paciente sintetico que combine as tres fontes, e dizer isso no
relatorio.

App Streamlit publicado num Hugging Face Space, que e o que se grava no video
como integracao com nuvem.

### Fase 4

Relatorio tecnico (`reports/`) com fluxo multimodal, modelos por tipo de
dado, resultados, exemplos de anomalias e **a secao de desvios de escopo**
cobrindo as seis decisoes acima. Um desvio documentado e lido como decisao
de engenharia; um desvio silencioso e lido como requisito nao entregue.

Video de ate 15 min no YouTube ou Vimeo demonstrando analise de audio e
video, deteccao e resposta a anomalias, integracao com o servico em nuvem e o
fluxo do alerta.

## Pendencia com o grupo

O trabalho e em grupo e vale 90% da nota de todas as disciplinas da fase.
A remocao do Azure custa dois bullets obrigatorios e nao e decisao de uma
pessoa so. A usuaria foi orientada a alinhar isso com o grupo. Se alguem
conseguir criar conta no Azure, reconsiderar e o caminho mais seguro para a
nota: o tier F0 e gratuito permanente e trava em vez de cobrar.

## Convencoes

Codigo, comentarios e documentacao em portugues, sem acentos em nomes de
arquivo e sem emojis. Mensagens de commit em portugues, explicando o porque
da mudanca. Nada de dados ou credenciais no versionamento.
