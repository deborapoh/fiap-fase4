# Datasets

Registro das fontes de dados usadas, incluindo o que foi tentado e falhou.
Esta pagina alimenta a secao de desvios de escopo do relatorio tecnico.

## Resumo

| Frente | Dataset | Situacao | Tamanho local |
|---|---|---|---|
| Video | Keraal (IMT Atlantique / CHRU Brest) | Baixado | 4,7 GB |
| Audio | TORGO (University of Toronto), via Hugging Face | Baixado, amostra pareada de 80 audios | 25 MB |
| Sinais vitais | BIDMC PPG and Respiration (PhysioNet) | Baixado e extraido | 417 MB |
| Prescricoes | MIMIC-IV Clinical Database Demo (PhysioNet) | Baixado e extraido | 130 MB |

Nenhum exige credenciamento. O comando `./scripts/download_datasets.sh todos`
reproduz os downloads; `python scripts/download_torgo.py` refaz a amostra de audio.

## Video: Keraal, no lugar do KIMORE

O plano inicial era usar o KIMORE, citado na maior parte da literatura de
avaliacao de exercicios de reabilitacao. **Ele esta indisponivel**:

- O site do laboratorio (`vrai.dii.univpm.it`) retorna HTTP 403 em todo o
  dominio, nao apenas na pagina do dataset.
- O link do SharePoint citado no artigo original retorna HTTP 404.
- O mirror no Zenodo (`10.5281/zenodo.20262128`) contem apenas dois CSVs de
  features de esqueleto ja processadas do exercicio 1, sem nenhum frame de
  video. Inutil para YOLOv8, que precisa da imagem.

O [Keraal](https://keraal.enstb.org/KeraalDataset.html) substitui com vantagem.
Foi coletado em um estudo clinico controlado em dois centros de reabilitacao na
Bretanha, com pacientes reais de lombalgia cronica, e traz:

- **Video RGB** em MP4, que e o que o YOLOv8 consome.
- **Esqueletos OpenPose ja calculados** em formato COCO. Isso atende ao
  requisito de OpenPose do enunciado sem precisar compilar o OpenPose, que nao
  tem build oficial para Apple Silicon.
- Esqueletos Kinect 3D e BlazePose como alternativas.
- **Anotacoes de dois medicos independentes**, indicando se a execucao esta
  correta, qual o erro, qual parte do corpo o causou e em que janela de tempo.

Licenca CC-BY-NC-SA, uso academico permitido.

### Grupos baixados

| Grupo | Sujeitos | Gravacoes | Anotado |
|---|---|---|---|
| 2A | 6 adultos saudaveis | 51 | Sim |
| 1A | 6 pacientes com lombalgia cronica | 249 | Sim |

A comparacao saudavel contra paciente e o que sustenta a deteccao de
"movimento fora do padrao" pedida no enunciado.

Distribuicao das avaliacoes no grupo 2A: 35 corretas, 11 incorretas,
4 incompletas, 1 sem movimento.

### Formato

Cada gravacao aparece com o mesmo identificador em seis pastas:

```
videos/       G2A-Anon-CTK-S1-Brest-029.mp4    MJPEG 480x360, 10 fps
openpose/     G2A-OP-CTK-S1-Brest-029.json     keypoints COCO por frame
blazepose/    G2A-BP-CTK-S1-Brest-029.json     33 keypoints 3D por frame
kinect/       G2A-Kinect-CTK-S1-Brest-029.txt  posicoes e orientacoes 3D
annotatorA/   G2A-CTK-S1-Brest-029.anvil       XML UTF-16 do medico A
annotatorB/   G2A-CTK-S1-Brest-029.anvil       XML UTF-16 do medico B
```

Os tres exercicios sao CTK (esconder o rosto), ELK (alongamento lateral) e
RTK (rotacao de tronco).

As anotacoes tem tres trilhas: `Global evaluation` (Correct, Incorrect,
Incomplete, Motionless), `Global error` e `Temporal error`, as duas ultimas com
severidade, tipo do erro e parte do corpo responsavel.

## Audio: TORGO

Fala de pessoas com disartria (paralisia cerebral ou ELA) e um grupo controle
saudavel. Usamos o espelho `abnerh/TORGO-database` no Hugging Face, que e
aberto e ja traz transcricao, genero, duracao e o rotulo do grupo.

O dataset tem 1,5 GB em quatro shards parquet e vem **ordenado por locutor**:
todos os controles saudaveis aparecem antes dos disartricos. Por isso o script
baixa os shards e amostra localmente, em vez de usar streaming, que teria que
percorrer quase o dataset inteiro para encontrar o segundo grupo.

A amostra padrao e de 40 audios por grupo, WAV PCM 16 bits mono a 16 kHz, que e
o formato nativo esperado pelos modelos de transcricao.

### Por que a amostra e pareada

A primeira versao do script pegava os N primeiros audios de cada grupo. Isso
gerou uma comparacao invalida, e o erro so apareceu quando as metricas sairam
com o sinal trocado: o grupo disartrico apresentava **menos** jitter e shimmer
e HNR 90% **maior** que o controle, ou seja, voz aparentemente mais saudavel.

A causa e a ordenacao do dataset. Os N primeiros de cada grupo caiam numa unica
locutora saudavel (FC01, microfone de array) contra um unico locutor disartrico
(M05, microfone de cabeca). Havia tres fontes de confusao empilhadas:

- **Microfone**: o de array fica longe da boca e capta ruido de sala, o que
  derruba o HNR e infla jitter e shimmer do grupo saudavel.
- **Sexo**: uma mulher contra um homem explica sozinho a diferenca de f0
  (190 Hz contra 121 Hz).
- **Locutor unico**: nenhuma variabilidade individual nos dois lados.

O amostrador atual controla as tres. So usa `headMic`, presente nos dois
grupos; so usa frases ditas pelos dois grupos, entrando com um audio de cada
lado, de modo que o conteudo fonetico e identico; e distribui a cota por sexo e
por locutor, com rodizio. O resultado sao 40 frases de 7 a 14 palavras, cada
uma falada pelos dois grupos, cobrindo os 15 locutores do corpus com a mesma
proporcao de homens e mulheres nos dois lados.

Fica registrado como aprendizado metodologico: em corpus clinico ordenado por
sujeito, amostragem por ordem de arquivo mede o protocolo de gravacao, nao a
patologia.

Vale registrar uma limitacao: **as transcricoes do TORGO nao tem vocabulario
clinico**. Os participantes leem frases do TIMIT, digitos e alfabeto
radiofonico. Por isso a identificacao de termos criticos nao usa NER biomedico
(que nao encontraria nada nesse texto) e sim um dicionario curado, e a analise
de texto clinico de verdade e feita sobre os dados do MIMIC.

## Sinais vitais: BIDMC

53 gravacoes de 8 minutos de pacientes de UTI, extraidas do MIMIC-II. Cada uma
traz frequencia cardiaca, frequencia respiratoria e SpO2 amostrados a 1 Hz, alem
de ECG e PPG a 125 Hz e anotacoes manuais de respiracao feitas por dois
anotadores. Vem em CSV, WFDB e MATLAB.

Extraido em `data/raw/bidmc/bidmc-ppg-and-respiration-dataset-1.0.0/`. Para a
deteccao de anomalias interessam os 53 arquivos
`bidmc_csv/bidmc_##_Numerics.csv`, com as colunas `Time [s], HR, PULSE, RESP,
SpO2`, 481 linhas cada. Os sinais de alta frequencia estao nos `*_Signals.csv`
e nos arquivos WFDB (`.dat`/`.hea`), que a lib `wfdb` le direto. Idade, sexo e
unidade de internacao de cada paciente estao nos `bidmc_##_Fix.txt`.

Tres armadilhas, todas ja tratadas em `src/anomaly/sinais_vitais.py`:

- **Os nomes das colunas tem espaco a esquerda** (` HR`, ` PULSE`). Sem
  `skipinitialspace=True` o acesso por nome falha.
- Ha faltantes, concentrados em `PULSE` e `SpO2`: 413 valores em 102 mil
  leituras, sempre em trechos curtos de perda do oximetro.
- **Nao ha rotulo de anomalia.** Nenhum evento vem anotado, o que impede medir
  acerto diretamente. Por isso a validacao usa anomalia injetada, decisao
  descrita em [handoff.md](handoff.md).

## Prescricoes: MIMIC-IV Demo

Subconjunto aberto de 100 pacientes do MIMIC-IV, com o mesmo schema da base
completa. Traz `prescriptions`, `pharmacy` e `poe`, que sustentam o requisito de
detectar alteracoes inesperadas no tratamento, alem de `chartevents` e
`transfers` para o contexto clinico.

Extraido em `data/raw/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2/`, com
as tabelas em `.csv.gz` divididas entre `hosp/` (22 tabelas, inclui
`prescriptions`) e `icu/` (9 tabelas). O pandas le o `.csv.gz` direto, sem
descompactar.

A `prescriptions` tem 18.087 ordens de 250 internacoes de 100 pacientes, com
631 medicamentos distintos. Detalhes que afetam o processamento:

- `dose_val_rx` e texto e as vezes vem como faixa, entao precisa de conversao
  antes de qualquer conta.
- **753 ordens tem `stoptime` anterior ao `starttime`**, 4% do total. Nao e
  erro de leitura: e ordem revista ou cancelada depois de emitida, e o pipeline
  a trata como evento de baixa severidade.
- As datas sao deslocadas para o futuro pelo processo de desidentificacao do
  MIMIC, o que nao atrapalha: todas as regras olham intervalo, nao data
  absoluta.

## Citacoes obrigatorias

- Nguyen, S. M., Devanne, M., Remy-Neris, O., Lempereur, M., Thepaut, A. (2024).
  A Medical Low-Back Pain Physical Rehabilitation Database for Human Body
  Movement Analysis. IJCNN.
- Rudzicz, F., Namasivayam, A. K., Wolff, T. (2012). The TORGO database of
  acoustic and articulatory speech from speakers with dysarthria. Language
  Resources and Evaluation, 46(4), 523-541.
- Pimentel, M. A. F. et al. (2016). Toward a Robust Estimation of Respiratory
  Rate from Pulse Oximeters. IEEE TBME, 64(8), 1914-1923.
- Johnson, A. E. W. et al. (2023). MIMIC-IV, a freely accessible electronic
  health record dataset. Scientific Data.
