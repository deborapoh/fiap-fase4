# Relatório — Tech Challenge Fase 4

## 1. Introdução

Sistema multimodal de monitoramento clínico que integra análise de vídeo (OpenPose + YOLOv8), áudio de consulta (Azure + librosa), detecção de anomalias em sinais vitais, prescrições, evolução clínica e movimentação do paciente, com fusão de scores e alertas para a equipe médica.

## 2. Arquitetura

Ver diagrama no README e no plano do projeto.

## 3. Módulo Vídeo

- OpenPose: keypoints KERAAL (JSON)
- YOLOv8: detecção em frames
- Relatório JSON de desvios articulares

## 4. Módulo Áudio

- Azure STT + Text Analytics
- librosa: F0 e energia (fadiga/disartria)
- Termos críticos PT-BR

## 5. Anomalias

- Vitais: Isolation Forest
- Prescrições: regras dose/opioide
- Notas clínicas: termos críticos
- Movimentação: velocidade OpenPose

## 6. Fusão

Score ponderado e alerta `{severity, message, evidence[]}` via FastAPI.

## 7. Reprodução

`python scripts/run_demo.py --no-yolo`
