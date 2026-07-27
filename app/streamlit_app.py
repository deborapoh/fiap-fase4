"""Demo do sistema de monitoramento hospitalar multimodal.

Publicado num Hugging Face Space (serviço gerenciado em nuvem), no lugar dos
serviços Azure previstos no enunciado. Consome os CSVs em data/processed/
gerados pelos pipelines locais — a demo mostra fusão, alertas e as três
frentes, sem reprocessar áudio/vídeo na hora.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_AQUI = Path(__file__).resolve().parent
# No Space o app é a raiz; no repo o arquivo fica em app/ e os CSVs em
# data/processed/ na raiz OU em app/data/processed/ (cópia para o Space).
_CANDIDATOS = (
    _AQUI / "data" / "processed",
    _AQUI.parent / "data" / "processed",
)
PROCESSADOS = next((p for p in _CANDIDATOS if p.exists()), _CANDIDATOS[0])

st.set_page_config(
    page_title="Monitoramento multimodal — Fase 4",
    page_icon=None,
    layout="wide",
)


@st.cache_data
def carregar(nome: str) -> pd.DataFrame:
    caminho = PROCESSADOS / nome
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho)


def main() -> None:
    st.title("Monitoramento hospitalar multimodal")
    st.caption(
        "Tech Challenge FIAP — Fase 4. Fusão de áudio (TORGO), vídeo (Keraal), "
        "sinais vitais (BIDMC) e prescrições (MIMIC-IV Demo). Pacientes são "
        "sintéticos: as bases públicas não compartilham a mesma pessoa."
    )

    fusao = carregar("fusao_risco.csv")
    alertas = carregar("fusao_alertas.csv")
    audio = carregar("audio_metricas.csv")
    vitais = carregar("anomalias_vitais_resumo.csv")
    video = carregar("video_desvios_resumo.csv")
    presc = carregar("anomalias_prescricoes_resumo.csv")
    vitais_evt = carregar("anomalias_vitais.csv")
    presc_evt = carregar("anomalias_prescricoes.csv")

    if fusao.empty:
        st.warning(
            "CSVs de fusão não encontrados. Rode os pipelines e "
            "`python scripts/fundir_risco.py` antes de abrir a demo."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pacientes sintéticos", len(fusao))
    c2.metric("Alertas ativos", int(fusao["alerta"].sum()))
    c3.metric("Escore médio", f"{fusao['escore_risco'].mean():.2f}")
    c4.metric("Severidade alta", int((fusao["severidade"] == "alta").sum()))

    aba_alerta, aba_fusao, aba_frentes, aba_fluxo = st.tabs(
        ["Fila de alertas", "Fusão multimodal", "Frentes", "Fluxo do alerta"]
    )

    with aba_alerta:
        st.subheader("Fila para a equipe médica")
        fila = alertas if not alertas.empty else fusao[fusao["alerta"]]
        if fila.empty:
            st.info("Nenhum alerta no momento.")
        else:
            fila = fila.sort_values("escore_risco", ascending=False)
            for _, row in fila.iterrows():
                with st.container(border=True):
                    cols = st.columns([2, 1, 1, 4])
                    cols[0].markdown(f"**{row['id_sintetico']}**")
                    cols[1].markdown(f"risco `{row['escore_risco']:.2f}`")
                    cols[2].markdown(f"**{row['severidade']}**")
                    cols[3].caption(str(row.get("motivos", "")))

    with aba_fusao:
        st.subheader("Escore por frente")
        st.dataframe(
            fusao.sort_values("escore_risco", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(
            fusao.set_index("id_sintetico")[
                ["escore_audio", "escore_vitais", "escore_video",
                 "escore_prescricoes"]
            ]
        )

    with aba_frentes:
        c_a, c_v = st.columns(2)
        with c_a:
            st.markdown("#### Áudio (TORGO)")
            if audio.empty:
                st.caption("sem dados")
            else:
                resumo = audio.groupby("grupo").agg(
                    n=("arquivo", "count"),
                    wer=("wer", "mean"),
                    pausa=("proporcao_pausa", "mean"),
                    criticidade=("escore_criticidade", "mean"),
                ).round(3)
                st.dataframe(resumo, use_container_width=True)
        with c_v:
            st.markdown("#### Vídeo (Keraal)")
            if video.empty:
                st.caption("sem dados")
            else:
                resumo = video.groupby(["grupo", "avaliacao"]).size().unstack(
                    fill_value=0
                )
                st.dataframe(resumo, use_container_width=True)
                st.caption(
                    f"Alertas de movimento: {int(video['alerta'].sum())} / "
                    f"{len(video)}"
                )

        c_s, c_p = st.columns(2)
        with c_s:
            st.markdown("#### Sinais vitais (BIDMC)")
            if vitais.empty:
                st.caption("sem dados")
            else:
                st.dataframe(
                    vitais.sort_values("escore_risco", ascending=False).head(10),
                    use_container_width=True, hide_index=True,
                )
            if not vitais_evt.empty:
                with st.expander("Eventos recentes"):
                    st.dataframe(vitais_evt.head(20), use_container_width=True,
                                 hide_index=True)
        with c_p:
            st.markdown("#### Prescrições (MIMIC)")
            if presc.empty:
                st.caption("sem dados")
            else:
                st.dataframe(
                    presc.sort_values("escore_risco", ascending=False).head(10),
                    use_container_width=True, hide_index=True,
                )
            if not presc_evt.empty:
                with st.expander("Eventos recentes"):
                    cols = [c for c in ("hadm_id", "drug", "tipo", "severidade",
                                        "detalhe", "escore")
                            if c in presc_evt.columns]
                    st.dataframe(presc_evt[cols].head(20),
                                 use_container_width=True, hide_index=True)

    with aba_fluxo:
        st.subheader("Fluxo do alerta")
        st.markdown(
            """
1. **Coleta** — áudio (Whisper + métricas vocais), vídeo (OpenPose + YOLOv8),
   sinais vitais (BIDMC) e prescrições (MIMIC).
2. **Detecção** — cada frente gera um escore em \\[0, 1\\] e alertas locais.
3. **Fusão** — paciente sintético combina as quatro entradas com pesos
   (áudio 0,20 · vitais 0,30 · vídeo 0,30 · prescrições 0,20).
4. **Alerta à equipe** — severidade alta se o risco fundido >= 0,75 ou se
   duas ou mais frentes disparam; a fila desta demo é o que a equipe vê.
5. **Nuvem** — este app roda num Hugging Face Space (serviço gerenciado),
   no lugar do Azure previsto no enunciado.
            """
        )
        paciente = st.selectbox(
            "Simular acompanhamento de um paciente",
            fusao["id_sintetico"].tolist(),
        )
        row = fusao[fusao["id_sintetico"] == paciente].iloc[0]
        st.write(
            f"**Risco fundido:** {row['escore_risco']:.2f} · "
            f"**Severidade:** {row['severidade']} · "
            f"**Alerta:** {'SIM' if row['alerta'] else 'não'}"
        )
        st.progress(min(1.0, float(row["escore_risco"])))
        st.json({
            "audio": row["escore_audio"],
            "vitais": row["escore_vitais"],
            "video": row["escore_video"],
            "prescricoes": row["escore_prescricoes"],
            "motivos": row.get("motivos", ""),
            "fontes": {
                "audio": row.get("id_audio"),
                "vitais": row.get("id_vitais"),
                "video": row.get("id_video"),
                "prescricao": row.get("id_prescricao"),
            },
        })


if __name__ == "__main__":
    main()
