import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, DDPG

# Oculta warnings de conversão de precisão do Gymnasium
warnings.filterwarnings(
    "ignore",
    message=r".*Box (low|high)'s precision lowered by casting to float32.*",
    category=UserWarning
)

# Adiciona o diretório raiz ao caminho de importação
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

from envs.sghaul_env import SGHaulEnv


def avaliar_agente_rapido(env, model, episodios=5):
    th_urllc_list, th_embb_list = [], []
    sla_urllc_list, sla_embb_list = [], []

    for _ in range(episodios):
        obs, _ = env.reset()
        done = False

        ep_viol_urllc, ep_tot_urllc = 0, 0
        ep_viol_embb, ep_tot_embb = 0, 0
        ep_th_urllc, ep_th_embb = 0.0, 0.0
        passos = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            ep_th_urllc += info.get(
                'throughput_urllc_mbps', 0.0
            ) * 1000.0

            ep_th_embb += info.get(
                'throughput_embb_mbps', 0.0
            ) * 1000.0

            ep_viol_urllc += info.get('violacoes_uc16', 0)
            ep_tot_urllc += info.get('total_pacotes_uc16', 0)

            ep_viol_embb += info.get('violacoes_uc3', 0)
            ep_tot_embb += info.get('total_pacotes_uc3', 0)

            passos += 1
            done = terminated or truncated

        th_urllc_list.append(
            ep_th_urllc / passos if passos > 0 else 0.0
        )

        th_embb_list.append(
            ep_th_embb / passos if passos > 0 else 0.0
        )

        sla_urllc_list.append(
            (1.0 - ep_viol_urllc / ep_tot_urllc) * 100
            if ep_tot_urllc > 0 else 0.0
        )

        sla_embb_list.append(
            (1.0 - ep_viol_embb / ep_tot_embb) * 100
            if ep_tot_embb > 0 else 0.0
        )

    return {
        "SLA URLLC (%)": np.mean(sla_urllc_list),
        "SLA eMBB (%)": np.mean(sla_embb_list),
        "Vazão URLLC (Kbps)": np.mean(th_urllc_list),
        "Vazão eMBB (Kbps)": np.mean(th_embb_list)
    }


if __name__ == "__main__":

    perfis = {
        "1_Conservador": {
            "w1": 50.0, "w2": 500.0, "w4": 50.0
        },
        "2_Equilibrado": {
            "w1": 150.0, "w2": 250.0, "w4": 50.0
        },
        "3_Agressivo": {
            "w1": 300.0, "w2": 100.0, "w4": 50.0
        },
        "4_Ingenuo": {
            "w1": 100.0, "w2": 100.0, "w4": 100.0
        },
        "2A_eMBB_Aumentado": {
            "w1": 175.0, "w2": 250.0, "w4": 50.0
        },
        "2B_eMBB_Reduzido": {
            "w1": 125.0, "w2": 250.0, "w4": 50.0
        },
        "2C_URLLC_Aumentado": {
            "w1": 150.0, "w2": 300.0, "w4": 50.0
        },
        "2D_URLLC_Reduzido": {
            "w1": 150.0, "w2": 200.0, "w4": 50.0
        }
    }

    algoritmos = {
        "PPO": PPO,
        "SAC": SAC,
        "DDPG": DDPG
    }

    timesteps = 150000
    resultados_finais = []

    # Cria pastas para salvar resultados e modelos
    os.makedirs("results", exist_ok=True)

    pasta_modelos = "models/sensitivity_models"
    os.makedirs(pasta_modelos, exist_ok=True)

    print("=== INICIANDO BATERIA COMPLETA DE SENSIBILIDADE ===\n")

    for nome_perfil, pesos in perfis.items():

        print(
            f"\n{'=' * 60}\n"
            f" PERFIL: {nome_perfil}\n"
            f" Pesos: w1={pesos['w1']} | "
            f"w2={pesos['w2']} | w4={pesos['w4']}\n"
            f"{'=' * 60}"
        )

        for nome_algo, AlgoClass in algoritmos.items():

            print(f"\n[{nome_perfil}] Treinando {nome_algo}...")

            env = SGHaulEnv(
                w1=pesos['w1'],
                w2=pesos['w2'],
                w3=10.0,
                w4=pesos['w4']
            )

            # Instancia o modelo
            if nome_algo == "PPO":
                model = AlgoClass(
                    "MlpPolicy",
                    env,
                    learning_rate=0.001,
                    ent_coef=0.1,
                    clip_range=0.2,
                    verbose=0
                )
            else:
                model = AlgoClass(
                    "MlpPolicy",
                    env,
                    learning_rate=0.001,
                    verbose=0
                )

            # Inicia o cronômetro
            start_time = time.time()

            # Treina com barra de progresso
            model.learn(
                total_timesteps=timesteps,
                progress_bar=True
            )

            # Calcula o tempo de treinamento
            end_time = time.time()
            tempo_decorrido = end_time - start_time

            tempo_formatado = time.strftime(
                "%H:%M:%S",
                time.gmtime(tempo_decorrido)
            )

            # Salva o modelo treinado
            caminho_modelo = os.path.join(
                pasta_modelos,
                f"{nome_algo}_{nome_perfil}.zip"
            )

            model.save(caminho_modelo)

            # Avalia o agente
            print(f"Avaliando {nome_algo}...")

            metricas = avaliar_agente_rapido(env, model)

            resultados_finais.append({
                "Perfil": nome_perfil,
                "Algoritmo": nome_algo,
                "Tempo de Treino": tempo_formatado,
                "SLA URLLC (%)": (
                    f"{metricas['SLA URLLC (%)']:.2f}"
                ),
                "SLA eMBB (%)": (
                    f"{metricas['SLA eMBB (%)']:.2f}"
                ),
                "Vazão URLLC (Kbps)": (
                    f"{metricas['Vazão URLLC (Kbps)']:.2f}"
                ),
                "Vazão eMBB (Kbps)": (
                    f"{metricas['Vazão eMBB (Kbps)']:.2f}"
                ),
                "Arquivo do Modelo": caminho_modelo
            })

            print(f" -> Concluído em {tempo_formatado}")

            print(
                f" -> SLA URLLC: "
                f"{metricas['SLA URLLC (%)']:.2f}% | "
                f"SLA eMBB: {metricas['SLA eMBB (%)']:.2f}%"
            )

            env.close()

    # Salva e exibe a tabela final
    df_resultados = pd.DataFrame(resultados_finais)

    caminho_csv = "results/full_sensitivity_analysis.csv"

    df_resultados.to_csv(caminho_csv, index=False)

    print(
        f"\n\n=== TREINAMENTO CONCLUÍDO! "
        f"TABELA SALVA EM {caminho_csv} ==="
    )

    print(df_resultados.to_string(index=False))
