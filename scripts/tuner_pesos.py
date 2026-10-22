import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from envs.sghaul_env import SGHaulEnv
from stable_baselines3.common.evaluation import evaluate_policy

def run_tuner():
    # A "Grade" de Fine-Tuning para o NOVO ambiente normalizado
    pesos_w1 = [50.0, 100.0, 150.0]    # Aumentamos DRASTICAMENTE o incentivo de vazão
    pesos_w2 = [50.0, 100.0, 150.0]    # Testamos reduzir a multa seca, já que o acúmulo ajuda
    pesos_w3 = [100.0, 150.0, 200.0]   # Multa de Drop

    resultados = []
    print("--- INICIANDO CAMPEONATO DE PESOS (AJUSTE FINO) ---\n")
    
    for w1 in pesos_w1:
        for w2 in pesos_w2:
            for w3 in pesos_w3:
                # Regra para evitar o "Hackeamento": Drop >= Atraso
                if w3 < w2:
                    continue 
                
                print(f"[Treinando] Testando -> w1: {w1} | w2: {w2} | w3: {w3} ...")
                
                env = SGHaulEnv(w1=w1, w2=w2, w3=w3)
                
                # IA silenciosa
                model = PPO("MlpPolicy", env, verbose=0, n_steps=2048)
                
                # 40.000 a 60.000 passos é ótimo para o Tuner perceber o potencial
                model.learn(total_timesteps=50000)
                
                # Avalia o desempenho
                mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
                print(f"  ↳ Recompensa Média: {mean_reward:.2f} ± {std_reward:.2f}\n")
                
                resultados.append({
                    "w1": w1,
                    "w2": w2,
                    "w3": w3,
                    "recompensa": mean_reward
                })

    print("\n=== PÓDIO FINAL DE PESOS ===")
    resultados_ordenados = sorted(resultados, key=lambda x: x["recompensa"], reverse=True)
    
    for i, res in enumerate(resultados_ordenados[:5]): 
        print(f"{i+1}º Lugar -> w1: {res['w1']} | w2: {res['w2']} | w3: {res['w3']} | Score: {res['recompensa']:.2f}")

if __name__ == "__main__":
    run_tuner()