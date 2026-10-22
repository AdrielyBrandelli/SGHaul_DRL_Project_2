import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.sghaul_env import SGHaulEnv

env = SGHaulEnv()
env.reset()

print("Rodando 1.000 passos (100 segundos)... Aguarde.")
print("Mostrando apenas os momentos em que a vazão for maior que zero:\n")

total_vazao = 0.0

for i in range(1000):
    obs, reward, term, trunc, info = env.step([0.5, 0.5])
    vazao_real = info.get('throughput_mbps')
    violacao = info.get('violacoes_uc16')
    
    if vazao_real > 0:
        print(f"Step {i+1} | Tivemos Vazão! -> {vazao_real:.4f} Mbps | Violações: {violacao}")
        total_vazao += vazao_real

print(f"\n--- Fim do Episódio ---")
print(f"Vazão total acumulada nos 1000 steps: {total_vazao:.4f} Mbps")