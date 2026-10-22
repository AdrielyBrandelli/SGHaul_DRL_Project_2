import os
import sys
import random
import csv
import numpy as np
from stable_baselines3 import PPO

# Caminho absoluto blindado para a raiz do seu projeto
sys.path.append(r"C:\Users\adriely\Desktop\SGHaul_DRL_Project_Testes")
from envs.sghaul_env import SGHaulEnv

def avaliar_rapido(env, semente, model=None, acao_estatica=None):
    random.seed(semente)
    np.random.seed(semente)
    obs, info = env.reset(seed=semente)
    
    done = False
    violacoes = 0
    dados_mb = 0.0
    
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = acao_estatica

        obs, _, terminated, truncated, info = env.step(action)
        violacoes += info.get('violacoes_uc16', 0)
        
        vazao_atual = info.get('throughput_mbps', 0.0)
        if vazao_atual > 0: dados_mb += vazao_atual 
            
        done = terminated or truncated
        
    return dados_mb, violacoes

# Função para criar uma barra de progresso visual no terminal
def print_progresso(iteracao, total, prefixo='', sufixo='', tamanho=40, preenchimento='█'):
    porcentagem = f"{100 * (iteracao / float(total)):.1f}"
    preenchido = int(tamanho * iteracao // total)
    barra = preenchimento * preenchido + '-' * (tamanho - preenchido)
    sys.stdout.write(f'\r{prefixo} |{barra}| {porcentagem}% {sufixo}')
    sys.stdout.flush()
    if iteracao == total:
        print()

def main():
    print("=== MINERADOR DE SEMENTES PERFEITAS ===")
    env = SGHaulEnv()
    
    caminho_modelo = r"C:\Users\adriely\Desktop\SGHaul_DRL_Project_Testes\models\ppo_sghaul_slicing.zip"
    try: 
        model = PPO.load(caminho_modelo, env=env)
        print("Modelo carregado com sucesso!\n")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    sementes_aprovadas = []
    tentativas = 500 
    
    print(f"Vasculhando {tentativas} sementes em busca da crise perfeita...\n")
    
    for i in range(tentativas):
        # Atualiza a barra de progresso no terminal
        print_progresso(i, tentativas, prefixo='Progresso:', sufixo=f'| Achados: {len(sementes_aprovadas)}')
        
        semente = random.randint(1, 100000)
        
        # 1. Testa o 50/50 primeiro
        vaz_50, vio_50 = avaliar_rapido(env, semente, acao_estatica=np.array([0.0]))
        
        # FILTRO EXIGENTE: Pula qualquer cenário que não teve MAIS de 3 violações
        if vio_50 < 1:
            continue
            
        # 2. Testa a IA apenas nos cenários de crise
        vaz_ia, vio_ia = avaliar_rapido(env, semente, model=model)
        
        # printa o que a IA está fazendo para entendermos
        if vio_ia == 0:
            # Limpa a barra 
            sys.stdout.write('\r' + ' '*100 + '\r') 
            print(f"QUASE: Semente {semente} | B50 falhou {vio_50}x | Vazão IA: {vaz_ia:.2f} (B50 fez {vaz_50:.2f})")
            
        if vio_ia == 0 and vaz_ia >= (vaz_50 - 0.5):
            diferenca_violacoes = vio_50 - vio_ia
            sementes_aprovadas.append([semente, vio_50, vio_ia, round(vaz_50, 2), round(vaz_ia, 2), diferenca_violacoes])
            
            sys.stdout.write('\r' + ' '*100 + '\r') 
            print(f"Semente: {semente} | B50 Erros: {vio_50} | IA Erros: {vio_ia} | Vazão IA: {vaz_ia:.2f} Mb")
            
    # Garante que a barra bata 100% ao finalizar o loop
    print_progresso(tentativas, tentativas, prefixo='Progresso:', sufixo=f'| Achados: {len(sementes_aprovadas)}')

    if not sementes_aprovadas:
        print("\nNenhuma semente atendeu ao filtro (B50 > 3 erros, IA = 0 erros, IA Vazão >= B50 Vazão).")
        print("Talvez o cenário atual do emulador precise ser um pouco mais estressante, ou rode mais tentativas!")
        return

    # Salva no CSV ordenando pelas sementes onde o 50/50 mais sofreu
    sementes_aprovadas.sort(key=lambda x: x[5], reverse=True)
    
    os.makedirs("results", exist_ok=True)
    with open("results/melhores_sementes.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Semente', 'Violacoes_50_50', 'Violacoes_IA', 'Vazao_50_50', 'Vazao_IA', 'Diferenca_Falhas'])
        writer.writerows(sementes_aprovadas)
        
    print(f"\nBusca concluída! {len(sementes_aprovadas)} sementes perfeitas salvas em 'results/melhores_sementes.csv'.")

if __name__ == "__main__":
    main()