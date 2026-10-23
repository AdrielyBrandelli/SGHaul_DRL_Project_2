import sys
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Garante que o Python encontre a pasta 'envs' mesmo rodando de dentro de 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from envs.sghaul_env import SGHaulEnv
import threading
import ctypes
import warnings
warnings.filterwarnings("ignore")

# --- 1. PREVENÇÃO DE SUSPENSÃO ---
#def anti_idle_ninja():
#    while True:
#        ctypes.windll.user32.keybd_event(0x7E, 0, 0, 0)
#        ctypes.windll.user32.keybd_event(0x7E, 0, 2, 0)
#        time.sleep(180)

#threading.Thread(target=anti_idle_ninja, daemon=True).start()

# --- 2. BASELINE HEURÍSTICO ---
class QueueProportionalHeuristic:
    def predict(self, obs, deterministic=True):
        if len(obs.shape) > 1: obs = obs[0]
        
        # 1. Lê as latências exatas do estado (URLLC, eMBB, mMTC)
        lats = np.array([max(0, obs[0]), max(0, obs[1]), max(0, obs[2])])
        t = np.sum(lats)
        
        # 2. Caso de contingência: rede sem latência (distribuição uniforme)
        if t == 0:
            return np.array([0.0, 0.0, 0.0]), None
            
        # 3. Proporção desejada
        p = lats / t
        
        # 4. Inversão do Softmax via logaritmo
        epsilon = 1e-5
        p = np.clip(p, epsilon, 1.0)
        a = np.log(p)
        
        # 5. Ancoragem de translação dentro do action_space [-1, 1]
        a = a - np.max(a) + 1.0
        a = np.clip(a, -1.0, 1.0)
        
        return a, None

def evaluate_model_detailed(modelo, env, seeds_list):
    resultados = []
    for s in seeds_list:
        np.random.seed(s) 
        obs, _ = env.reset(seed=s) 
        topologia_cenario = getattr(env.unwrapped, 'topologia_atual', "Desconhecida")
        v_u, t_u, v_e, t_e = 0, 0, 0, 0
        lats_u = []
        tempos_inferencia = []
        consumos_energia = []
        custos_operacionais = []
        
        for _ in range(2000):
            t_inicio = time.perf_counter()
            action, _ = modelo.predict(obs, deterministic=True)
            t_fim = time.perf_counter()
            tempos_inferencia.append((t_fim - t_inicio) * 1000.0) 
            
            obs, reward, terminated, truncated, info = env.step(action)
            v_u += info['violacoes_uc16']
            t_u += info.get('total_pacotes_uc16', 0)
            v_e += info['violacoes_uc3']
            t_e += info.get('total_pacotes_uc3', 0)
            lats_u.append(info['e2e_latency_uc16'])
            
            # Coleta de consumo e custo
            if 'consumo_energia' in info:
                consumos_energia.append(info['consumo_energia'])
            if 'custo_operacional' in info:
                custos_operacionais.append(info['custo_operacional'])

            if terminated or truncated: break
                
        sla_u = max(0, 100.0 * (1 - (v_u / max(1, t_u))))
        sla_e = max(0, 100.0 * (1 - (v_e / max(1, t_e))))
        lat_u = np.mean(lats_u) if lats_u else 0.0
        inferencia_media = np.mean(tempos_inferencia)
        consumo_medio = np.mean(consumos_energia) if consumos_energia else 0.0
        custo_medio = np.mean(custos_operacionais) if custos_operacionais else 0.0
        
        resultados.append({
            'eval_seed': s, 'topologia': topologia_cenario, 
            'sla_u': sla_u, 'sla_e': sla_e, 
            'lat_u': lat_u, 'inferencia_ms': inferencia_media,
            'consumo_kw': consumo_medio, 'custo_op': custo_medio
        })
    return resultados

def pipeline_definitivo_artigo(max_steps=80000):
    # Caminho base apontando direto para o seu Google Drive
    PASTA_DRIVE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resultados"))
    os.makedirs(PASTA_DRIVE, exist_ok=True)
    
    # Criando as pastas dentro do seu Drive em tempo real
    os.makedirs(f"{PASTA_DRIVE}/data", exist_ok=True)
    os.makedirs(f"{PASTA_DRIVE}/plots", exist_ok=True)
    pasta_modelos = f"{PASTA_DRIVE}/models/final_pipeline"
    os.makedirs(pasta_modelos, exist_ok=True)
    
    arq_val = f"{PASTA_DRIVE}/data/validation_results_detailed.csv"

    perfis = [
        {'nome': 'Balanceado',  'w1': 100.0, 'w2': 100.0, 'w3': 50.0, 'w4': 50.0},
        {'nome': 'Foco_eMBB',   'w1': 300.0, 'w2': 50.0,  'w3': 50.0, 'w4': 50.0},
        {'nome': 'Foco_URLLC',  'w1': 50.0,  'w2': 300.0, 'w3': 50.0, 'w4': 50.0},
        {'nome': 'Eco_Saver',   'w1': 100.0, 'w2': 100.0, 'w3': 200.0, 'w4': 200.0},
        {'nome': 'High_Stakes', 'w1': 300.0, 'w2': 300.0, 'w3': 100.0, 'w4': 100.0}
    ]
    nomes_perfis = [p['nome'] for p in perfis]
    modelos_ai = ["SAC", "PPO", "DDPG"]
    
    TRAIN_SEEDS = [10, 20, 30]
    VAL_SEEDS = [101, 102, 103]   
    TEST_SEEDS = list(range(201, 221)) 

    if os.path.exists(arq_val):
        df_val = pd.read_csv(arq_val)
        print(f"[*] Retomando progresso... ({len(df_val)} avaliações de validação salvas)")
    else:
        # Colunas atualizadas com tempo de treino, consumo e custo
        df_val = pd.DataFrame(columns=[
            'algoritmo', 'perfil', 'train_seed', 'eval_seed', 
            'topologia', 'sla_u', 'sla_e', 'lat_u', 'inferencia_ms',
            'tempo_treino_s', 'consumo_kw', 'custo_op'
        ])

    # ==========================================
    # FASE 1 & 2: TREINO E VALIDAÇÃO
    # ==========================================
    for algo in modelos_ai:
        for perfil in perfis:
            for tr_seed in TRAIN_SEEDS:
                caminho_modelo = f"{pasta_modelos}/{algo}_{perfil['nome']}_seed-{tr_seed}.zip"
                ja_avaliado = df_val[(df_val['algoritmo'] == algo) & (df_val['perfil'] == perfil['nome']) & (df_val['train_seed'] == tr_seed)]
                
                if len(ja_avaliado) == len(VAL_SEEDS) and os.path.exists(caminho_modelo): 
                    continue
                if not ja_avaliado.empty:
                    df_val = df_val.drop(ja_avaliado.index)

                #env_treino = SGHaulEnv(w1=perfil['w1'], w2=perfil['w2'], w3=perfil['w3'], w4=perfil['w4'])
                # Empacota os parâmetros
                env_kwargs = {'w1': perfil['w1'], 'w2': perfil['w2'], 'w3': perfil['w3'], 'w4': perfil['w4']}

                # Cria 8 simulações simultâneas usando os 8 núcleos físicos do i7
                env_treino = make_vec_env(SGHaulEnv, n_envs=8, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv, seed=tr_seed)

                env_eval = SGHaulEnv(w1=perfil['w1'], w2=perfil['w2'], w3=perfil['w3'], w4=perfil['w4'])
                np.random.seed(tr_seed)
                
                if algo == "SAC": mod = SAC("MlpPolicy", env_treino, verbose=0, seed=tr_seed)
                elif algo == "PPO": mod = PPO("MlpPolicy", env_treino, verbose=0, seed=tr_seed)
                elif algo == "DDPG": mod = DDPG("MlpPolicy", env_treino, action_noise=NormalActionNoise(mean=np.zeros(3), sigma=0.1*np.ones(3)), verbose=0, seed=tr_seed)
                
                stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=4, min_evals=3, verbose=0)
                eval_callback = EvalCallback(env_eval, eval_freq=5000, callback_after_eval=stop_train_callback, verbose=0)

                print(f"\n Treinando {algo} (Perfil: {perfil['nome']} | seed={tr_seed})")
                
                # Medição do tempo de treino
                t_inicio_treino = time.perf_counter()
                mod.learn(total_timesteps=max_steps, callback=eval_callback, progress_bar=True)
                t_fim_treino = time.perf_counter()
                tempo_treino_s = t_fim_treino - t_inicio_treino
                
                mod.save(caminho_modelo)
                
                resultados_val = evaluate_model_detailed(mod, env_eval, VAL_SEEDS)
                
                novos_dados = []
                for res in resultados_val:
                    novos_dados.append({
                        'algoritmo': algo, 'perfil': perfil['nome'], 'train_seed': tr_seed, 
                        'eval_seed': res['eval_seed'], 'topologia': res['topologia'], 
                        'sla_u': res['sla_u'], 'sla_e': res['sla_e'], 
                        'lat_u': res['lat_u'], 'inferencia_ms': res['inferencia_ms'],
                        'tempo_treino_s': tempo_treino_s,
                        'consumo_kw': res['consumo_kw'], 'custo_op': res['custo_op']
                    })
                df_val = pd.concat([df_val, pd.DataFrame(novos_dados)], ignore_index=True)
                df_val.to_csv(arq_val, index=False)

    # ==========================================
    # FASE 3: SELEÇÃO MULTI-OBJETIVO E TESTE NO HOLD-OUT
    # ==========================================
    print("\n" + "="*80)
    print(" SELEÇÃO MULTI-OBJETIVO E TESTE NO HOLD-OUT (20 CENÁRIOS)")
    print("="*80)
    
    dados_teste_bruto = []
    env_test_heuristica = SGHaulEnv()
    heuristica = QueueProportionalHeuristic()
    res_h = evaluate_model_detailed(heuristica, env_test_heuristica, TEST_SEEDS)
    
    for r in res_h:
        dados_teste_bruto.append({
            'algoritmo': 'Heuristica', 'perfil': 'N/A', 'train_seed': 0, 
            'eval_seed': r['eval_seed'], 'topologia': r['topologia'], 
            'sla_u': r['sla_u'], 'sla_e': r['sla_e'], 
            'lat_u': r['lat_u'], 'inferencia_ms': r['inferencia_ms'],
            'tempo_treino_s': 0.0, 'consumo_kw': r['consumo_kw'], 'custo_op': r['custo_op']
        })
        
    resultados_finais = [{
        'Modelo': 'Heuristica', 'Melhor Perfil': 'N/A (Fixo)', 
        'SLA URLLC': np.mean([r['sla_u'] for r in res_h]), 
        'SLA eMBB': np.mean([r['sla_e'] for r in res_h]), 
        'Lat URLLC (ms)': np.mean([r['lat_u'] for r in res_h]),
        'Inf E2 (ms)': np.mean([r['inferencia_ms'] for r in res_h]),
        'Consumo Medio': np.mean([r['consumo_kw'] for r in res_h]),
        'Custo Medio': np.mean([r['custo_op'] for r in res_h])
    }]

    for algo in modelos_ai:
        df_algo = df_val[df_val['algoritmo'] == algo]
        medias_redes = df_algo.groupby(['perfil', 'train_seed']).mean(numeric_only=True).reset_index()
        medias_perfis = medias_redes.groupby('perfil').mean(numeric_only=True)
        
        perfis_viaveis = medias_perfis[medias_perfis['sla_e'] >= 90.0]
        if not perfis_viaveis.empty:
            melhor_perfil = perfis_viaveis['sla_u'].idxmax()
        else:
            melhor_perfil = (medias_perfis['sla_u'] + medias_perfis['sla_e']).idxmax()
            
        print(f"🎯 Melhor combinação para {algo}: {melhor_perfil}")

        perfil_dict = next(p for p in perfis if p['nome'] == melhor_perfil)
        env_test_ia = SGHaulEnv(w1=perfil_dict['w1'], w2=perfil_dict['w2'], w3=perfil_dict['w3'], w4=perfil_dict['w4'])

        res_teste_total = []
        for tr_seed in TRAIN_SEEDS:
            mod_campeao = eval(algo).load(f"{pasta_modelos}/{algo}_{melhor_perfil}_seed-{tr_seed}.zip")
            res_detalhado = evaluate_model_detailed(mod_campeao, env_test_ia, TEST_SEEDS)
            res_teste_total.extend(res_detalhado)
            
            # Obtém o tempo de treino médio para essa semente específica na validação
            tempo_treino_semente = df_algo[(df_algo['perfil'] == melhor_perfil) & (df_algo['train_seed'] == tr_seed)]['tempo_treino_s'].mean()
            
            for r in res_detalhado:
                dados_teste_bruto.append({
                    'algoritmo': algo, 'perfil': melhor_perfil, 'train_seed': tr_seed, 
                    'eval_seed': r['eval_seed'], 'topologia': r['topologia'], 
                    'sla_u': r['sla_u'], 'sla_e': r['sla_e'], 
                    'lat_u': r['lat_u'], 'inferencia_ms': r['inferencia_ms'],
                    'tempo_treino_s': tempo_treino_semente, 
                    'consumo_kw': r['consumo_kw'], 'custo_op': r['custo_op']
                })
        
        resultados_finais.append({
            'Modelo': algo, 'Melhor Perfil': melhor_perfil, 
            'SLA URLLC': np.mean([r['sla_u'] for r in res_teste_total]), 
            'SLA eMBB': np.mean([r['sla_e'] for r in res_teste_total]), 
            'Lat URLLC (ms)': np.mean([r['lat_u'] for r in res_teste_total]),
            'Inf E2 (ms)': np.mean([r['inferencia_ms'] for r in res_teste_total]),
            'Consumo Medio': np.mean([r['consumo_kw'] for r in res_teste_total]),
            'Custo Medio': np.mean([r['custo_op'] for r in res_teste_total])
        })

    pd.DataFrame(dados_teste_bruto).to_csv(f"{PASTA_DRIVE}/data/test_results_detailed.csv", index=False)
    print("✅ Dados brutos do teste final salvos em: data/test_results_detailed.csv")

    print(f"\n{'Algoritmo':<12} | {'Perfil':<15} | {'SLA URL (%)':<11} | {'SLA eMBB (%)':<12} | {'Lat (ms)':<8} | {'Inf (ms)':<8} | {'Consumo':<8} | {'Custo':<8}")
    print("-" * 95)
    for res in resultados_finais:
        print(f"{res['Modelo']:<12} | {str(res['Melhor Perfil']):<15} | {res['SLA URLLC']:>11.2f} | {res['SLA eMBB']:>12.2f} | {res['Lat URLLC (ms)']:>8.1f} | {res['Inf E2 (ms)']:>8.2f} | {res['Consumo Medio']:>8.2f} | {res['Custo Medio']:>8.2f}")

    # ==========================================
    # GRÁFICO ESTILO PROFESSOR (Validação)
    # ==========================================
    plt.figure(figsize=(10, 6))
    cores = {"SAC": "blue", "PPO": "green", "DDPG": "red"}
    x_pos = np.arange(len(nomes_perfis))
    
    for algo in modelos_ai:
        df_algo = df_val[df_val['algoritmo'] == algo]
        medias_redes = df_algo.groupby(['perfil', 'train_seed'])['sla_u'].mean().reset_index()
        mu = [medias_redes[medias_redes['perfil'] == p]['sla_u'].mean() for p in nomes_perfis]
        std = [medias_redes[medias_redes['perfil'] == p]['sla_u'].std() for p in nomes_perfis]
        
        plt.plot(x_pos, mu, 'o-', color=cores[algo], label=f'{algo} (Validação)')
        plt.fill_between(x_pos, np.array(mu) - np.array(std), np.array(mu) + np.array(std), color=cores[algo], alpha=0.15)

    plt.axhline(y=resultados_finais[0]['SLA URLLC'], color='black', linestyle='--', linewidth=2, label=f"Heurística (Baseline: {resultados_finais[0]['SLA URLLC']:.1f}%)")
    plt.xticks(x_pos, nomes_perfis, rotation=15)
    plt.xlabel('Combinações de Pesos (Perfis de Recompensa)')
    plt.ylabel('SLA Médio de Validação (%) - URLLC')
    plt.title('Sensibilidade da Função de Recompensa (Multi-Pesos)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PASTA_DRIVE}/plots/sensibilidade_perfis_algoritmos.png", dpi=300)
    print("\n✅ Gráfico salvo em: plots/sensibilidade_perfis_algoritmos.png")
    plt.show()

if __name__ == "__main__":
    pipeline_definitivo_artigo()