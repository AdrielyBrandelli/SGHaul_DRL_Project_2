import os
import sys
import numpy as np

# Garante que o Python encontre as nossas pastas executando de qualquer lugar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO, SAC, DDPG 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from envs.sghaul_env import SGHaulEnv

#def run_static_baseline(env, rho_c_static=0.5, rho_m_static=0.5, episodes=5):
def run_static_baseline(env, action_logits=[0.0, 0.0, 0.0], nome_baseline="Equilibrado 33%"): #Troca para 3 slices dividos iguais
    """
    Roda um baseline de alocação estática (ex: 50% / 50%).
    Serve como a nossa 'linha de base humana' para provarmos matematicamente o quanto a Inteligência Artificial supera uma configuração de rede engessada.
    
    Como usamos Softmax no ambiente, enviar [0, 0, 0] resulta em 33% para cada fatia.
    Enviar [2.0, 0.0, 0.0] dá ampla prioridade à primeira fatia (URLLC).
    """
    total_rewards = []
    
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            # Ação fixa engessada (sem inteligência)
            action = np.array(action_logits, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
        total_rewards.append(ep_reward)
        
    print(f"[Baseline Estático {nome_baseline}] Recompensa Média: {np.mean(total_rewards):.2f}")


if __name__ == "__main__":
    print("--- INICIANDO TREINAMENTO SGHaul DRL ---\n")
    
    # 1. Instanciando e Validando a Engenharia do Ambiente
    '''env = SGHaulEnv()
    
    print("Verificando conformidade do ambiente...")
    check_env(env, warn=True)
    print("Ambiente aprovado!\n")

    # 2. Configurando o Algoritmo (Cérebro da IA) e os Logs
    log_dir = "logs/ppo_sghaul_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models/best_model", exist_ok=True)

    # --- SINTONIA FINA DO ALGORITMO PPO ---
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=0.001,      # Acelerado para convergência mais rápida
        ent_coef=0.1,             # Alto coeficiente de entropia: força o agente a explorar decisões agressivas em vez de ficar na zona de conforto
        clip_range=0.2,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/"
    )

    # 3. Configurando a Eficiência de Recursos (Early Stopping)
    eval_env = SGHaulEnv() 
    
    # Gatilho Inteligente: Aborta o treinamento se o modelo estagnar 
    # por 5 avaliações seguidas. Isso poupa CPU/GPU e tempo de projeto.
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, 
        min_evals=5, 
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10000,
        callback_after_eval=stop_train_callback,
        best_model_save_path='./models/best_model/',
        verbose=1
    )

    # 4. Treinamento Principal
    timesteps_treinamento = 500000 
    print(f"Treinando o agente por até {timesteps_treinamento} timesteps...")
    
    model.learn(total_timesteps=timesteps_treinamento, callback=eval_callback)

    model_path = "models/ppo_sghaul_slicing"
    model.save(model_path)
    print(f"\nModelo final salvo em '{model_path}.zip'!\n")

    print("--- AVALIAÇÃO DE DESEMPENHO ---")
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"[Agente PPO Treinado] Recompensa Média: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    run_static_baseline(env, rho_c_static=0.0) # Equivalente a 50/50 no nosso mapeamento
    run_static_baseline(env, rho_c_static=1.0) # Equivalente a prioridade 100% pro URLLC
    '''
    env_teste = SGHaulEnv()
    print("Verificando conformidade do ambiente...")
    check_env(env_teste, warn=True)
    print("Ambiente aprovado!\n")

    # Dicionário separando as características exclusivas de cada algoritmo
    algoritmos = {
        "PPO": {
            "classe": PPO,
            "kwargs": {
                "policy": "MlpPolicy",
                "learning_rate": 0.001,      
                "ent_coef": 0.1,             
                "clip_range": 0.2,
                "gamma": 0.99,
                "verbose": 0
            }
        },
        "SAC": {
            "classe": SAC,
            "kwargs": {
                "policy": "MlpPolicy",
                "learning_rate": 0.001,
                "gamma": 0.99,
                "verbose": 0
            }
        },
        "DDPG": {
            "classe": DDPG,
            "kwargs": {
                "policy": "MlpPolicy",
                "learning_rate": 0.001,
                "gamma": 0.99,
                "verbose": 0
            }
        }
    }

    timesteps_treinamento = 500000 

    # Loop de Automação do Treinamento
    for nome_algo, config in algoritmos.items():
        print(f"\n{'='*50}")
        print(f" INICIANDO TREINAMENTO: {nome_algo}")
        print(f"{'='*50}")

        env_treino = SGHaulEnv()
        eval_env = SGHaulEnv()
        
        log_dir = f"logs/{nome_algo.lower()}_sghaul_tensorboard/"
        best_model_dir = f"./models/best_model_{nome_algo.lower()}/"
        model_path = f"models/{nome_algo.lower()}_sghaul_slicing_3fatias"
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)

        AlgoClass = config["classe"]
        kwargs = config["kwargs"]

        # Instancia o modelo dinamicamente
        model = AlgoClass(env=env_treino, tensorboard_log=log_dir, **kwargs)

        stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5, 
            min_evals=5, 
            verbose=1
        )

        eval_callback = EvalCallback(
            eval_env,
            eval_freq=10000,
            callback_after_eval=stop_train_callback,
            best_model_save_path=best_model_dir,
            verbose=1
        )
        
        print(f"Treinando o agente {nome_algo}...")
        model.learn(total_timesteps=timesteps_treinamento, callback=eval_callback)
        model.save(model_path)
        print(f"Modelo salvo em '{model_path}.zip'!")

        print(f"\n--- AVALIAÇÃO DE DESEMPENHO ({nome_algo}) ---")
        mean_reward, std_reward = evaluate_policy(model, env_treino, n_eval_episodes=10)
        print(f"[Agente {nome_algo}] Recompensa Média: {mean_reward:.2f} +/- {std_reward:.2f}\n")

    print("\n" + "="*50)
    print(" TESTANDO BASELINES ESTÁTICOS")
    print("="*50)
    env_baseline = SGHaulEnv()
    run_static_baseline(env_baseline, action_logits=[0.0, 0.0, 0.0], nome_baseline="Divisão Igual (33% para cada)")
    run_static_baseline(env_baseline, action_logits=[2.0, 0.0, 0.0], nome_baseline="Prioridade URLLC (Aprox. 70% / 15% / 15%)")
    run_static_baseline(env_baseline, action_logits=[0.0, 2.0, 0.0], nome_baseline="Prioridade eMBB (Aprox. 15% / 70% / 15%)")