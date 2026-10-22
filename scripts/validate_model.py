import os
import sys
import time
import random
import csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from envs.sghaul_env import SGHaulEnv

def avaliar_estrategia(env, nome, semente_rodada, model=None, acao_estatica=None, episodios=1):
    historico_dados, historico_violacoes = [], []
    tempos_grafico, viol_inst_grafico = [], []
    dem_urllc_grafico, dem_embb_grafico = [], []
    aloc_urllc_grafico, aloc_embb_grafico = [], []
    fila_urllc_grafico, fila_embb_grafico = [], [] # ADICIONADO AQUI
    
    dados_debug_csv = [] 

    for ep in range(episodios):
        random.seed(semente_rodada)
        np.random.seed(semente_rodada)
        obs, info = env.reset(seed=semente_rodada)
            
        done = False
        violacoes_ep, dados_mb_ep, passos_ep = 0, 0.0, 0
        
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = acao_estatica

            obs, reward, terminated, truncated, info = env.step(action)
            
            viol_atual = info.get('violacoes_uc16', 0)
            violacoes_ep += viol_atual
            
            vazao_atual = info.get('throughput_mbps', 0.0)
            if vazao_atual > 0: dados_mb_ep += vazao_atual 
            
            # --- CAPTURA DIRETA DO AMBIENTE (Verdade Absoluta) ---
            dem_u = info.get('demanda_urllc_mbps', 0.0)
            dem_e = info.get('demanda_embb_mbps', 0.0)
            aloc_u = info.get('alocado_urllc_mbps', 0.0)
            aloc_e = info.get('alocado_embb_mbps', 0.0)
            fila_u = info.get('fila_urllc_kb', 0.0)
            fila_e = info.get('fila_embb_kb', 0.0)
            
            if ep == 0 and (dem_u > 0.0 or dem_e > 0.0):
                dados_debug_csv.append([
                    passos_ep, passos_ep * 0.01, 
                    round(dem_u, 6), round(dem_e, 6), 
                    round(aloc_u, 6), round(aloc_e, 6), viol_atual
                ])

            if ep == 0:
                tempos_grafico.append(passos_ep * 0.01)
                dem_urllc_grafico.append(dem_u)
                dem_embb_grafico.append(dem_e)
                aloc_urllc_grafico.append(aloc_u)
                aloc_embb_grafico.append(aloc_e)
                fila_urllc_grafico.append(fila_u) # ADICIONADO AQUI
                fila_embb_grafico.append(fila_e)  # ADICIONADO AQUI
                viol_inst_grafico.append(viol_atual)
                
            done = terminated or truncated
            passos_ep += 1
            
        historico_violacoes.append(violacoes_ep)
        historico_dados.append(dados_mb_ep)
        
    media_dados = np.mean(historico_dados)
    media_violacoes = np.mean(historico_violacoes)
    print(f"[{nome}] Vazão Total: {media_dados:.2f} Mb | Violações SLA: {media_violacoes:.0f}")
    
    nome_arquivo_csv = f"results/log_trafego_{nome.replace('/', '_').replace(' ', '_')}.csv"
    with open(nome_arquivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Passo', 'Tempo_s', 'Dem_URLLC_Mbps', 'Dem_eMBB_Mbps', 'Aloc_URLLC_Mbps', 'Aloc_eMBB_Mbps', 'Falhas_SLA'])
        writer.writerows(dados_debug_csv)
        
    # ATENÇÃO AOS RETORNOS (Muitas variáveis)
    return media_dados, media_violacoes, tempos_grafico, aloc_urllc_grafico, aloc_embb_grafico, dem_urllc_grafico, dem_embb_grafico, fila_urllc_grafico, fila_embb_grafico, viol_inst_grafico

def plotar_demanda_vs_acao(tempos, aloc_u, aloc_e, dem_u, dem_e, fila_u, fila_e, violacoes, titulo, nome_arquivo):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    limite = len(tempos)
    t_plot = tempos[:limite]
    
    # === PAINEL 1: URLLC ===
    ax1.plot(t_plot, dem_u[:limite], color='#E74C3C', label='Demanda Real (Mbps)', linewidth=2, linestyle='--')
    ax1.plot(t_plot, aloc_u[:limite], color='#8E44AD', label='Espaço Alocado (Mbps)', linewidth=2.5, drawstyle='steps-post', alpha=0.8)
    ax1.set_ylabel('Banda (Mbps)', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Eixo Secundário da Fila URLLC (Sombra)
    # 2. Desenha a FILA DE ESPERA como sombra preta visível (Eixo Direito - KB)
    ax1_twin = ax1.twinx()
    # Pinta o fundo de preto com mais opacidade
    ax1_twin.fill_between(t_plot, 0, fila_u[:limite], color='black', alpha=0.5, step='post')
    # Contorna com uma linha preta grossa para o "fio de cabelo" não sumir!
    ax1_twin.plot(t_plot, fila_u[:limite], color='black', linewidth=2.0, drawstyle='steps-post', label='Buffer (Fila KB)')
    
    ax1_twin.set_ylabel('Fila de Espera (KB)', fontweight='bold', color='black')
    ax1_twin.tick_params(axis='y', labelcolor='black')
    ax1_twin.set_ylim(0, max(fila_u[:limite]) * 1.5 if max(fila_u[:limite]) > 0 else 10)

    t_viols = [t_plot[i] for i, v in enumerate(violacoes[:limite]) if v > 0]
    y_viols = [dem_u[i] for i, v in enumerate(violacoes[:limite]) if v > 0]
    if t_viols:
        ax1.scatter(t_viols, y_viols, color='red', s=80, zorder=5, label='Falha SLA (Estouro de Tempo)')

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc='upper right', fontsize=9)
    ax1.set_title("Slice URLLC: Banda Alocada vs Acúmulo no Buffer", fontsize=11, fontweight='bold')

    # === PAINEL 2: eMBB ===
    ax2.plot(t_plot, dem_e[:limite], color='#2980B9', label='Demanda Real (Mbps)', linewidth=2, linestyle='--')
    ax2.plot(t_plot, aloc_e[:limite], color='#F39C12', label='Espaço Alocado (Mbps)', linewidth=2.5, drawstyle='steps-post', alpha=0.8)
    ax2.set_ylabel('Banda (Mbps)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Eixo Secundário da Fila eMBB (Sombra)
    # 2. Desenha a FILA DE ESPERA como sombra preta visível (Eixo Direito - KB)
    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(t_plot, 0, fila_e[:limite], color='black', alpha=0.5, step='post')
    ax2_twin.plot(t_plot, fila_e[:limite], color='black', linewidth=2.0, drawstyle='steps-post', label='Buffer (Fila KB)')
    
    ax2_twin.set_ylabel('Fila de Espera (KB)', fontweight='bold', color='black')
    ax2_twin.tick_params(axis='y', labelcolor='black')
    ax2_twin.set_ylim(0, max(fila_e[:limite]) * 1.5 if max(fila_e[:limite]) > 0 else 10)
    
    l3, lab3 = ax2.get_legend_handles_labels()
    l4, lab4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(l3 + l4, lab3 + lab4, loc='upper right', fontsize=9)
    ax2.set_title("Slice eMBB: Banda Alocada vs Acúmulo no Buffer", fontsize=11, fontweight='bold')
    ax2.set_xlabel('Tempo de Simulação (s)', fontweight='bold')
    
    # Zoom no gráfico (Mostra apenas os primeiros 25 segundos)
    ax1.set_xlim(0, 25) 
    
    plt.suptitle(titulo, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    plt.close()

def plotar_barras_resumo(nomes, vazoes, violacoes):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(nomes))
    width = 0.35

    ax1.bar(x - width/2, vazoes, width, label='Vazão Total (Mb)', color='royalblue', edgecolor='black')
    ax1.set_ylabel('Total de Dados Entregues (Mb)', color='royalblue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, violacoes, width, label='Violações SLA Totais', color='crimson', edgecolor='black')
    ax2.set_ylabel('Total de Violações PDB', color='crimson', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='crimson')

    ax1.set_xticks(x)
    ax1.set_xticklabels(nomes)
    plt.title("Resumo Geral: Trade-off Vazão vs Confiabilidade")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig("results/grafico_0_resumo_barras.png", dpi=300)
    plt.close()


def main():
    print("=== AVALIAÇÃO DE ENGENHARIA (O-RAN SLICING) ===\n")
    #semente_rodada = 83604
    semente_rodada = random.randint(1, 100000)
    print(f"Semente desta rodada: {semente_rodada}")
    env = SGHaulEnv()
    
    caminho = "models/best_model/best_model.zip"
    if not os.path.exists(caminho): caminho = "models/ppo_sghaul_slicing.zip"
        
    try: model = PPO.load(caminho, env=env)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return
    
    os.makedirs("results", exist_ok=True)
    nomes = ['Agente PPO\n(IA)', 'Baseline\n(50/50)', 'Baseline\n(100/0)']
    vazoes_totais, violacoes_totais = [], []
    
    # 1. AVALIA O AGENTE IA (Faltou isso rodar no seu terminal!)
    print("Testando Agente IA...")
    vaz, vio, t_ia, aloc_u_ia, aloc_e_ia, dem_u_ia, dem_e_ia, fila_u_ia, fila_e_ia, v_inst_ia = avaliar_estrategia(env, "IA O-RAN", semente_rodada, model=model)
    vazoes_totais.append(vaz)
    violacoes_totais.append(vio)
    plotar_demanda_vs_acao(t_ia, aloc_u_ia, aloc_e_ia, dem_u_ia, dem_e_ia, fila_u_ia, fila_e_ia, v_inst_ia, "Desempenho O-RAN: Agente DRL", "results/grafico_1_demanda_acao_IA.png")
    
    # 2. AVALIA O BASELINE 50/50 (Ação = 0.0)
    print("Testando Baseline 50/50...")
    vaz, vio, t_b50, aloc_u_b50, aloc_e_b50, dem_u_b50, dem_e_b50, fila_u_b50, fila_e_b50, v_inst_b50 = avaliar_estrategia(env, "Baseline 50/50", semente_rodada, acao_estatica=np.array([0.0]))
    vazoes_totais.append(vaz)
    violacoes_totais.append(vio)
    plotar_demanda_vs_acao(t_b50, aloc_u_b50, aloc_e_b50, dem_u_b50, dem_e_b50, fila_u_b50, fila_e_b50, v_inst_b50, "Desempenho Estático: Baseline 50/50", "results/grafico_2_demanda_acao_Base50.png")
    
    # 3. AVALIA O BASELINE 100/0 (Ação = 1.0)
    print("Testando Baseline 100/0...")
    vaz, vio, t_b100, aloc_u_b100, aloc_e_b100, dem_u_b100, dem_e_b100, fila_u_b100, fila_e_b100, v_inst_b100 = avaliar_estrategia(env, "Baseline 100/0", semente_rodada, acao_estatica=np.array([1.0]))
    vazoes_totais.append(vaz)
    violacoes_totais.append(vio)
    plotar_demanda_vs_acao(t_b100, aloc_u_b100, aloc_e_b100, dem_u_b100, dem_e_b100, fila_u_b100, fila_e_b100, v_inst_b100, "Desempenho URLLC: Baseline 100/0", "results/grafico_3_demanda_acao_Base100.png")

    plotar_barras_resumo(nomes, vazoes_totais, violacoes_totais)
    print("\n✅ Gráficos gerados com sucesso na pasta 'results/'!")

if __name__ == "__main__":
    main()