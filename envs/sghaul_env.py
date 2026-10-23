import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random  

from envs.wisun_emulator import WiSUN_Emulator
from envs.buffer_manager import UplinkBuffer5G

class SGHaulEnv(gym.Env):
    """
    Ambiente Customizado de Network Slicing (Fatiamento de Rede).
    Aqui a Inteligência Artificial (Agente DRL) atua como o 'Cérebro' do Gateway 5G,
    decidindo em tempo real como fatiar a banda sob condições extremas.
    """
    def __init__(self, w1=150.0, w2=200.0, w3=50.0, w4=50.0): 
        super(SGHaulEnv, self).__init__()
        
        # Banda base de 1 MHz para forçar o estrangulamento da rede (cenário de crise)
        self.bandwidth_hz = 1 * 1e6 
        self.step_duration = 0.01 
        self.max_steps = 10000
        
        self.topologia_atual = "Linear"
        self.wisun = WiSUN_Emulator(topology_name=self.topologia_atual)
        self.buffer = UplinkBuffer5G()
        
        # Ação da IA: Controle percentual (Softmax)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # --- Limites Parametrizados (Fácil ajuste para justificar aos revisores) ---
        self.max_wisun_delay_ms = 250.0

        # Observação da IA (10 Dimensões): 
        # [Lat 5G URLLC, Lat 5G eMBB, Lat 5G mMTC, 
        #  Fila URLLC, Fila eMBB, Fila mMTC, 
        #  Atraso Malha Wi-SUN URLLC, Atraso Malha Wi-SUN eMBB, Atraso Malha Wi-SUN mMTC, 
        #  SINR]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )
        
        self.current_time = 0.0
        self.current_step = 0
        
        # Pesos da Recompensa
        self.w1 = w1 
        self.w2 = w2 
        self.w3 = w3 
        self.w4 = w4

    def reset(self, seed=None, options=None):
        """Reinicia o ambiente no começo de cada 'partida' (Episódio)"""
        super().reset(seed=seed)
        
        topologias = ["Linear", "Mesh", "Quadrada"]
        self.topologia_atual = random.choice(topologias)
        
        self.wisun = WiSUN_Emulator(topology_name=self.topologia_atual)
        self.buffer = UplinkBuffer5G()
        self.current_time = 0.0
        self.current_step = 0
        
        sinr_inicial = self._gerar_sinr()
        sinr_norm = min(sinr_inicial / 10.0, 1.0) 
        
        # Inicializando com 10 estados zerados (9 zeros + sinr)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sinr_norm], dtype=np.float32)
        
        return self.state, {}

    def _gerar_sinr(self) -> float:
        """Simula a oscilação natural da qualidade do sinal de rádio 5G (Log-Normal Shadowing)"""
        return max(0.1, np.random.lognormal(mean=2.0, sigma=0.5))

    def step(self, action):
        self.current_step += 1

        # 1. Alocação de banda pela IA
        exp_action = np.exp(action)
        band_allocation = exp_action / np.sum(exp_action)
        rho_c = band_allocation[0]
        rho_m = band_allocation[1]
        rho_mmtc = band_allocation[2]

        # 2. Capacidade da Rede (Fator Mágico Removido)
        # O gargalo agora ocorre naturalmente pois a largura de banda física fixa 
        # precisa lidar com payloads pesados de 400B da norma.
        sinr = self._gerar_sinr()
        capacidade_total_bytes = (self.bandwidth_hz * np.log2(1 + sinr) * self.step_duration) / 8.0
        
        capacidade_c = capacidade_total_bytes * rho_c
        capacidade_m = capacidade_total_bytes * rho_m
        capacidade_mmtc = capacidade_total_bytes * rho_mmtc
        
        # 3. Recepção de Tráfego e Viés de Sobrevivência
        pacotes_chegando = self.wisun.step(self.current_time, self.step_duration)
        
        # O buffer agora retorna um dicionário informando exatamente de qual fatia o pacote caiu
        drops_dict = self.buffer.receive_packets(pacotes_chegando, self.current_time)
        
        # O processamento transmite os pacotes (com Work-Conserving interno)
        stats = self.buffer.serve_queues(capacidade_c, capacidade_m, capacidade_mmtc, self.current_time)

        # CORREÇÃO DO VIÉS DE SOBREVIVÊNCIA: Pacotes descartados entram como Violação de SLA
        stats['pdb_violations_uc16'] += drops_dict['UC16']
        stats['pdb_violations_uc3'] += drops_dict['UC3']
        stats['pdb_violations_mmtc'] += drops_dict['mMTC']
        
        stats['total_pacotes_uc16'] += drops_dict['UC16']
        stats['total_pacotes_uc3'] += drops_dict['UC3']
        stats['total_pacotes_mmtc'] += drops_dict['mMTC']
        
        pacotes_descartados = sum(drops_dict.values())

        # 4. Cálculo de Métricas (Throughput e Latência)
        throughput_urllc_mbps = (stats['throughput_uc16_bytes'] * 8) / (self.step_duration * 1_000_000)
        throughput_embb_mbps = (stats['throughput_uc3_bytes'] * 8) / (self.step_duration * 1_000_000)
        throughput_mmtc_mbps = (stats['throughput_mmtc_bytes'] * 8) / (self.step_duration * 1_000_000)
        throughput_m_mbps = throughput_embb_mbps 
        throughput_c_mbps = throughput_urllc_mbps
        throughput_total_mbps = throughput_urllc_mbps + throughput_embb_mbps + throughput_mmtc_mbps
        
        bytes_uc16, bytes_uc3, bytes_mmtc = self.buffer.get_queue_sizes()
        avg_lat_c = stats.get('avg_latency_uc16', 0.0)
        avg_lat_mmtc = stats.get('avg_latency_mmtc', 0.0)
        avg_lat_embb = stats.get('avg_latency_uc3', 0.0)

        avg_wisun_delay_c = stats.get('avg_wisun_delay_uc16', 0.0)
        avg_wisun_delay_embb = stats.get('avg_wisun_delay_uc3', 0.0)
        avg_wisun_delay_mmtc = stats.get('avg_wisun_delay_mmtc', 0.0)
        
        e2e_latency_urllc = avg_lat_c + avg_wisun_delay_c

        # 5. Padronização Matemática do Buffer (512 KB)
        # Substitui o "10240.0" antigo para alinhar com a memória do roteador industrial
        teto_fila = 512.0 * 1024.0 
        
        penalidade_lat_urllc = (avg_lat_c / 300.0) * (self.w2 * 0.1) 
        penalidade_lat_mmtc = (avg_lat_mmtc / 5000.0) * (self.w4 * 0.1)

        penalidade_acumulo_urllc = min(bytes_uc16 / teto_fila, 1.0) * (self.w2 * 2.0)
        penalidade_acumulo_mmtc = min(bytes_mmtc / teto_fila, 1.0) * (self.w4 * 2.0)

        # 6. Função de Recompensa
        reward = (self.w1 * throughput_m_mbps) \
                 - (self.w2 * stats['pdb_violations_uc16']) \
                 - (self.w4 * stats['pdb_violations_mmtc']) \
                 - (self.w3 * pacotes_descartados) \
                 - penalidade_lat_urllc \
                 - penalidade_acumulo_urllc \
                 - penalidade_lat_mmtc \
                 - penalidade_acumulo_mmtc
                 
        reward = reward / 100.0  
                 
        self.current_time += self.step_duration
        
        # 7. Normalização do Estado (State Space de 10 Dimensões)
        lat_norm_c = min(avg_lat_c / 300.0, 1.0) 
        lat_norm_embb = min(avg_lat_embb / 1000.0, 1.0) 
        lat_norm_mmtc = min(avg_lat_mmtc / 5000.0, 1.0)

        bytes_uc16_norm = min(bytes_uc16 / teto_fila, 1.0)
        bytes_uc3_norm = min(bytes_uc3 / teto_fila, 1.0)
        bytes_mmtc_norm = min(bytes_mmtc / teto_fila, 1.0)
        
        wisun_delay_norm_c = min(avg_wisun_delay_c / self.max_wisun_delay_ms, 1.0)
        wisun_delay_norm_embb = min(avg_wisun_delay_embb / self.max_wisun_delay_ms, 1.0)
        wisun_delay_norm_mmtc = min(avg_wisun_delay_mmtc / self.max_wisun_delay_ms, 1.0)
        
        sinr_norm = min(sinr / 10.0, 1.0) 
        
        self.state = np.array([
            lat_norm_c, lat_norm_embb, lat_norm_mmtc, 
            bytes_uc16_norm, bytes_uc3_norm, bytes_mmtc_norm, 
            wisun_delay_norm_c, wisun_delay_norm_embb, wisun_delay_norm_mmtc, 
            sinr_norm
        ], dtype=np.float32)
        
        terminated = False
        truncated = bool(self.current_step >= self.max_steps)

        # 8. Extração de Informações para Relatórios
        demanda_urllc_mbps = throughput_c_mbps + ((float(bytes_uc16) * 8.0) / (1e6 * float(self.step_duration)))
        demanda_embb_mbps = throughput_m_mbps + ((float(bytes_uc3) * 8.0) / (1e6 * float(self.step_duration)))
        demanda_mmtc_mbps = throughput_mmtc_mbps + ((float(bytes_mmtc) * 8.0) / (1e6 * float(self.step_duration)))
        alocado_urllc_mbps = (float(capacidade_c) * 8.0) / (1e6 * float(self.step_duration))
        alocado_embb_mbps = (float(capacidade_m) * 8.0) / (1e6 * float(self.step_duration))
        alocado_mmtc_mbps = (float(capacidade_mmtc) * 8.0) / (1e6 * float(self.step_duration))
        
        info = {
            'throughput_total_mbps': throughput_total_mbps,
            'throughput_urllc_mbps': throughput_urllc_mbps, 
            'throughput_embb_mbps': throughput_embb_mbps,
            'throughput_mmtc_mbps': throughput_mmtc_mbps,

            'demanda_urllc_mbps': demanda_urllc_mbps, 
            'demanda_embb_mbps': demanda_embb_mbps, 
            'demanda_mmtc_mbps': demanda_mmtc_mbps,

            'alocado_urllc_mbps': alocado_urllc_mbps, 
            'alocado_embb_mbps': alocado_embb_mbps,    
            'alocado_mmtc_mbps': alocado_mmtc_mbps,    

            'fila_urllc_kb': bytes_uc16 / 1024.0, 
            'fila_embb_kb': bytes_uc3 / 1024.0, 
            'fila_mmtc_kb': bytes_mmtc / 1024.0,

            'drops': pacotes_descartados,
            'topologia': self.topologia_atual,

            'violacoes_uc16': stats['pdb_violations_uc16'],
            'total_pacotes_uc16': stats['total_pacotes_uc16'],
            'avg_latency_uc16': avg_lat_c,
            
            'avg_wisun_delay_uc16': avg_wisun_delay_c,
            'avg_wisun_delay_uc3': avg_wisun_delay_embb,
            'avg_wisun_delay_mmtc': avg_wisun_delay_mmtc,
            'e2e_latency_uc16': e2e_latency_urllc,

            'violacoes_uc3': stats['pdb_violations_uc3'],
            'total_pacotes_uc3': stats['total_pacotes_uc3'],
            'avg_latency_uc3': avg_lat_embb,

            'violacoes_mmtc': stats['pdb_violations_mmtc'],
            'total_pacotes_mmtc': stats['total_pacotes_mmtc'],
            'avg_latency_mmtc': avg_lat_mmtc
        }

        return self.state, reward, terminated, truncated, info