import numpy as np
import json
import os
from dataclasses import dataclass
from typing import List

@dataclass
class Packet:
    """
    Estrutura do pacote DLMS virtual gerado na malha Wi-SUN.
    Guarda o histórico do pacote desde o medidor até chegar no gateway 5G.
    """
    id_pacote: int
    tipo_trafego: str
    tamanho_bytes: int
    hop_count: int
    d_wisun: float
    timestamp_chegada: float 

class WiSUN_Emulator:
    def __init__(self,
                 topology_name: str = "Linear",
                 lambda_uc16: float = 10.0,
                 lambda_mmtc: float = 50.0,
                 burst_uc3: int = 500,
                 periodo_uc3: float = 5.0,
                 max_hops: int = 10):
        self.topology_name = topology_name
        self.lambda_uc16 = lambda_uc16
        self.lambda_mmtc = lambda_mmtc
        self.burst_uc3 = burst_uc3
        self.periodo_uc3 = periodo_uc3
        self.max_hops = max_hops
        
        # Valores de fallback de segurança
        self.alpha_uc16 = 20.0
        self.alpha_uc3 = 35.0
        self.beta = 10.0
        self.std_dev_jitter = 5.0
        self.pdr_slope = -0.015
        self.pdr_intercept = 1.0
        
        self.packet_counter = 0
        self.last_uc3_burst_time = 0.0
        
        # NOVOS Fallbacks definidos ANTES de carregar a calibração
        self.tamanho_uc16 = 100
        self.tamanho_uc3 = 400
        self.tamanho_mmtc = 150
        
        # Agora sim injeta as variáveis físicas reais, sem gerar erro
        self._carregar_calibracao()

    def _carregar_calibracao(self):
        """Lê os parâmetros físicos da topologia e aplica o fator de estresse."""
        caminho_json = os.path.join("data", "calibracao.json")
        if os.path.exists(caminho_json):
            try:
                with open(caminho_json, "r") as f:
                    dados_completos = json.load(f)
                
                if self.topology_name in dados_completos:
                    dados = dados_completos[self.topology_name]
                    self.alpha_uc16 = dados.get('alpha_uc16', self.alpha_uc16)
                    self.alpha_uc3 = dados.get('alpha_uc3', self.alpha_uc3)
                    self.beta = dados.get('beta', self.beta)
                    self.std_dev_jitter = dados.get('std_dev_jitter', self.std_dev_jitter)
                    self.pdr_slope = dados.get('pdr_slope', self.pdr_slope)
                    self.pdr_intercept = dados.get('pdr_intercept', self.pdr_intercept)

                    # --- GARGALO INTENCIONAL ---
                    # Multiplicamos os alarmes (150x) e medidores (20x) para simular 
                    # um cenário de crise aguda na rede (ex: apagão geral). 
                    # É isso que justifica o uso de Inteligência Artificial depois.
                    fator_estresse_alarmes = 150.0 
                    fator_estresse_medidores = 20.0 
                    
                    self.lambda_uc16 = dados.get('lambda_uc16', self.lambda_uc16) * fator_estresse_alarmes
                    self.lambda_mmtc = dados.get('lambda_mmtc', self.lambda_mmtc)
                    self.burst_uc3 = int(dados.get('burst_uc3', self.burst_uc3) * fator_estresse_medidores)
                    self.periodo_uc3 = dados.get('periodo_uc3', self.periodo_uc3)

                    self.periodo_uc3 = dados.get('periodo_uc3', self.periodo_uc3)
                    
                    # CARREGAR TAMANHOS DE PACOTE DO JSON
                    self.tamanho_uc16 = dados.get('tamanho_uc16', self.tamanho_uc16)
                    self.tamanho_uc3 = dados.get('tamanho_uc3', self.tamanho_uc3)
                    self.tamanho_mmtc = dados.get('tamanho_mmtc', self.tamanho_mmtc)
                    
            except Exception as e:
                #print(f"Aviso WiSUN_Emulator: Falha ao ler JSON ({e}).")
                pass

    def _calcular_atraso(self, hop_count: int, tipo_trafego: str) -> float:
        """
        Calcula o atraso do pacote dentro da malha Wi-SUN antes de chegar no 5G.
        Matemática: Atraso = (Tempo por Salto * Qtde Saltos) + Atraso de Proc. + Jitter
        """
        alpha = self.alpha_uc16 if tipo_trafego == 'UC16' else self.alpha_uc3
        epsilon = np.random.normal(loc=0.0, scale=self.std_dev_jitter)
        d_wisun = (alpha * hop_count) + self.beta + epsilon
        
        return max(1.0, d_wisun)


    def _simular_pdr(self, hop_count: int) -> bool:
        """
        Simula a perda de pacotes natural do rádio (Packet Delivery Ratio).
        Quanto mais saltos (hops) o pacote dá, maior a chance de se perder pelo caminho.
        """
        prob_sucesso = (self.pdr_slope * hop_count) + self.pdr_intercept
        prob_sucesso = max(0.0, min(1.0, prob_sucesso))
        return np.random.rand() <= prob_sucesso

    def step(self, current_time: float, step_duration: float) -> List[Packet]:
        """Avança a simulação no tempo e gera o tráfego concorrente."""
        pacotes_gerados = []
        
        # 1. Tráfego Crítico (UC16 - Alarmes)
        # Usa distribuição de Poisson porque alarmes são imprevisíveis
        qtd_uc16 = np.random.poisson(self.lambda_uc16 * step_duration)
        for _ in range(qtd_uc16):
            hop_count = np.random.randint(1, self.max_hops + 1)
            if self._simular_pdr(hop_count): # Só enfileira se o rádio não perdeu o pacote
                self.packet_counter += 1
                pacotes_gerados.append(Packet(
                    id_pacote=self.packet_counter,
                    tipo_trafego='UC16',
                    tamanho_bytes=self.tamanho_uc16,
                    #tamanho_bytes=64,
                    hop_count=hop_count,
                    d_wisun=self._calcular_atraso(hop_count, 'UC16'),
                    timestamp_chegada=current_time + np.random.uniform(0, step_duration)
                ))
                
        # 2. Tráfego Massivo (UC3 - Medidores)
        # Usa rajadas periódicas programadas (comportamento real de faturamento)
        if (current_time - self.last_uc3_burst_time) >= self.periodo_uc3:
            self.last_uc3_burst_time = current_time
            for _ in range(self.burst_uc3):
                hop_count = np.random.randint(1, self.max_hops + 1)
                if self._simular_pdr(hop_count):
                    self.packet_counter += 1
                    pacotes_gerados.append(Packet(
                        id_pacote=self.packet_counter,
                        tipo_trafego='UC3',
                        tamanho_bytes=self.tamanho_uc3,
                        #tamanho_bytes=64,
                        hop_count=hop_count,
                        d_wisun=self._calcular_atraso(hop_count, 'UC3'),
                        timestamp_chegada=current_time + np.random.uniform(0, step_duration)
                    ))

        # 3. Tráfego de Manutenção (mMTC)
        qtd_mmtc = np.random.poisson(self.lambda_mmtc * step_duration)
        for _ in range(qtd_mmtc):
            hop_count = np.random.randint(1, self.max_hops + 1)
            if self._simular_pdr(hop_count):
                self.packet_counter += 1
                pacotes_gerados.append(Packet(
                    id_pacote=self.packet_counter,
                    tipo_trafego='mMTC',
                    tamanho_bytes=self.tamanho_mmtc,
                    #tamanho_bytes=64, 
                    hop_count=hop_count,
                    d_wisun=self._calcular_atraso(hop_count, 'mMTC'),
                    timestamp_chegada=current_time + np.random.uniform(0, step_duration)
                ))            
        # Ordena cronologicamente para garantir precisão no timestamp de entrada na fila 5G
        return sorted(pacotes_gerados, key=lambda p: p.timestamp_chegada)