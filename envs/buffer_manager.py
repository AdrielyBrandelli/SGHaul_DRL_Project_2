from collections import deque
from typing import List, Dict
from envs.wisun_emulator import Packet 

class UplinkBuffer5G:
    """
    Simula os buffers (filas de espera) na borda da rede (Edge/Gateway 5G).
    Gerencia o armazenamento temporário e calcula as violações de Packet Delay Budget (PDB).
    """
    def __init__(self, 
                 pdb_uc16: float = 300.0,      # Limite estrito de latência (SLA) para tráfego crítico (ms)
                 pdb_uc3: float = 2000.0,      # Limite tolerante de latência para tráfego massivo (ms)
                 pdb_mmtc: float = 5000.0,     # limite de tolerância do mmtc
                 max_buffer_bytes: int = 1024 * 512): # Buffer limitado a 512 KB por fila para simular HW real
                
        # Usamos deque (double-ended queue) para garantir performance O(1) no popleft (FIFO)
        self.queue_uc16 = deque()
        self.queue_uc3 = deque()
        self.queue_mmtc = deque()
        
        self.pdb_uc16 = pdb_uc16
        self.pdb_uc3 = pdb_uc3
        self.pdb_mmtc = pdb_mmtc
        self.max_buffer_bytes = max_buffer_bytes
        
        # Contadores em bytes para rápida checagem do estado (Observation State da IA)
        self.bytes_uc16 = 0
        self.bytes_uc3 = 0
        self.bytes_mmtc = 0

    def receive_packets(self, packets: List[Packet], current_time: float) -> int:
        """
        Ingresso de dados no Gateway. 
        O diferencial desta função é o 'carimbo de tempo', crucial para avaliar o SLA depois.
        """
        dropped_packets = 0
        for p in packets:
            # Separação estrita de fatias (Slicing Isolado) logo na entrada
            if p.tipo_trafego == 'UC16':
                if self.bytes_uc16 + p.tamanho_bytes <= self.max_buffer_bytes:
                    # CARIMBO DE TEMPO: Salva o pacote na fila junto com o instante exato em que ele chegou.
                    self.queue_uc16.append((p, current_time)) 
                    self.bytes_uc16 += p.tamanho_bytes
                else:
                    # Buffer overflow: a rede gerou mais tráfego do que o hardware suporta segurar
                    dropped_packets += 1
                    
            elif p.tipo_trafego == 'UC3':
                if self.bytes_uc3 + p.tamanho_bytes <= self.max_buffer_bytes:
                    # CARIMBO DE TEMPO: Essencial para não misturar pacotes velhos com novos
                    self.queue_uc3.append((p, current_time))
                    self.bytes_uc3 += p.tamanho_bytes
                else:
                    dropped_packets += 1

            elif p.tipo_trafego == 'mMTC':
                if self.bytes_mmtc + p.tamanho_bytes <= self.max_buffer_bytes:
                    self.queue_mmtc.append((p, current_time))
                    self.bytes_mmtc += p.tamanho_bytes
                else:
                    dropped_packets += 1

        return dropped_packets

    def serve_queues(self, capacity_uc16_bytes: float, capacity_uc3_bytes: float, capacity_mmtc_bytes: float, current_time: float) -> Dict[str, float]:
        """
        Processo de esvaziamento (transmissão) ditado pelas ações da Inteligência Artificial.
        'capacity_..._bytes' é o limite físico imposto pela IA neste exato milissegundo.
        """
        stats = {
            'throughput_uc3_bytes': 0.0,
            'throughput_uc16_bytes': 0.0,
            'throughput_mmtc_bytes': 0.0, 
            'pdb_violations_uc16': 0,
            'pdb_violations_uc3': 0,
            'pdb_violations_mmtc': 0,
            'avg_latency_uc16': 0.0,
            'avg_latency_uc3': 0.0,
            'avg_latency_mmtc': 0.0, 
            'avg_wisun_delay_uc16': 0.0, # NOVO
            'avg_wisun_delay_uc3': 0.0,  # NOVO
            'avg_wisun_delay_mmtc': 0.0, # NOVO
            'total_pacotes_uc16': 0, 
            'total_pacotes_uc3': 0,   
            'total_pacotes_mmtc': 0
        }
        
        # --- PROCESSAMENTO DO TRÁFEGO CRÍTICO (ALARMES) ---
        bytes_enviados_uc16 = 0
        latencias_uc16 = []

        # fila de pacotes para o URLLC (UC16)
        # Só transmite enquanto tiver pacote na fila E a IA tiver dado banda suficiente para passar
        while self.queue_uc16 and (bytes_enviados_uc16 + self.queue_uc16[0][0].tamanho_bytes) <= capacity_uc16_bytes:
            p, tempo_entrada_fila = self.queue_uc16.popleft()
            
            stats['total_pacotes_uc16'] += 1  
            self.bytes_uc16 -= p.tamanho_bytes
            bytes_enviados_uc16 += p.tamanho_bytes
            
            # CÁLCULO DE LATÊNCIA END-TO-END (Matemática alinhada com o Artigo)
            # 1. Quanto tempo o pacote ficou preso nesta fila aguardando a IA decidir?
            atraso_fila = (current_time - tempo_entrada_fila) * 1000.0 
            
            # 2. Latência Total = Atraso na malha rádio (Wi-SUN) + Atraso na fila do Gateway (5G)
            latencia_e2e = p.d_wisun + atraso_fila 
            latencias_uc16.append(latencia_e2e)
            
            # CHECAGEM DO SLA: Se a soma dos atrasos estourar 300ms, marca como violação.
            if latencia_e2e > self.pdb_uc16:
                stats['pdb_violations_uc16'] += 1

        # --- PROCESSAMENTO DO TRÁFEGO MASSIVO (MEDIDORES) ---
        # Mesma lógica do crítico, mas consumindo a banda alocada para o eMBB (UC3)
        bytes_enviados_uc3 = 0
        latencias_uc3 = []
        while self.queue_uc3 and (bytes_enviados_uc3 + self.queue_uc3[0][0].tamanho_bytes) <= capacity_uc3_bytes:
            p, tempo_entrada_fila = self.queue_uc3.popleft()
            
            stats['total_pacotes_uc3'] += 1  
            self.bytes_uc3 -= p.tamanho_bytes
            bytes_enviados_uc3 += p.tamanho_bytes
            
            atraso_fila = (current_time - tempo_entrada_fila) * 1000.0
            latencia_e2e = p.d_wisun + atraso_fila
            latencias_uc3.append(latencia_e2e)
            
            if latencia_e2e > self.pdb_uc3:
                stats['pdb_violations_uc3'] += 1

        # --- PROCESSAMENTO DO TRÁFEGO mMTC (UC1 e UC17) ---
        bytes_enviados_mmtc = 0
        latencias_mmtc = []
        while self.queue_mmtc and (bytes_enviados_mmtc + self.queue_mmtc[0][0].tamanho_bytes) <= capacity_mmtc_bytes:
            p, tempo_entrada_fila = self.queue_mmtc.popleft()
            
            stats['total_pacotes_mmtc'] += 1  
            self.bytes_mmtc -= p.tamanho_bytes
            bytes_enviados_mmtc += p.tamanho_bytes
            
            atraso_fila = (current_time - tempo_entrada_fila) * 1000.0
            latencia_e2e = p.d_wisun + atraso_fila
            latencias_mmtc.append(latencia_e2e)
            
            if latencia_e2e > self.pdb_mmtc:
                stats['pdb_violations_mmtc'] += 1
        
        # --- COMPILAÇÃO E RETORNO DOS DADOS PARA O AMBIENTE ---
        stats['throughput_uc16_bytes'] = bytes_enviados_uc16
        stats['throughput_uc3_bytes'] = bytes_enviados_uc3
        stats['throughput_mmtc_bytes'] = bytes_enviados_mmtc
        
        if latencias_uc16:
            stats['avg_latency_uc16'] = sum(latencias_uc16) / len(latencias_uc16)
        if latencias_uc3:
            stats['avg_latency_uc3'] = sum(latencias_uc3) / len(latencias_uc3)
        if latencias_mmtc:
            stats['avg_latency_mmtc'] = sum(latencias_mmtc) / len(latencias_mmtc)
            
        # NOVO: Calcula o atraso médio Wi-SUN de TODAS as fatias
        if self.queue_uc16:
            stats['avg_wisun_delay_uc16'] = sum(item[0].d_wisun for item in self.queue_uc16) / len(self.queue_uc16)
        if self.queue_uc3:
            stats['avg_wisun_delay_uc3'] = sum(item[0].d_wisun for item in self.queue_uc3) / len(self.queue_uc3)
        if self.queue_mmtc:
            stats['avg_wisun_delay_mmtc'] = sum(item[0].d_wisun for item in self.queue_mmtc) / len(self.queue_mmtc)
            
        return stats

    def get_queue_sizes(self) -> tuple:
        """Retorna a pressão atual nas filas para a IA 'enxergar' o estado da rede"""
        return self.bytes_uc16, self.bytes_uc3, self.bytes_mmtc