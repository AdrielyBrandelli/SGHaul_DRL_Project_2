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

    def receive_packets(self, packets: List[Packet], current_time: float) -> Dict[str, int]:
        """Ingresso de dados no Gateway, rastreando pacotes descartados por fatia."""
        drops = {'UC16': 0, 'UC3': 0, 'mMTC': 0}
        
        for p in packets:
            if p.tipo_trafego == 'UC16':
                if self.bytes_uc16 + p.tamanho_bytes <= self.max_buffer_bytes:
                    self.queue_uc16.append((p, current_time)) 
                    self.bytes_uc16 += p.tamanho_bytes
                else:
                    drops['UC16'] += 1
                    
            elif p.tipo_trafego == 'UC3':
                if self.bytes_uc3 + p.tamanho_bytes <= self.max_buffer_bytes:
                    self.queue_uc3.append((p, current_time))
                    self.bytes_uc3 += p.tamanho_bytes
                else:
                    drops['UC3'] += 1

            elif p.tipo_trafego == 'mMTC':
                if self.bytes_mmtc + p.tamanho_bytes <= self.max_buffer_bytes:
                    self.queue_mmtc.append((p, current_time))
                    self.bytes_mmtc += p.tamanho_bytes
                else:
                    drops['mMTC'] += 1

        return drops

    def serve_queues(self, capacity_uc16_bytes: float, capacity_uc3_bytes: float, capacity_mmtc_bytes: float, current_time: float) -> Dict[str, float]:
        stats = {
            'throughput_uc3_bytes': 0.0, 'throughput_uc16_bytes': 0.0, 'throughput_mmtc_bytes': 0.0, 
            'pdb_violations_uc16': 0, 'pdb_violations_uc3': 0, 'pdb_violations_mmtc': 0,
            'avg_latency_uc16': 0.0, 'avg_latency_uc3': 0.0, 'avg_latency_mmtc': 0.0, 
            'avg_wisun_delay_uc16': 0.0, 'avg_wisun_delay_uc3': 0.0, 'avg_wisun_delay_mmtc': 0.0,
            'total_pacotes_uc16': 0, 'total_pacotes_uc3': 0, 'total_pacotes_mmtc': 0
        }
        
        # --- PROCESSAMENTO URLLC ---
        bytes_enviados_uc16 = 0
        latencias_uc16 = []
        while self.queue_uc16 and (bytes_enviados_uc16 + self.queue_uc16[0][0].tamanho_bytes) <= capacity_uc16_bytes:
            p, tempo_entrada_fila = self.queue_uc16.popleft()
            stats['total_pacotes_uc16'] += 1  
            self.bytes_uc16 -= p.tamanho_bytes
            bytes_enviados_uc16 += p.tamanho_bytes
            
            atraso_fila = (current_time - tempo_entrada_fila) * 1000.0 
            latencia_e2e = p.d_wisun + atraso_fila 
            latencias_uc16.append(latencia_e2e)
            
            if latencia_e2e > self.pdb_uc16:
                stats['pdb_violations_uc16'] += 1

        # ALOCAÇÃO WORK-CONSERVING: Redistribui banda ociosa do URLLC
        sobra_uc16 = capacity_uc16_bytes - bytes_enviados_uc16
        if sobra_uc16 > 0:
            capacity_mmtc_bytes += sobra_uc16 / 2.0
            capacity_uc3_bytes += sobra_uc16 / 2.0

        # --- PROCESSAMENTO mMTC ---
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
                
        # ALOCAÇÃO WORK-CONSERVING: eMBB herda o que sobrou do mMTC
        sobra_mmtc = capacity_mmtc_bytes - bytes_enviados_mmtc
        if sobra_mmtc > 0:
            capacity_uc3_bytes += sobra_mmtc

        # --- PROCESSAMENTO eMBB ---
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
        
        # --- COMPILAÇÃO ---
        stats['throughput_uc16_bytes'] = bytes_enviados_uc16
        stats['throughput_uc3_bytes'] = bytes_enviados_uc3
        stats['throughput_mmtc_bytes'] = bytes_enviados_mmtc
        
        if latencias_uc16: stats['avg_latency_uc16'] = sum(latencias_uc16) / len(latencias_uc16)
        if latencias_uc3: stats['avg_latency_uc3'] = sum(latencias_uc3) / len(latencias_uc3)
        if latencias_mmtc: stats['avg_latency_mmtc'] = sum(latencias_mmtc) / len(latencias_mmtc)
            
        if self.queue_uc16: stats['avg_wisun_delay_uc16'] = sum(item[0].d_wisun for item in self.queue_uc16) / len(self.queue_uc16)
        if self.queue_uc3: stats['avg_wisun_delay_uc3'] = sum(item[0].d_wisun for item in self.queue_uc3) / len(self.queue_uc3)
        if self.queue_mmtc: stats['avg_wisun_delay_mmtc'] = sum(item[0].d_wisun for item in self.queue_mmtc) / len(self.queue_mmtc)
            
        return stats

    def serve_queues(self, capacity_uc16_bytes: float, capacity_uc3_bytes: float, capacity_mmtc_bytes: float, current_time: float) -> Dict[str, float]:
        stats = {
            'throughput_uc3_bytes': 0.0, 'throughput_uc16_bytes': 0.0, 'throughput_mmtc_bytes': 0.0, 
            'pdb_violations_uc16': 0, 'pdb_violations_uc3': 0, 'pdb_violations_mmtc': 0,
            'avg_latency_uc16': 0.0, 'avg_latency_uc3': 0.0, 'avg_latency_mmtc': 0.0, 
            'avg_wisun_delay_uc16': 0.0, 'avg_wisun_delay_uc3': 0.0, 'avg_wisun_delay_mmtc': 0.0,
            'total_pacotes_uc16': 0, 'total_pacotes_uc3': 0, 'total_pacotes_mmtc': 0
        }
        
        # --- PROCESSAMENTO URLLC ---
        bytes_enviados_uc16 = 0
        latencias_uc16 = []
        while self.queue_uc16 and (bytes_enviados_uc16 + self.queue_uc16[0][0].tamanho_bytes) <= capacity_uc16_bytes:
            p, tempo_entrada_fila = self.queue_uc16.popleft()
            stats['total_pacotes_uc16'] += 1  
            self.bytes_uc16 -= p.tamanho_bytes
            bytes_enviados_uc16 += p.tamanho_bytes
            
            atraso_fila = (current_time - tempo_entrada_fila) * 1000.0 
            latencia_e2e = p.d_wisun + atraso_fila 
            latencias_uc16.append(latencia_e2e)
            
            if latencia_e2e > self.pdb_uc16:
                stats['pdb_violations_uc16'] += 1

        # ALOCAÇÃO WORK-CONSERVING: Redistribui banda ociosa do URLLC
        sobra_uc16 = capacity_uc16_bytes - bytes_enviados_uc16
        if sobra_uc16 > 0:
            capacity_mmtc_bytes += sobra_uc16 / 2.0
            capacity_uc3_bytes += sobra_uc16 / 2.0

        # --- PROCESSAMENTO mMTC ---
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
                
        # ALOCAÇÃO WORK-CONSERVING: eMBB herda o que sobrou do mMTC
        sobra_mmtc = capacity_mmtc_bytes - bytes_enviados_mmtc
        if sobra_mmtc > 0:
            capacity_uc3_bytes += sobra_mmtc

        # --- PROCESSAMENTO eMBB ---
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
        
        # --- COMPILAÇÃO ---
        stats['throughput_uc16_bytes'] = bytes_enviados_uc16
        stats['throughput_uc3_bytes'] = bytes_enviados_uc3
        stats['throughput_mmtc_bytes'] = bytes_enviados_mmtc
        
        if latencias_uc16: stats['avg_latency_uc16'] = sum(latencias_uc16) / len(latencias_uc16)
        if latencias_uc3: stats['avg_latency_uc3'] = sum(latencias_uc3) / len(latencias_uc3)
        if latencias_mmtc: stats['avg_latency_mmtc'] = sum(latencias_mmtc) / len(latencias_mmtc)
            
        if self.queue_uc16: stats['avg_wisun_delay_uc16'] = sum(item[0].d_wisun for item in self.queue_uc16) / len(self.queue_uc16)
        if self.queue_uc3: stats['avg_wisun_delay_uc3'] = sum(item[0].d_wisun for item in self.queue_uc3) / len(self.queue_uc3)
        if self.queue_mmtc: stats['avg_wisun_delay_mmtc'] = sum(item[0].d_wisun for item in self.queue_mmtc) / len(self.queue_mmtc)
            
        return stats

    def get_queue_sizes(self) -> tuple:
        """Retorna a pressão atual nas filas para a IA 'enxergar' o estado da rede"""
        return self.bytes_uc16, self.bytes_uc3, self.bytes_mmtc