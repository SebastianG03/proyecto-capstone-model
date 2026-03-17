from enum import Enum
import logging

import psutil
import os
import gc
import sys
import json
import threading
import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import inspect
import traceback

from app.logger import get_logger 
from app.utils.routes import OUTPUT_REPORTS_DIR

class MemoryTrigger(str, Enum):
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    ALERT = "alert"

@dataclass
class MemorySnapshot:
    timestamp: str
    total_memory_mb: float
    memory_used_mb: float
    memory_percent: float
    top_variables: List[Dict[str, Any]]
    trigger: MemoryTrigger = MemoryTrigger.MANUAL
    
    def to_dict(self):
        return asdict(self)


class MemoryReporter:
    def __init__(
        self,
        match_id: int,
        interval_seconds: int = 300,
        alert_threshold_mb: float = 1000,
        top_n_variables: int = 15,
        enable_scheduled_monitoring: bool = True
    ):
        self.report_file = OUTPUT_REPORTS_DIR / f"memory_report_{match_id}.json"
        self.interval = interval_seconds
        self.alert_threshold = alert_threshold_mb
        self.top_n = top_n_variables
        self.enable_scheduled = enable_scheduled_monitoring
        
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self.logger = get_logger(logging.DEBUG)
        
        self._init_report_file()
        
        if self.enable_scheduled:
            self.start_scheduled_monitoring()
    
    def _init_report_file(self):
        """
        Inicializa el archivo de reporte de memoria, si no existe.
        
        Escribe metadata de la inicializacion en el archivo de reporte.
        """
        try:
            if not self.report_file.exists() and not self.report_file.is_file():
                self.report_file.touch(exist_ok=False)
                self.logger.info(f"[MemoryReporter] Archivo de reporte de memoria creado. {self.report_file.as_posix()}")
                metadata = self.generate_metadata()

                result = self.report_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
                self.logger.info(f"[MemoryReporter] Metadata guardada en el archivo de reporte de memoria {result}.")
        except Exception as e:
            self.logger.error(f"Error al crear el archivo de reporte de memoria: {traceback.format_exc()}")
            raise

    def generate_metadata(self, updated: bool = False) -> dict:
        metadata = {
                "metadata": {
                    "pid": os.getpid(),
                    "interval_seconds": self.interval,
                    "alert_threshold_mb": self.alert_threshold
                },
                "snapshots": []
            }
        actual_time = datetime.datetime.now().isoformat()
        
        if updated:
            metadata["metadata"]["updated_at"] = actual_time
        else:
            metadata["metadata"]["created_at"] = actual_time

        return metadata

    def get_memory_info(self) -> Dict[str, float]:
        """
        Obtiene informacion de memoria del proceso actual.
        
        Devuelve un diccionario con las siguientes claves:
        - rss_mb: memoria RSS utilizada por el proceso en megabytes.
        - vms_mb: memoria VMS utilizada por el proceso en megabytes.
        - percent: porcentaje de memoria utilizada por el proceso.
        - system_total_mb: memoria total disponible en el sistema en megabytes.
        - system_available_mb: memoria disponible en el sistema en megabytes.
        - system_percent: porcentaje de memoria disponible en el sistema.
        
        :return: Diccionario con informacion de memoria.
        :rtype: Dict[str, float]
        """
        mem_info = self.process.memory_info()
        system_mem = psutil.virtual_memory()
        
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent(),
            "system_total_mb": system_mem.total / 1024 / 1024,
            "system_available_mb": system_mem.available / 1024 / 1024,
            "system_percent": system_mem.percent
        }
    
    def get_large_variables(self, local_scope: Optional[dict] = None) -> List[Dict[str, Any]]:
        """
        Devuelve una lista de los N variables más grandes en memoria, 
        incluyendo su nombre, tipo, tamaño en bytes y megabytes.
        
        Puede especificar un ámbito de variables locales (local_scope) 
        para buscar variables en ese ámbito en lugar del actual.
        
        Ignora variables que comienzan con '_' y variables internas como 
        gc, sys, psutil, json, inspect, etc.
        
        Para contenedores como listas, tuplas, conjuntos y diccionarios, 
        estima el tamaño total incluyendo sus elementos.
        
        Solo incluye variables que ocupen más de 1KB de memoria.
        
        :param local_scope: Diccionario con variables locales para buscar.
        :type local_scope: Optional[dict]
        :return: Lista de variables más grandes en memoria.
        :rtype: List[Dict[str, Any]]
        """
        gc.collect()
        
        if local_scope is None:
            frame = inspect.currentframe().f_back
            if frame is None:
                return []

            variables = {**frame.f_locals, **frame.f_globals}
        else:
            variables = local_scope
        
        sizes = []
        
        for name, obj in variables.items():
            if name.startswith('_') or name in (
                'gc', 'sys', 'psutil', 'json', 
                'In', 'Out', 'memory_reporter',
                'inspect', 'frame'):
                continue
            
            try:
                size_bytes = sys.getsizeof(obj)
                
                if isinstance(obj, (list, tuple, set)):
                    try:
                        size_bytes += sum(sys.getsizeof(item) for item in obj)
                    except:
                        pass
                elif isinstance(obj, dict):
                    try:
                        size_bytes += sum(sys.getsizeof(k) + sys.getsizeof(v) 
                                        for k, v in obj.items())
                    except:
                        pass
                
                if size_bytes > 1024:
                    size_mb = size_bytes / 1024 / 1024
                    
                    type_info = type(obj).__name__
                    extra_info = {}
                    
                    if hasattr(obj, 'shape'):
                        extra_info['shape'] = str(obj.shape)
                        extra_info['dtype'] = str(getattr(obj, 'dtype', 'unknown'))
                    elif hasattr(obj, '__len__'):
                        try:
                            extra_info['length'] = len(obj)
                        except:
                            pass
                    
                    sizes.append({
                        "variable_name": name,
                        "type": type_info,
                        "size_mb": round(size_mb, 4),
                        "size_bytes": size_bytes,
                        **extra_info
                    })
                    
            except Exception as e:
                self.logger.error(f"Memory Tracker Error on extracting variables {traceback.format_exc()}")
                continue
        
        sizes.sort(key=lambda x: x['size_mb'], reverse=True)
        return sizes[:self.top_n]
    
    def take_snapshot(self, trigger: MemoryTrigger = MemoryTrigger.MANUAL, local_scope: Optional[dict] = None) -> MemorySnapshot:
        """
        Take a memory snapshot of the current process.

        Args:
            trigger (str): The reason for taking the snapshot. Defaults to "manual".
            local_scope (dict, optional): The local variables to include in the snapshot. Defaults to None.

        Returns:
            MemorySnapshot: The snapshot of the current process memory usage.
        """
        mem_info = self.get_memory_info()
        large_vars = self.get_large_variables(local_scope)
        
        snapshot = MemorySnapshot(
            timestamp=datetime.datetime.now().isoformat(),
            total_memory_mb=round(mem_info['system_total_mb'], 2),
            memory_used_mb=round(mem_info['rss_mb'], 2),
            memory_percent=round(mem_info['percent'], 2),
            trigger=trigger,
            top_variables=large_vars
        )
        
        self._save_snapshot(snapshot)
        
        if mem_info['rss_mb'] > self.alert_threshold or trigger == 'alert':
            print(f"🚨 ALERTA MEMORIA: {mem_info['rss_mb']:.1f} MB usados")
            for var in large_vars[:5]:
                print(f"   🔴 {var['variable_name']}: {var['size_mb']:.2f} MB ({var['type']})")
        
        return snapshot
    
    def _save_snapshot(self, snapshot: MemorySnapshot):
        """Guarda la snapshot en el archivo JSON"""
        try:
            raw_data = self.report_file.read_text('utf-8')
            data = json.loads(raw_data)
            
            data['snapshots'].append(snapshot.to_dict())
            
            self.report_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except FileNotFoundError as fe:
            self.logger.error(f"Memory Tracker Error on saving snapshot: {traceback.format_exc()}")
            raise fe
        except json.JSONDecodeError as jde:
            self.logger.error(f"Memory Tracker Error on saving snapshot: {traceback.format_exc()}")
            raise jde
        except Exception as e:
            self.logger.error(f"Memory Tracker Error on saving snapshot: {traceback.format_exc()}")
            raise e
    
    def _scheduled_monitor(self):
        """
        Monitoreo programado que toma snapshots de memoria
        periodicamente.

        Mientras no se haya establecido el evento de parada,
        toma un snapshot de memoria cada `interval` segundos y
        lo guarda en el archivo de reporte.

        :param self:
        :type self: MemoryReporter
        :return: None
        :rtype: None
        """
        while not self._stop_event.is_set():
            self.take_snapshot(trigger=MemoryTrigger.SCHEDULED)
            self._stop_event.wait(self.interval)
    
    def start_scheduled_monitoring(self):
        """
        Inicia el monitoreo programado que toma snapshots de memoria
        periodicamente.

        Mientras no se haya establecido el evento de parada,
        toma un snapshot de memoria cada `interval` segundos y
        lo guarda en el archivo de reporte.

        :param self:
        :type self: MemoryReporter
        :return: None
        :rtype: None
        """
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._scheduled_monitor,
                daemon=True,
                name="MemoryMonitor"
            )
            self._monitor_thread.start()
            print(f"✅ Monitoreo de memoria iniciado cada {self.interval/60:.0f} minutos")
            print(f"📁 Reportes guardados en: {self.report_file.as_posix()}")
    
    def stop(self):
        """
        Detiene el monitoreo programado que toma snapshots de memoria
        periodicamente.
        
        :param self:
        :type self: MemoryReporter
        :return: None
        :rtype: None
        """
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        print("⏹️ Monitoreo de memoria detenido")
    
    def after_loop(self, label: str = "post_loop", local_vars: Optional[dict] = None):
        """
        Llama esto después de cada bucle importante.
        Opcionalmente pasa locals() para análisis más preciso.
        """
        return self.take_snapshot(trigger=f"loop_{label}", local_scope=local_vars)
    
    def get_summary(self) -> Dict:
        """
        Returns a summary of the memory usage of the current process.

        The summary contains the following information:

        - total_snapshots: The total number of snapshots taken.
        - first_timestamp: The timestamp of the first snapshot.
        - last_timestamp: The timestamp of the last snapshot.
        - memory_stats: A dictionary containing memory usage statistics.
        - highest_memory_snapshot: The snapshot with the highest memory usage.

        If there is an error while reading the report file, returns a dictionary with an "error" key containing the error message.

        :return: A dictionary containing the memory usage summary.
        :rtype: Dict
        """
        try:
            data = self.report_file.read_text('utf-8')
            data = json.loads(data)
            # with open(self.report_file, 'r', encoding='utf-8') as f:
            #     data = json.load(f)
            
            snapshots = data['snapshots']
            if not snapshots:
                return {"message": "No hay snapshots registradas"}
            
            mem_usage = [s['memory_used_mb'] for s in snapshots]
            
            return {
                "total_snapshots": len(snapshots),
                "first_timestamp": snapshots[0]['timestamp'],
                "last_timestamp": snapshots[-1]['timestamp'],
                "memory_stats": {
                    "min_mb": min(mem_usage),
                    "max_mb": max(mem_usage),
                    "avg_mb": sum(mem_usage) / len(mem_usage),
                    "current_mb": mem_usage[-1]
                },
                "highest_memory_snapshot": max(snapshots, key=lambda x: x['memory_used_mb'])
            }
        except Exception as e:
            return {"error": str(e)}
