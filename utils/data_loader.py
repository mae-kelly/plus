import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
import logging

logger = logging.getLogger(__name__)

class TabularDataset(Dataset):
    def __init__(self, data: Dict[str, List], features: np.ndarray = None):
        self.data = data
        self.features = features
        self.columns = list(data.keys())
        self.length = len(next(iter(data.values())))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        row = {}
        for col in self.columns:
            row[col] = self.data[col][idx]
        
        if self.features is not None:
            return self.features[idx], row
        
        return row

class FastTabularDataLoader:
    def __init__(self, batch_size: int = 128, num_workers: int = 4, device: str = 'mps'):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.prefetch_factor = 2
    
    def create_loader(self, data: Dict[str, List], features: np.ndarray = None) -> DataLoader:
        dataset = TabularDataset(data, features)
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=None,
            persistent_workers=False
        )
        
        return loader
    
    def load_parquet_batch(self, file_path: str, batch_size: int = 10000):
        parquet_file = pq.ParquetFile(file_path)
        
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            
            data = {col: df[col].tolist() for col in df.columns}
            
            yield data
    
    def load_csv_streaming(self, file_path: str, chunksize: int = 10000):
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            data = {col: chunk[col].tolist() for col in chunk.columns}
            
            yield data
    
    def parallel_load(self, file_paths: List[str], loader_func) -> List[Dict]:
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for file_path in file_paths:
                future = executor.submit(loader_func, file_path)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to load file: {e}")
        
        return results
    
    def prepare_batch(self, batch: Dict[str, List]) -> torch.Tensor:
        numerical_features = []
        
        for col, values in batch.items():
            numerical = []
            
            for val in values:
                try:
                    numerical.append(float(val))
                except:
                    numerical.append(0.0)
            
            numerical_features.append(numerical)
        
        features = torch.tensor(numerical_features, dtype=torch.float32).T
        
        return features.to(self.device)
    
    def create_embedding_batch(self, texts: List[str], model) -> torch.Tensor:
        with torch.no_grad():
            embeddings = model.encode(
                texts,
                convert_to_tensor=True,
                device=self.device,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
        
        return embeddings

class CachedDataLoader:
    def __init__(self, cache_dir: str = 'cache', max_cache_size_gb: float = 8.0):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024
        self.cache = {}
        self.cache_size = 0
    
    def load_with_cache(self, key: str, loader_func, *args, **kwargs):
        if key in self.cache:
            logger.debug(f"Cache hit: {key}")
            return self.cache[key]
        
        data = loader_func(*args, **kwargs)
        
        data_size = self._estimate_size(data)
        
        if self.cache_size + data_size > self.max_cache_size:
            self._evict_cache()
        
        self.cache[key] = data
        self.cache_size += data_size
        
        logger.debug(f"Cached: {key} ({data_size / 1024 / 1024:.2f} MB)")
        
        return data
    
    def _estimate_size(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        elif isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        else:
            import sys
            return sys.getsizeof(obj)
    
    def _evict_cache(self):
        if not self.cache:
            return
        
        oldest_key = next(iter(self.cache))
        evicted_size = self._estimate_size(self.cache[oldest_key])
        
        del self.cache[oldest_key]
        self.cache_size -= evicted_size
        
        logger.debug(f"Evicted from cache: {oldest_key}")
    
    def clear_cache(self):
        self.cache.clear()
        self.cache_size = 0
        logger.info("Cache cleared")