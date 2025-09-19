import time
import logging
import functools
import asyncio
from typing import Callable, Any
import psutil
import os

logger = logging.getLogger(__name__)

def monitor_performance(func_name: str = None):
    """
    Decorator to monitor performance function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wrap(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss/1024/1024

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                execution_time = (time.time() - start_time) * 1000 # ms
                end_memory = psutil.Process().memory_info().rss/1024/1024
                memory_used = end_memory - start_memory

                name = func_name or func.__name__
                logger.info(f"Performance: {name} - {execution_time:.2f}ms, Memory: + {memory_used:.2f}MB")

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                name = func_name or func.__name__
                logger.error(f"Performance: {name} FAILED after {execution_time:.2f}ms - {str(e)}")
                raise

            return async_wrapper
        return decorator
    
class PerformanceOptimizer:
    """
    Handles performance optimization for your hardware specs
    """

    @staticmethod
    def get_optimal_batch_size() -> int:
        """
        Calculate optimal batch size for embedding generation based on available memory
        """

        # Get available memory 
        memory_gb = psutil.virtual_memory().available/ (1024*3)

        # Conservative batch sizes for 8GB ram system
        if memory_gb > 6:
            return 32
        elif memory_gb > 4:
            return 16
        else:
            return 8
        
    @staticmethod
    def optimize_faiss_index(dimension: int, expected_vectors: int) -> str:
        """
        Choose optimal FAISS index type based on expected data size
        """

        # For this use case (100+ documents, likely <10k vectors)
        if expected_vectors < 1000:
            return "IndexFlatIP" # Exact search, good for small datasets
        elif expected_vectors < 10000:
            return "IndexIVFFlat" # Good balance for medium datasets
        else:
            return "IndexIVFPQ" # Memory efficient for large datasets
        
    @staticmethod
    def should_use_cpu_optimization() -> bool:
        """
        Determine if CPU optimization should be enabled
        """
        cpu_count = psutil.cpu_count()
        return cpu_count >= 4 # My cpu has 4 cores
    
    def get_processing_recommendation() -> dict:
        """
        Get performance recommendations for current system
        """
        memory_gb = psutil.virtual_memory().total/(1024*3)
        cpu_count = psutil.cpu_count()

        return {
            "batch_size": PerformanceOptimizer.get_optimal_batch_size(),
            "parallel_processing": cpu_count >= 4,
            "memory_warning": memory_gb < 8,
            "recommended_max_file_size": min(50, int(memory_gb*6)),
            "faiss_index_type": PerformanceOptimizer.optimize_faiss_index(384, 5000)
        }
