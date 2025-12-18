from typing import Callable, Any
from datetime import datetime

def calculate_time(func: Callable) -> Callable:

    # Closure
    def closure(*args, **kwargs) -> Any:

        start = datetime.now()

        result = func(*args, **kwargs)

        end = datetime.now()

        # Elapsed time in seconds
        elapsed_time = end - start
        elapsed_time = elapsed_time.total_seconds()

        print(f"Function {func.__name__} executed in {elapsed_time} seconds.")

        return result
    
    return closure