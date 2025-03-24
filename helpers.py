import os
import json
import time
import yaml
import shutil

from datetime import datetime
from typing import Callable, Any
from functools import partial, wraps
from typing import Any, List, Union, Callable, Optional

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from django.core.cache import cache
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from rest_framework.response import Response
from tqdm import tqdm

def is_null_value(value: Any, check_empty_string: bool = True) -> bool:
    """Check if the passed value is null, optionally checks for empty strings as nulls"""
    if pd.isnull(value) or (check_empty_string and isinstance(value, str) and len(value.strip()) == 0):
        return True
    return False

def get_current_timestamp(timestamp_type: str="unix") -> Union[int, str]:
    """Returns the current timestamp in the specified format."""
    current_dt = datetime.now()
    
    if timestamp_type == 'unix':
        return int(current_dt.timestamp())
    elif timestamp_type == 'formatted':
        return current_dt.strftime("%Y%m%d%H%M%S")
    else:
        raise ValueError("Invalid timestamp_type. Use 'unix' or 'formatted'.")

def format_timestamp(timestamp: Union[int, str]):
    """Converts a given timestamp to a human-readable format."""

    if isinstance(timestamp, int):  # Unix timestamp
        dt_object = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, str):  # String timestamp in format YYYYMMDDHHMMSS
        dt_object = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
    else:
        raise ValueError("Timestamp must be an int (Unix timestamp) or str (formatted timestamp)")

    formatted_str = dt_object.strftime("%Y%m%d%H%M%S")
    return formatted_str

def get_file_extension(filepath: str) -> str:
    """Return the file extension for a given filepath."""
    return filepath.split(".")[-1]

def read_file(filepath: str, log: bool=False, **kwargs) -> Any:
    """Read and return the content of a file, supporting json, yaml, csv, and xlsx formats."""
    if isinstance(filepath, str):
        extension = get_file_extension(filepath)
    else:
        extension = get_file_extension(filepath.name)

    try:
        if extension == "json":
            with open(filepath, "r") as file:
                data = json.load(file)
        
        elif extension == "yaml":
            with open(filepath, "r") as file:
                data = yaml.safe_load(file)

        elif extension == "csv":
            data = pd.read_csv(filepath)

        elif extension == "parquet":
            data = pd.read_parquet(filepath, **kwargs)

        elif extension == "xlsx":
            data = pd.read_excel(filepath, **kwargs)

        else:
            raise ValueError(f"Unsupported file with format {extension}.")
    except Exception:
        raise

    return data

def write_file(data: Any, filepath: str, **kwargs) -> None:
    """Write data to a file, supporting json, yaml, csv, and xlsx formats."""
    extension = filepath.split(".")[-1]
    try:
        if extension == "json":
            with open(filepath, "w") as file:
                json.dump(data, file, indent=4)

        elif extension == "yaml":
            with open(filepath, "w") as file:
                yaml.safe_dump(data, file)

        elif extension == "csv":
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame for CSV format.")
            save_index=kwargs.pop("index", False)
            data.to_csv(filepath, index=save_index, **kwargs)

        elif extension == "xlsx":
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame for XLSX format.")
            sheet_name = kwargs.get("sheet_name", "Sheet1")
            data.to_excel(filepath, index=False, sheet_name=sheet_name)
        
        elif extension == "parquet":
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame for Parquet format.")
            compression = kwargs.pop("compression", "snappy")
            engine = kwargs.pop("engine", "pyarrow")
            data.to_parquet(filepath, compression=compression, engine=engine, **kwargs)

        else:
            raise ValueError(f"Unsupported file format: {extension}.")

    except Exception as e:
        raise ValueError(f"An error occurred while writing to file: {e}")
    
def apply_function_concurrently(data_list, function, num_workers=5, task_name="Processing", **kwargs):
    """Apply a function to a list of data using multiple threads, preserving order, with additional constant arguments."""
    
    # Create a partial function including the additional arguments
    partial_function = partial(function, **kwargs)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit jobs to the executor, applying the partial function
        future_to_index = {executor.submit(partial_function, data): i for i, data in enumerate(data_list)}
        
        # Initialize a list of None values to hold the results
        results = [None] * len(data_list)
        
        # Use tqdm for progress tracking
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=task_name):
            try:
                result = future.result()  # This will re-raise any exception caught in the function
            except Exception as e:
                print(f"Error processing task: {e}")
                continue
            index = future_to_index[future]  # Get the original index
            results[index] = result  # Place the result in the correct position
            
    return results

def retry(max_retries: int = 3, delay: int = 2, return_value: Optional[Any] = None):
    """A decorator that retries a function if it raises an exception, 
    re-raising the original exception with the function name appended if all retries fail,
    or returning a provided value if return_value is specified."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None  # Track the last exception
            for attempt in range(1, max_retries + 1):
                try:
                    # Attempt to call the function
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt} failed in function '{func.__name__}': {e}")
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay)  # Wait before the next retry
                    else:
                        print(f"Max retries reached in function '{func.__name__}'.")
                        if return_value is not None:
                            print(f"Returning the fallback value due to repeated failures: {return_value}")
                            return return_value
                        # Raise the exception with the function name appended
                        raise type(e)(f"{e} (in function '{func.__name__}')") from e
        return wrapper
    return decorator

def remove_path(path):
    """Remove a file or directory."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)
    else:
        raise ValueError(f"Path {path} does not exist or is not a file or directory")

def get_cache_key(filepath):
    """To create the cache key for the given filename"""
    return f"data_{filepath}"

def set_or_update_cache(filepath, data, timeout=600):
    """Update the existing cache using the key"""
    cache_key = get_cache_key(filepath)
    cache.set(cache_key, data, timeout=timeout)
    print("Cache Updated!")

def get_file_data(filepath):
    """Helper method to read file data and cache it if not already cached."""
    cache_key = get_cache_key(filepath)
    data = cache.get(cache_key)

    if data is None:
        print("Reading data. Not found in cache.")
        data = read_file(filepath)
        set_or_update_cache(filepath, data)
    else:
        print(f"Data {cache_key} found in cache.")
    
    return data

def get_paginated_response(data, page_number, page_size):
    """Helper method to paginate the data."""
    paginator = Paginator(data.to_dict(orient='records'), per_page=page_size)
    try:
        page = paginator.page(page_number)
    except PageNotAnInteger:
        page = paginator.page(1)
    except EmptyPage:
        page = paginator.page(paginator.num_pages)

    return {
        'count': paginator.count,
        'total_pages': paginator.num_pages,
        'current_page': page.number,
        'next_page': page.next_page_number() if page.has_next() else None,
        'previous_page': page.previous_page_number() if page.has_previous() else None,
        'results': page.object_list
    }


def get_error_response(message, status_code):
    """Helper method to return a standardized error response."""
    return Response({'status': 'error', 'message': message}, status=status_code)