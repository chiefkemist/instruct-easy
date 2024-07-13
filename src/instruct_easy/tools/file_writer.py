
from functools import wraps
from typing import Callable, List, Any

from pydantic import BaseModel, Field


class FileDetails(BaseModel):
    file_name: str = Field(..., description="The name of the file to be written.")
    file_content: str = Field(..., description="The content to be written to the file.")
    file_path: str = Field(..., description="The path where the file will be written.")
    file_extension: str = Field(..., description="The file extension of the file to be written.")
    file_mode: str = Field(..., description="The mode in which the file will be written.")
    file_encoding: str = Field(..., description="The encoding of the file to be written.")


def file_writer(base_dir: str):
    def decorator_file_writer(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(param: str, input: List[FileDetails] = None) -> Any:
            os.makedirs(base_dir, exist_ok=True)
            for file_detail in input:
                file_path = f"{base_dir}/{file_detail.file_path}/{file_detail.file_name}.{file_detail.file_extension}"
                with open(file_path, file_detail.file_mode, encoding=file_detail.file_encoding) as file:
                    file.write(file_detail.file_content)
            result = func(param, input)
            return result

        return wrapper

    return decorator_file_writer
