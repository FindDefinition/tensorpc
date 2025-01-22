from typing import Any, Iterator
from pydantic.dataclasses import dataclass, is_pydantic_dataclass
from pydantic import Field as field

from dataclasses import asdict, is_dataclass, fields, replace
from pydantic import BaseModel, ConfigDict, ValidationError, Field

# from dataclasses import dataclass
# from dataclasses import field


class PyDanticConfigForAnyObject:
    arbitrary_types_allowed = True


def validate_selected_fields_from_model(model: Any, data_dict: dict):
    """
    Validates fields one by one, yielding any ValidationErrors encountered.
    This allows immediate error handling as soon as any validation fails.
    see also https://github.com/pydantic/pydantic/discussions/7367

    Args:
        model: The Pydantic model to validate against
        data_dict: Dictionary containing data to validate
        
    Yields:
        ValidationError: Any validation errors encountered during the process
    """
    for k, v in data_dict.items():
        model.__pydantic_validator__.validate_assignment(model.model_construct(), k, v)
