# from dataclasses import dataclass as _dataclass_need_patch
# from functools import partial
# dataclass = partial(_dataclass_need_patch, eq=False)

from typing import List

from typing_extensions import Annotated
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import PydanticUndefined, core_schema

from pydantic import BaseModel, PlainSerializer, PydanticUndefinedAnnotation

CustomStr = Annotated[
    List, PlainSerializer(lambda x: ' '.join(x), return_type=PydanticUndefined)
]

class StudentModel(BaseModel):
    courses: CustomStr
    name: SkipJsonSchema[str]

student = StudentModel(courses=['Math', 'Chemistry', 'English'], name="wtf")
print(student.model_dump_json())
