import dataclasses
from typing import Any, Dict, List, Union
from typing_extensions import Literal
from tensorpc.flow.jsonlike import DataClassWithUndefined, Undefined, undefined

@dataclasses.dataclass
class PyrightConfig:
    disableLanguageServices: Union[bool, Undefined] = undefined
    disableOrganizeImports: Union[bool, Undefined] = undefined
    openFilesOnly: Union[bool, Undefined] = undefined
    useLibraryCodeForTypes: Union[bool, Undefined] = undefined

@dataclasses.dataclass
class PythonAnalysisConfig:
    autoImportCompletions: Union[bool, Undefined] = undefined
    autoSearchPaths: Union[bool, Undefined] = undefined
    diagnosticMode: Union[Literal["openFilesOnly", "workspace"], Undefined] = undefined
    diagnosticSeverityOverrides: Union[Dict[str, Any], Undefined] = undefined
    extraPaths: Union[List[str], Undefined] = undefined
    logLevel: Union[Literal["Error", "Warning", "Information", "Trace"], Undefined] = undefined
    stubPath: Union[str, Undefined] = undefined
    typeCheckingMode: Union[Literal["basic", "strict", "off"], Undefined] = undefined
    typeshedPaths: Union[List[str], Undefined] = undefined
    useLibraryCodeForTypes: Union[bool, Undefined] = undefined
    pythonPath: Union[str, Undefined] = undefined
    venvPath: Union[str, Undefined] = undefined
    include: Union[List[str], Undefined] = undefined
    exclude: Union[List[str], Undefined] = undefined

@dataclasses.dataclass
class PythonConfig:
    analysis: PythonAnalysisConfig = dataclasses.field(default_factory=PythonAnalysisConfig)

@dataclasses.dataclass
class LanguageServerConfig(DataClassWithUndefined):
    pyright: PyrightConfig = dataclasses.field(default_factory=PyrightConfig)
    python: PythonConfig = dataclasses.field(default_factory=PythonConfig)