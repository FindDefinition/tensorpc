import abc
import copy
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import (Any, Generic, Optional, TypeVar, Union,
                    get_type_hints)

from mashumaro.codecs.basic import BasicDecoder, BasicEncoder
import base64
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.annolib import (AnnotatedType, BackendOnlyProp,
                                   DataclassType, Undefined,
                                   as_dict_no_undefined, child_type_generator,
                                   child_type_generator_with_dataclass,
                                   get_type_hints_with_cache, is_annotated,
                                   parse_type_may_optional_undefined)
from tensorpc.core.tree_id import UniqueTreeId

from .draft import (DraftASTNode, DraftASTType, DraftUpdateOp, JMESPathOpType, evaluate_draft_ast,
                    apply_draft_update_ops, apply_draft_update_ops_to_json)

T = TypeVar("T", bound=DataclassType)

class DraftStoreBackendBase(abc.ABC):
    @abc.abstractmethod
    async def read(self, path: str) -> Optional[Any]: 
        """Read a data from path. if return None, it means the path is not exist."""

    @abc.abstractmethod
    async def write(self, path: str, data: Any) -> None:
        """Write data to path"""

    @abc.abstractmethod
    async def update(self, path: str, ops: list[DraftUpdateOp]) -> None:
        """Update data in path by draft update ops"""

    @abc.abstractmethod
    async def remove(self, path: str) -> None:
        """Remove data in path"""

class DraftFileStoreBackendBase(DraftStoreBackendBase):
    @abc.abstractmethod
    async def glob_read(self, path_with_glob: str) -> dict[str, Any]: 
        """Read a data from path via glob (not rglob). usually used when you use real file system 
        as backend.
        """

class DraftFileStoreBackendInMemory(DraftFileStoreBackendBase):
    def __init__(self):
        self._data: dict[str, Any] = {}

    async def read(self, path: str) -> Optional[Any]:
        return self._data.get(path)

    async def write(self, path: str, data: Any) -> None:
        self._data[path] = data

    async def update(self, path: str, ops: list[DraftUpdateOp]) -> None:
        data = self._data.get(path)
        if data is None:
            raise ValueError(f"path {path} not exist")
        apply_draft_update_ops_to_json(data, ops)
        # self._data[path] = new_data

    async def remove(self, path: str) -> None:
        self._data.pop(path, None)

    async def glob_read(self, path_with_glob: str) -> dict[str, Any]:
        res = {}
        for k, v in self._data.items():
            if Path(k).match(path_with_glob):
                res[k] = v
        return res


@dataclasses.dataclass
class DraftStoreMetaBase:
    pass


@dataclasses.dataclass
class DraftStoreMapMeta(DraftStoreMetaBase):
    key: str = ""
    lazy_key_field: Optional[str] = None

    @property 
    def path(self):
        return "{}"

    def get_glob_path(self):
        return self.path.replace('{}', '*')

    def encode_key(self, key: str):
        # encode to b64
        return base64.b64encode(key.encode()).decode()

    def decode_key(self, key: str):
        return base64.b64decode(key.encode()).decode()


def _asdict_map_trace_inner(obj, field_types: list, dict_factory, obj_factory=None, cur_field_type=None):
    if dataclasses.is_dataclass(obj):
        result = []
        type_hints = get_type_hints_with_cache(type(obj), include_extras=True)
        for f in dataclasses.fields(obj):
            field_types_field = field_types + [(type_hints[f.name], f.name, False)]
            v = getattr(obj, f.name)
            value = _asdict_map_trace_inner(v, field_types_field, dict_factory,
                                  obj_factory, type_hints[f.name] if isinstance(v, dict) else None)
            result.append((f.name, value, field_types_field))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(
            *[_asdict_map_trace_inner(v, field_types, dict_factory, obj_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_asdict_map_trace_inner(v, field_types, dict_factory, obj_factory)
                         for v in obj)
    elif isinstance(obj, dict):
        res = []
        for k, v in obj.items():
            field_types_field = field_types + [(cur_field_type, k, True)]
            kk = _asdict_map_trace_inner(k, field_types, dict_factory, obj_factory)
            vv = _asdict_map_trace_inner(v, field_types_field, dict_factory, obj_factory)
            res.append((kk, vv))
        return type(obj)(res)
    else:
        if obj_factory is not None:
            obj = obj_factory(obj)
        return copy.deepcopy(obj)

def _default_asdict_map_trace_factory(obj: list[tuple[str, Any, Any]]):
    return {k: v for k, v, _ in obj}

def asdict_map_trace(obj, dict_factory=_default_asdict_map_trace_factory):
    return _asdict_map_trace_inner(obj, [], dict_factory)

def _get_first_store_meta(ty):
    annotype = parse_type_may_optional_undefined(ty)
    if annotype.annometa:
        for m in annotype.annometa:
            if isinstance(m, DraftStoreMetaBase):
                return annotype, m
    return annotype, None

def _get_first_store_meta_by_annotype(annotype: AnnotatedType):
    if annotype.annometa:
        for m in annotype.annometa:
            if isinstance(m, DraftStoreMetaBase):
                return m
    return None

class _StoreAsDict:
    def __init__(self):
        self._store_pairs = []

    def _get_path_parts(self, types: list[tuple[Any, str, bool]]):
        parts: list[str] = []
        for t, k, is_dict in types:
            annotype, store_meta = _get_first_store_meta(t)

            if not is_dict:
                # k is field name or custom name
                if isinstance(store_meta, DraftStoreMapMeta):
                    parts.append(store_meta.key if store_meta.key else k)
            else:
                if isinstance(store_meta, DraftStoreMapMeta):
                    assert annotype.get_dict_key_anno_type().origin_type is str
                    parts.append(store_meta.encode_key(k))
                    # break
        return parts

    def _asdict_map_trace_factory(self, obj: list[tuple[str, Any, Any]]):
        res = {}
        for k, v, types in obj:
            t = types[-1][0]
            parts = self._get_path_parts(types[:-1])
            annotype, store_meta = _get_first_store_meta(t)
            if isinstance(store_meta, DraftStoreMapMeta):
                assert isinstance(v, Mapping) and annotype.get_dict_key_anno_type().origin_type is str
                storage_key = store_meta.key if store_meta.key else k
                for kk, vv in v.items():
                    self._store_pairs.append((parts + [storage_key, store_meta.encode_key(kk)], vv))
                continue 
            if isinstance(v, UniqueTreeId):
                res[k] = v.uid_encoded
            elif not isinstance(v, (Undefined, BackendOnlyProp)):
                res[k] = v
        return res

def validate_splitted_model_type(model_type: type[T]):
    type_hints = get_type_hints(model_type, include_extras=True)
    has_splitted_store = False
    for field in dataclasses.fields(model_type):
        field_type = type_hints[field.name]
        annotype, store_meta = _get_first_store_meta(field_type)
        if isinstance(store_meta, DraftStoreMapMeta):
            has_splitted_store = True
            assert annotype.is_dict_type() and annotype.get_dict_key_anno_type().origin_type is str
            value_type = annotype.get_dict_value_anno_type()
            if value_type.is_dataclass_type():
                validate_splitted_model_type(value_type.origin_type)
            else:
                # all non-dataclass field type shouldn't contain any store meta
                for child_type in annotype.child_types:
                    for t in child_type_generator_with_dataclass(child_type):
                        if is_annotated(t):
                            annometa = t.__metadata__
                            for m in annometa:
                                if isinstance(m, DraftStoreMetaBase):
                                    raise ValueError(f"subtype of field {field.name} with type {child_type} can't contain any store meta")
            continue 
        if annotype.is_dataclass_type():
            has_splitted_store |= validate_splitted_model_type(annotype.origin_type)
        else:
            # all non-dataclass field type shouldn't contain any store meta
            for t in child_type_generator_with_dataclass(field_type):
                if is_annotated(t):
                    annometa = t.__metadata__
                    for m in annometa:
                        if isinstance(m, DraftStoreMetaBase):
                            raise ValueError(f"subtype of field {field.name} with type {field_type} can't contain any store meta")
    return has_splitted_store

def get_splitted_update_model_ops(root_path: str, ops: list[DraftUpdateOp], model_before_update: Any):
    ops_with_paths: dict[str, list[DraftUpdateOp]] = {}
    new_items: dict[str, Any] = {}
    remove_items: set[str] = set()
    for op in ops:
        node: DraftASTNode = op.node 
        all_dict_getitem_nodes: list[DraftASTNode] = []
        paths: list[str] = []
        # convert absolute path (node) to relative path
        relative_node: DraftASTNode = copy.deepcopy(node)
        relative_node_cur = relative_node
        relative_node_last = None
        while relative_node_cur.children:
            if relative_node_cur.type == DraftASTType.DICT_GET_ITEM and isinstance(relative_node_cur.userdata, AnnotatedType):
                store_meta = _get_first_store_meta_by_annotype(relative_node_cur.userdata)
                if isinstance(store_meta, DraftStoreMapMeta):
                    if relative_node_last is not None:
                        relative_node_last.children = [DraftASTNode(DraftASTType.NAME, [], "")]
                    else:
                        relative_node = DraftASTNode(DraftASTType.NAME, [], "")
                    break 
            relative_node_last = relative_node_cur
            relative_node_cur = relative_node_cur.children[0]
        # print("?", op.op.name, op.opData, op.node.get_jmes_path(), relative_node.get_jmes_path())
        while node.children:
            if node.type == DraftASTType.DICT_GET_ITEM and isinstance(node.userdata, AnnotatedType):
                all_dict_getitem_nodes.append(node)
            elif node.type == DraftASTType.GET_ATTR and isinstance(node.userdata, AnnotatedType):
                all_dict_getitem_nodes.append(node)
            node = node.children[0]

        all_dict_getitem_nodes = all_dict_getitem_nodes[::-1]
        for node in all_dict_getitem_nodes:
            if node.type == DraftASTType.DICT_GET_ITEM:
                assert isinstance(node.userdata, AnnotatedType)
                store_meta = _get_first_store_meta_by_annotype(node.userdata)
                if isinstance(store_meta, DraftStoreMapMeta):
                    assert len(node.children) == 1, "getitem key can't be draft object (node)"
                    # paths.append()
                    paths.append(store_meta.encode_key(node.value))
            elif node.type == DraftASTType.GET_ATTR:
                assert isinstance(node.userdata, AnnotatedType)
                store_meta = _get_first_store_meta_by_annotype(node.userdata)
                if isinstance(store_meta, DraftStoreMapMeta):
                    storage_key = store_meta.key if store_meta.key else node.value
                    paths.append(storage_key)
        # print(paths)
        if not paths:
            if root_path not in ops_with_paths:
                ops_with_paths[root_path] = []
            ops_with_paths[root_path].append(op)
        else:
            # replace root node of relative_node with "value"
            relative_node_cur = relative_node
            relative_node_prev = None
            while relative_node_cur.children:
                relative_node_prev = relative_node_cur
                relative_node_cur = relative_node_cur.children[0]
            if relative_node_prev is None:
                relative_node = DraftASTNode(DraftASTType.GET_ATTR, [relative_node], "value")
            else:
                relative_node_prev.children = [DraftASTNode(DraftASTType.GET_ATTR, [relative_node_cur], "value")]

            if op.op == JMESPathOpType.DictUpdate:
                store_meta = _get_first_store_meta_by_annotype(relative_node.userdata)
                if isinstance(store_meta, DraftStoreMapMeta):
                    # we need to query prev model to determine the existance of the key
                    obj = evaluate_draft_ast(op.node, model_before_update)
                    for k, v in op.opData["items"].items():
                        all_path = str(Path(root_path, *paths, store_meta.encode_key(k)))
                        if k not in obj:
                            new_items[all_path] = {
                                "value": v
                            }
                        else:
                            if all_path not in ops_with_paths:
                                ops_with_paths[all_path] = []
                            new_op = DraftUpdateOp(JMESPathOpType.DictUpdate, {
                                "items": {
                                    "value": v
                                }
                            }, relative_node.children[0].children[0])
                            ops_with_paths[all_path].append(new_op)
                    continue
            elif op.op == JMESPathOpType.ScalarInplaceOp:
                store_meta = _get_first_store_meta_by_annotype(relative_node.userdata)
                if isinstance(store_meta, DraftStoreMapMeta):
                    all_path = str(Path(root_path, *paths, store_meta.encode_key(op.opData["key"])))
                    if all_path not in ops_with_paths:
                        ops_with_paths[all_path] = []
                    new_opdata = copy.deepcopy(op.opData)
                    new_opdata["key"] = "value"
                    new_op = dataclasses.replace(op, opData=new_opdata, node=relative_node.children[0].children[0])
                    ops_with_paths[all_path].append(new_op)
                    continue
            elif op.op == JMESPathOpType.Delete:
                store_meta = _get_first_store_meta_by_annotype(relative_node.userdata)
                if isinstance(store_meta, DraftStoreMapMeta):
                    for key in op.opData["keys"]:
                        all_path = str(Path(root_path, *paths, store_meta.encode_key(key)))
                        remove_items.add(all_path)
                    continue
            all_path = str(Path(root_path, *paths))
            if all_path not in ops_with_paths:
                ops_with_paths[all_path] = []
            op = dataclasses.replace(op, node=relative_node)
            ops_with_paths[all_path].append(op)
    return ops_with_paths, new_items, remove_items


class DraftFileStorage(Generic[T]):
    def __init__(self, root_path: str, model: T, store: DraftFileStoreBackendBase):
        self._root_path = root_path
        self._store = store
        self._model = model
        assert dataclasses.is_dataclass(model)
        self._has_splitted_store = validate_splitted_model_type(type(model))
        self._mashumaro_decoder: Optional[BasicDecoder] = None
        self._mashumaro_encoder: Optional[BasicEncoder] = None

    def _lazy_get_mashumaro_coder(self):
        if self._mashumaro_decoder is None:
            self._mashumaro_decoder = BasicDecoder(type(self._model))
        if self._mashumaro_encoder is None:
            self._mashumaro_encoder = BasicEncoder(type(self._model))
        return self._mashumaro_decoder, self._mashumaro_encoder

    @staticmethod
    async def write_whole_model(store: DraftStoreBackendBase, model: T, path: str):
        asdict_obj = _StoreAsDict()
        model_dict = asdict_map_trace(model, asdict_obj._asdict_map_trace_factory)
        await store.write(path, model_dict)
        for p in asdict_obj._store_pairs:
            await store.write(str(Path(path, *p[0])), {
                "value": p[1]
            })

    async def _fetch_model_recursive(self, cur_type: type[T], cur_data: Any, parts: list[str]):
        type_hints = get_type_hints(cur_type, include_extras=True)
        for field in dataclasses.fields(cur_type):
            annotype, store_meta = _get_first_store_meta(type_hints[field.name])
            if isinstance(store_meta, DraftStoreMapMeta):
                glob_path = store_meta.get_glob_path()
                storage_key = store_meta.key if store_meta.key else field.name
                glob_path_all = Path(*parts, storage_key, glob_path)
                real_data = await self._store.glob_read(str(glob_path_all))
                real_data = {store_meta.decode_key(Path(k).stem): v["value"] for k, v in real_data.items()}
                cur_data[field.name] = real_data
                value_type = annotype.get_dict_value_anno_type()
                if value_type.is_dataclass_type():
                    for k, vv in real_data.items():
                        await self._fetch_model_recursive(value_type.origin_type, vv, parts + [storage_key, store_meta.encode_key(k)])
                continue 
            if annotype.is_dataclass_type():
                await self._fetch_model_recursive(annotype.origin_type, cur_data[field.name], parts)

    async def fetch_model(self) -> T:
        data = await self._store.read(self._root_path)
        if data is None:
            # not exist, create new
            if self._has_splitted_store:
                await self.write_whole_model(self._store, self._model, self._root_path)
            else:
                await self._store.write(self._root_path, as_dict_no_undefined(self._model))
            return self._model
        if self._has_splitted_store:
            await self._fetch_model_recursive(type(self._model), data, [])
        if dataclasses.is_pydantic_dataclass(type(self._model)):
            self._model: T = type(self._model)(**data) # type: ignore
        else:
            # plain dataclass don't support create from dict, so we use `mashumaro` decoder here. it's fast.
            dec, _ = self._lazy_get_mashumaro_coder()
            self._model: T = dec.decode(data) # type: ignore
        return self._model

    async def update_model(self, ops: list[DraftUpdateOp]):
        if not self._has_splitted_store:
            await self._store.update(self._root_path, ops)
            return
        ops_with_paths, new_items, remove_items = get_splitted_update_model_ops(self._root_path, ops, self._model)
        for path, ops in ops_with_paths.items():
            await self._store.update(path, ops)
        # TODO if we modify and remove same key...
        for path, v in new_items.items():
            await self._store.write(path, v)
        for path in remove_items:
            await self._store.remove(path)