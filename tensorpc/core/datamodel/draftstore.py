import abc
import copy
from collections.abc import Mapping, MutableMapping
import enum
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

from .draft import (DraftASTNode, DraftASTType, DraftBase, DraftUpdateOp, JMESPathOpType, evaluate_draft_ast,
                    apply_draft_update_ops, apply_draft_update_ops_to_json, stabilize_getitem_path_in_op_main_path)

T = TypeVar("T", bound=DataclassType)

class StoreWriteOpType(enum.Enum):
    WRITE = 0
    UPDATE = 1
    REMOVE = 2

@dataclasses.dataclass
class StoreBackendOp:
    path: str
    type: StoreWriteOpType
    data: Any

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

    async def batch_update(self, ops: list[StoreBackendOp]) -> None:
        """Write/Update/Remove in batch, you can override this method to optimize the batch update"""
        for op in ops:
            if op.type == StoreWriteOpType.WRITE:
                await self.write(op.path, op.data)
            elif op.type == StoreWriteOpType.UPDATE:
                await self.update(op.path, op.data)
            elif op.type == StoreWriteOpType.REMOVE:
                await self.remove(op.path)


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


@dataclasses.dataclass(kw_only=True)
class DraftStoreMetaBase:
    # you can specific multiple store backend by this id.
    store_id: Optional[str] = None


@dataclasses.dataclass(kw_only=True)
class DraftStoreMapMeta(DraftStoreMetaBase):
    attr_key: str = ""
    lazy_key_field: Optional[str] = None
    base64_key: bool = True

    @property 
    def path(self):
        return "{}"

    def get_glob_path(self):
        return self.path.replace('{}', '*')

    def encode_key(self, key: str):
        if not self.base64_key:
            return key
        # encode to b64
        return base64.b64encode(key.encode()).decode()

    def decode_key(self, key: str):
        if not self.base64_key:
            return key
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
        self._store_pairs: list[tuple[list[str], Any, Optional[str]]] = []

    def _get_path_parts(self, types: list[tuple[Any, str, bool]]):
        parts: list[str] = []
        for t, k, is_dict in types:
            annotype, store_meta = _get_first_store_meta(t)

            if not is_dict:
                # k is field name or custom name
                if isinstance(store_meta, DraftStoreMapMeta):
                    parts.append(store_meta.attr_key if store_meta.attr_key else k)
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
                storage_key = store_meta.attr_key if store_meta.attr_key else k
                store_id = store_meta.store_id
                for kk, vv in v.items():
                    self._store_pairs.append((parts + [storage_key, store_meta.encode_key(kk)], vv, store_id))
                continue 
            if isinstance(v, UniqueTreeId):
                res[k] = v.uid_encoded
            elif not isinstance(v, (Undefined, BackendOnlyProp)):
                res[k] = v
        return res

def _validate_splitted_model_type(model_type: type[T], type_cache: set, all_store_ids: Optional[set[str]] = None):
    """Check a model have splitted KV store.
    All dict type of a nested path must be splitted, don't support splitted store inside a plain container.
    """
    type_hints = get_type_hints(model_type, include_extras=True)
    has_splitted_store = False
    for field in dataclasses.fields(model_type):
        field_type = type_hints[field.name]
        annotype, store_meta = _get_first_store_meta(field_type)
        if annotype.origin_type not in type_cache:
            type_cache.add(annotype.origin_type)
        else:
            # avoid nested check
            continue 
        if isinstance(store_meta, DraftStoreMapMeta):
            if store_meta.store_id is not None:
                if all_store_ids is not None:
                    assert store_meta.store_id in all_store_ids, f"store id {store_meta.store_id} not exist in {all_store_ids}"
            has_splitted_store = True
            assert annotype.is_dict_type() and annotype.get_dict_key_anno_type().origin_type is str
            value_type = annotype.get_dict_value_anno_type()
            if value_type.is_dataclass_type():
                _validate_splitted_model_type(value_type.origin_type, type_cache, all_store_ids)
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
            has_splitted_store |= _validate_splitted_model_type(annotype.origin_type, type_cache, all_store_ids)
        else:
            # all non-dataclass field type shouldn't contain any store meta
            for t in child_type_generator_with_dataclass(field_type):
                if is_annotated(t):
                    annometa = t.__metadata__
                    for m in annometa:
                        if isinstance(m, DraftStoreMetaBase):
                            raise ValueError(f"subtype of field {field.name} with type {field_type} can't contain any store meta")
    return has_splitted_store

def validate_splitted_model_type(model_type: type[T], all_store_ids: Optional[set[str]] = None):
    type_cache = set()
    return _validate_splitted_model_type(model_type, type_cache, all_store_ids)

@dataclasses.dataclass 
class SplitNewDeleteOp:
    is_new: bool
    key: str
    value: Any = None

def get_splitted_update_model_ops(root_path: str, ops: list[DraftUpdateOp], model_before_update: Any, main_store_id: str):
    ops_with_paths: dict[str, tuple[str, list[Union[DraftUpdateOp, SplitNewDeleteOp]]]] = {}
    new_items: set[str] = set()
    for op in ops:
        node: DraftASTNode = op.node 
        all_dict_getitem_nodes: list[DraftASTNode] = []
        paths: list[str] = []
        store_id = main_store_id
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
                    paths.append(store_meta.encode_key(node.value))
                    if store_meta.store_id is not None:
                        store_id = store_meta.store_id
            elif node.type == DraftASTType.GET_ATTR:
                assert isinstance(node.userdata, AnnotatedType)
                store_meta = _get_first_store_meta_by_annotype(node.userdata)
                if isinstance(store_meta, DraftStoreMapMeta):
                    storage_key = store_meta.attr_key if store_meta.attr_key else node.value
                    paths.append(storage_key)
                    if store_meta.store_id is not None:
                        store_id = store_meta.store_id
        # print(paths)
        if not paths:
            if root_path not in ops_with_paths:
                ops_with_paths[root_path] = (main_store_id, [])
            ops_with_paths[root_path][1].append(op)
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
                    store_id = store_meta.store_id
                    if store_id is None:
                        store_id = main_store_id
                    # we need to query prev model to determine the existance of the key
                    obj = None
                    for k, v in op.opData["items"].items():
                        all_path = str(Path(root_path, *paths, store_meta.encode_key(k)))
                        if all_path not in ops_with_paths:
                            ops_with_paths[all_path] = (store_id, [])
                        if all_path not in new_items:
                            if obj is None:
                                obj = evaluate_draft_ast(op.node, model_before_update)
                                assert isinstance(obj, Mapping)
                            is_new_item = k not in obj
                            new_items.add(all_path)
                        else:
                            is_new_item = False
                        if is_new_item:
                            ops_with_paths[all_path][1].append(SplitNewDeleteOp(True, k, {
                                "value": v
                            }))
                        else:
                            new_op = DraftUpdateOp(JMESPathOpType.DictUpdate, {
                                "items": {
                                    "value": v
                                }
                            }, relative_node.children[0].children[0])
                            ops_with_paths[all_path][1].append(new_op)
                    continue
            elif op.op == JMESPathOpType.ScalarInplaceOp:
                store_meta = _get_first_store_meta_by_annotype(relative_node.userdata)
                if isinstance(store_meta, DraftStoreMapMeta):
                    store_id = store_meta.store_id
                    if store_id is None:
                        store_id = main_store_id

                    all_path = str(Path(root_path, *paths, store_meta.encode_key(op.opData["key"])))
                    if all_path not in ops_with_paths:
                        ops_with_paths[all_path] = (store_id, [])
                    new_opdata = copy.deepcopy(op.opData)
                    new_opdata["key"] = "value"
                    new_op = dataclasses.replace(op, opData=new_opdata, node=relative_node.children[0].children[0])
                    ops_with_paths[all_path][1].append(new_op)
                    continue
            elif op.op == JMESPathOpType.Delete:
                store_meta = _get_first_store_meta_by_annotype(relative_node.userdata)
                if isinstance(store_meta, DraftStoreMapMeta):
                    for key in op.opData["keys"]:
                        all_path = str(Path(root_path, *paths, store_meta.encode_key(key)))
                        if all_path not in ops_with_paths:
                            ops_with_paths[all_path] = (store_id, [])
                        ops_with_paths[all_path][1].append(SplitNewDeleteOp(False, key))
                        if all_path in new_items:
                            new_items.remove(all_path)
                    continue
            all_path = str(Path(root_path, *paths))
            if all_path not in ops_with_paths:
                ops_with_paths[all_path] = (store_id, [])
            op = dataclasses.replace(op, node=relative_node)
            ops_with_paths[all_path][1].append(op)
    return ops_with_paths


class DraftFileStorage(Generic[T]):
    def __init__(self, root_path: str, model: T, store: Union[DraftStoreBackendBase, Mapping[str, DraftStoreBackendBase]], main_store_id: str = ""):
        self._root_path = root_path
        if not isinstance(store, Mapping):
            store = {main_store_id: store}
        self._store = store
        self._model = model
        self._main_store_id = main_store_id
        assert dataclasses.is_dataclass(model)
        all_store_ids = set(self._store.keys())
        self._has_splitted_store = validate_splitted_model_type(type(model), all_store_ids)
        self._mashumaro_decoder: Optional[BasicDecoder] = None
        self._mashumaro_encoder: Optional[BasicEncoder] = None

    def _lazy_get_mashumaro_coder(self):
        if self._mashumaro_decoder is None:
            self._mashumaro_decoder = BasicDecoder(type(self._model))
        if self._mashumaro_encoder is None:
            self._mashumaro_encoder = BasicEncoder(type(self._model))
        return self._mashumaro_decoder, self._mashumaro_encoder

    @staticmethod
    async def write_whole_model(store: Mapping[str, DraftStoreBackendBase], model: T, path: str, main_store_id: str = ""):
        asdict_obj = _StoreAsDict()
        model_dict = asdict_map_trace(model, asdict_obj._asdict_map_trace_factory)
        await store[main_store_id].write(path, model_dict)
        for p in asdict_obj._store_pairs:
            store_id = p[2]
            if store_id is None:
                store_id = main_store_id
            await store[store_id].write(str(Path(path, *p[0])), {
                "value": p[1]
            })

    async def _fetch_model_recursive(self, cur_type: type[T], cur_data: Any, parts: list[str]):
        type_hints = get_type_hints(cur_type, include_extras=True)
        for field in dataclasses.fields(cur_type):
            annotype, store_meta = _get_first_store_meta(type_hints[field.name])
            if isinstance(store_meta, DraftStoreMapMeta):
                glob_path = store_meta.get_glob_path()
                store_id = store_meta.store_id
                if store_id is None:
                    store_id = self._main_store_id
                storage_key = store_meta.attr_key if store_meta.attr_key else field.name
                glob_path_all = Path(*parts, storage_key, glob_path)
                store = self._store[store_id]
                assert isinstance(store, DraftFileStoreBackendBase)
                real_data = await store.glob_read(str(glob_path_all))
                real_data = {store_meta.decode_key(Path(k).stem): v["value"] for k, v in real_data.items()}
                cur_data[field.name] = real_data
                value_type = annotype.get_dict_value_anno_type()
                if value_type.is_dataclass_type():
                    for k, vv in real_data.items():
                        await self._fetch_model_recursive(value_type.origin_type, vv, parts + [storage_key, store_meta.encode_key(k)])
                continue 
            if annotype.is_dataclass_type():
                await self._fetch_model_recursive(annotype.origin_type, cur_data[field.name], parts)

    @property 
    def has_splitted_store(self):
        return self._has_splitted_store

    async def fetch_model(self) -> T:
        data = await self._store[self._main_store_id].read(self._root_path)
        if data is None:
            # not exist, create new
            if self._has_splitted_store:
                await self.write_whole_model(self._store, self._model, self._root_path, main_store_id=self._main_store_id)
            else:
                await self._store[self._main_store_id].write(self._root_path, as_dict_no_undefined(self._model))
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

    async def update_model(self, root_draft: Any, ops: list[DraftUpdateOp]):
        assert isinstance(root_draft, DraftBase)
        # convert dynamic node to static in op
        ops = ops.copy()
        for i in range(len(ops)):
            op = ops[i]
            if op.has_dynamic_node_in_main_path():
                ops[i] = stabilize_getitem_path_in_op_main_path(op, root_draft, self._model)
        ops = [o.to_json_update_op().to_userdata_removed() for o in ops]
        if not self._has_splitted_store:
            await self._store[self._main_store_id].update(self._root_path, ops)
            return
        ops_with_paths = get_splitted_update_model_ops(self._root_path, ops, self._model, self._main_store_id)
        for path, (store_id, ops_mixed) in ops_with_paths.items():
            batch_ops: list[StoreBackendOp] = []
            cur_update_ops: list[DraftUpdateOp] = []
            store = self._store[store_id]
            for op_mixed in ops_mixed:
                if isinstance(op_mixed, DraftUpdateOp):
                    cur_update_ops.append(op_mixed)
                else:
                    if cur_update_ops:
                        batch_ops.append(StoreBackendOp(path, StoreWriteOpType.UPDATE, cur_update_ops))
                        cur_update_ops = []
                    if op_mixed.is_new:
                        batch_ops.append(StoreBackendOp(path, StoreWriteOpType.WRITE, op_mixed.value))
                    else:
                        batch_ops.append(StoreBackendOp(path, StoreWriteOpType.REMOVE, None))
            if cur_update_ops:
                batch_ops.append(StoreBackendOp(path, StoreWriteOpType.UPDATE, cur_update_ops))
            await store.batch_update(batch_ops)

