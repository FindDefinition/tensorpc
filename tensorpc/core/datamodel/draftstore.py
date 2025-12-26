"""Draft Store

Format: 

dict -> dict -> endpoint

endpoint can be dict/list/scalar.

"""

import abc
import copy
from collections.abc import Mapping, MutableMapping
import enum
from pathlib import Path, PurePath, PurePosixPath
import shutil
import time
import traceback
from typing import (Any, Generic, Optional, TypeVar, Union, get_type_hints)

from mashumaro.codecs.basic import BasicDecoder, BasicEncoder
import base64
import dataclasses as dataclasses_plain
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.annolib import (
    AnnotatedType, BackendOnlyProp, DataclassType, Undefined,
    child_type_generator_with_dataclass,
    get_dataclass_field_meta_dict, get_type_hints_with_cache, is_annotated,
    parse_type_may_optional_undefined, resolve_type_hints)
from tensorpc.core.datamodel.asdict import as_dict_no_undefined

from tensorpc.core.datamodel.draftast import ROOT_NODE
from tensorpc.core.tree_id import UniqueTreeId
import json
from .draft import (DraftASTNode, DraftASTType, DraftBase, DraftFieldMeta,
                    DraftUpdateOp, JMESPathOpType,
                    apply_draft_update_ops_to_json_with_root,
                    evaluate_draft_ast, apply_draft_update_ops,
                    apply_draft_update_ops_to_json,
                    stabilize_getitem_path_in_op_main_path)

T = TypeVar("T", bound=DataclassType)


class StoreWriteOpType(enum.Enum):
    WRITE = 0
    UPDATE = 1
    REMOVE = 2
    REMOVE_FOLDER = 3


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

    @abc.abstractmethod
    async def remove_folder(self, path: str) -> Any:
        """remove all item (recursive) in path (folder).
        """

    async def batch_update(self, ops: list[StoreBackendOp]) -> None:
        """Write/Update/Remove in batch, you can override this method to optimize the batch update"""
        for op in ops:
            if op.type == StoreWriteOpType.WRITE:
                await self.write(op.path, op.data)
            elif op.type == StoreWriteOpType.UPDATE:
                await self.update(op.path, op.data)
            elif op.type == StoreWriteOpType.REMOVE:
                await self.remove(op.path)
            elif op.type == StoreWriteOpType.REMOVE_FOLDER:
                await self.remove_folder(op.path)


class DraftFileStoreBackendBase(DraftStoreBackendBase):

    @abc.abstractmethod
    async def read_all_childs(self, path: str) -> dict[str, Any]:
        """Read all child data from path (not recursive). usually used when you use real file system 
        as backend.
        """


class DraftFileStoreBackendInMemory(DraftFileStoreBackendBase):

    def __init__(self):
        self._data: dict[str, Any] = {}

    async def read(self, path: str) -> Optional[Any]:
        return json.loads(json.dumps(self._data.get(path)))

    async def write(self, path: str, data: Any) -> None:
        self._data[path] = json.loads(json.dumps(data))

    async def update(self, path: str, ops: list[DraftUpdateOp]) -> None:
        # for in-memory store, avoid store and frontend share same dict.
        ops = [
            dataclasses.replace(o, opData=json.loads(json.dumps(o.opData)))
            for o in ops
        ]
        data = self._data.get(path)
        if data is None:
            assert ops[
                0].op == JMESPathOpType.RootAssign, f"path {path} not exist, {ops[0]}"
        is_root_changed, root_obj = apply_draft_update_ops_to_json_with_root(
            data, ops)
        if is_root_changed:
            self._data[path] = root_obj

    async def remove(self, path: str) -> None:
        self._data.pop(path, None)

    async def read_all_childs(self, path: str) -> dict[str, Any]:
        res = {}
        path_p = Path(path)
        for k, v in self._data.items():
            k_p = Path(k)
            if k_p.parent == path_p:
                res[k] = json.loads(json.dumps(v))
        return res

    async def remove_folder(self, path: str) -> Any:
        remove_keys = []
        for k in self._data.keys():
            try:
                Path(k).relative_to(path)
            except ValueError:
                continue
            remove_keys.append(k)
        for k in remove_keys:
            self._data.pop(k)
        return remove_keys


class DraftMongoStoreBackend(DraftFileStoreBackendBase):
    """In mongodb backend, we store each path.parent as a collection.
    all document has format {key: key, value: value}
    """

    def __init__(self, db: Any):
        self._db = db
        self._root_key = "$"

    def _get_coll_and_key(self, path: str):
        path_p = PurePosixPath(path)
        if len(path_p.parts) == 1:
            coll = self._db[self._root_key]
        else:
            coll = self._db[str(path_p.parent)]
        key = path_p.name
        return coll, key

    async def read(self, path: str) -> Optional[Any]:
        coll, key = self._get_coll_and_key(path)
        res = coll.find_one({"key": key})
        if res is None:
            return None
        return res["value"]

    async def write(self, path: str, data: Any) -> None:
        coll, key = self._get_coll_and_key(path)
        coll.replace_one({"key": key}, {
            "key": key,
            "value": data
        },
                         upsert=True)

    async def update(self, path: str, ops: list[DraftUpdateOp]) -> None:
        # TODO better update for mongo
        coll, key = self._get_coll_and_key(path)
        data_doc = coll.find_one({"key": key})
        if data_doc is None:
            raise ValueError(f"path {path} not exist")
        data = data_doc["value"]
        is_root_changed, root_obj = apply_draft_update_ops_to_json_with_root(
            data, ops)
        if is_root_changed:
            coll.replace_one({"key": key}, {"key": key, "value": root_obj})
        else:
            coll.update_one({"key": key}, {"$set": {"value": data}})

    async def remove(self, path: str) -> None:
        coll, key = self._get_coll_and_key(path)
        coll.delete_one({"key": key})

    async def read_all_childs(self, path: str) -> dict[str, Any]:
        path_p = PurePosixPath(path)
        coll = self._db[str(path_p)]
        res = {}
        for post in coll.find():
            res[post["key"]] = post["value"]
        return res

    async def remove_folder(self, path: str) -> Any:
        # use startswith to remove all child
        path_p = PurePosixPath(path)
        coll = self._db[str(path_p)]
        coll.delete_many({"key": {"$regex": f"^{path_p.name}/"}})

class DraftSimpleFileStoreBackend(DraftFileStoreBackendBase):
    def __init__(self, root: Path, with_bak: bool = False, verbose_fs: bool = False, read_only: bool = False):
        self._root = root
        self._with_bak = with_bak
        self._verbose_fs = verbose_fs
        self._read_only = read_only
    
    def _get_abs_path(self, path: str, with_bak: bool = False) -> Path:
        if with_bak:
            path_p = self._root / Path(path + ".json.bak")
        else:
            path_p = self._root / Path(path + ".json")
        return path_p

    async def read(self, path: str) -> Optional[Any]:
        t = time.time()
        path_p = self._get_abs_path(path)
        path_bak_p = self._get_abs_path(path, with_bak=True)
        if not path_p.exists():
            # try bak
            path_p = path_bak_p
            if not path_p.exists():
                return None 
        try:
            with open(path_p, "r") as f:
                res = json.load(f)
            if self._verbose_fs:
                print(f"[DraftStore]Read File {path_p} cost time {time.time() - t}")
            return res 
        except:
            traceback.print_exc()
            # try bak
            try:
                with open(path_bak_p, "r") as f:
                    return json.load(f)
            except:
                return None 

    async def write(self, path: str, data: Any) -> None:
        if self._read_only:
            return 
        t = time.time()
        path_p = self._get_abs_path(path)
        path_p_parent = path_p.parent 
        if not path_p_parent.exists():
            path_p_parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        with open(path_p, "w") as f:
            json.dump(data, f)
        if self._with_bak:
            path_bak_p = self._get_abs_path(path, with_bak=True)
            with open(path_bak_p, "w") as f:
                json.dump(data, f)
        if self._verbose_fs:
            print(f"[DraftStore]Write File {path_p} cost time {time.time() - t}")

    async def update(self, path: str, ops: list[DraftUpdateOp]) -> None:
        # for in-memory store, avoid store and frontend share same dict.
        if self._read_only:
            return 
        ops = [
            dataclasses.replace(o, opData=json.loads(json.dumps(o.opData)))
            for o in ops
        ]
        data = await self.read(path)
        if data is None:
            assert ops[
                0].op == JMESPathOpType.RootAssign, f"path {path} not exist, {ops[0]}"
        is_root_changed, root_obj = apply_draft_update_ops_to_json_with_root(
            data, ops)
        if is_root_changed:
            await self.write(path, root_obj)
        else:
            await self.write(path, data)

    async def remove(self, path: str) -> None:
        if self._read_only:
            return 
        path_p = self._get_abs_path(path)
        path_p.unlink()
        if self._with_bak:
            path_bak_p = self._get_abs_path(path, with_bak=True)
            if path_bak_p.exists():
                path_bak_p.unlink()


    async def read_all_childs(self, path: str) -> dict[str, Any]:
        res = {}
        for p in self._root.glob(str(Path(path) / "*.json")):
            relative_path = p.relative_to(self._root)
            relative_path_no_suffix = relative_path.with_suffix("")
            with p.open("r") as f:
                res[str(relative_path_no_suffix)] = json.load(f)
        return res

    async def remove_folder(self, path: str) -> Any:
        if self._read_only:
            return 
        path_p = Path(path)
        if not path_p.exists():
            return False
        shutil.rmtree(path_p)
        return True


def _is_none_or_undefined(obj: Any):
    return obj is None or isinstance(obj, Undefined)


@dataclasses.dataclass(kw_only=True)
class DraftStoreMetaBase(DraftFieldMeta):
    # you can specific multiple store backend by this id.
    store_id: Optional[str] = None
    attr_key: str = ""


@dataclasses.dataclass(kw_only=True)
class DraftStoreScalarMeta(DraftStoreMetaBase):
    pass


@dataclasses.dataclass(kw_only=True)
class DraftStoreMapMeta(DraftStoreMetaBase):
    lazy_key_field: Optional[str] = None
    base64_key: bool = True

    def encode_key(self, key: str):
        if not self.base64_key:
            return key
        # encode to b64
        return base64.b64encode(key.encode()).decode()

    def decode_key(self, key: str):
        if not self.base64_key:
            return key
        return base64.b64decode(key.encode()).decode()


def _asdict_map_trace_inner(obj,
                            field_types: list,
                            exclude_field_ids,
                            dict_factory,
                            obj_factory=None,
                            cur_field_type=None):
    if dataclasses.is_dataclass(obj):
        result = []
        type_hints = get_type_hints_with_cache(type(obj), include_extras=True)
        for f in dataclasses.fields(obj):
            if id(f) in exclude_field_ids:
                continue
            field_types_field = field_types + [
                (type_hints[f.name], f.name, False)
            ]
            v = getattr(obj, f.name)
            value = _asdict_map_trace_inner(
                v, field_types_field, exclude_field_ids, dict_factory,
                obj_factory,
                type_hints[f.name] if isinstance(v, dict) else None)
            result.append((f.name, value, field_types_field))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(*[
            _asdict_map_trace_inner(v, field_types, exclude_field_ids,
                                    dict_factory, obj_factory) for v in obj
        ])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_asdict_map_trace_inner(
            v, field_types, exclude_field_ids, dict_factory, obj_factory)
                         for v in obj)
    elif isinstance(obj, dict):
        res = []
        for k, v in obj.items():
            field_types_field = field_types + [(cur_field_type, k, True)]
            kk = _asdict_map_trace_inner(k, field_types, exclude_field_ids,
                                         dict_factory, obj_factory)
            vv = _asdict_map_trace_inner(v, field_types_field,
                                         exclude_field_ids, dict_factory,
                                         obj_factory)
            res.append((kk, vv))
        return type(obj)(res)
    else:
        if obj_factory is not None:
            obj = obj_factory(obj)
        # no deepcopy here since we don't modify the object
        return obj


def _default_asdict_map_trace_factory(obj: list[tuple[str, Any, Any]]):
    return {k: v for k, v, _ in obj}


def asdict_map_trace(obj,
                     exclude_field_ids: set[int],
                     dict_factory=_default_asdict_map_trace_factory,
                     cur_field_type: Optional[Any] = None):
    return _asdict_map_trace_inner(obj, [],
                                   exclude_field_ids,
                                   dict_factory,
                                   cur_field_type=cur_field_type)


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
                if isinstance(store_meta, DraftStoreMetaBase):
                    parts.append(
                        store_meta.attr_key if store_meta.attr_key else k)
                else:
                    parts.append(k)
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
            if isinstance(v, Undefined):
                continue
            if v is not None:
                if isinstance(store_meta, DraftStoreMapMeta):
                    assert isinstance(
                        v, Mapping) and annotype.get_dict_key_anno_type(
                        ).origin_type is str
                    storage_key = store_meta.attr_key if store_meta.attr_key else k
                    store_id = store_meta.store_id
                    for kk, vv in v.items():
                        self._store_pairs.append(
                            (parts + [storage_key,
                                      store_meta.encode_key(kk)], vv,
                             store_id))
                    res[k] = {}
                    continue
                elif isinstance(store_meta, DraftStoreScalarMeta):
                    store_id = store_meta.store_id
                    storage_key = store_meta.attr_key if store_meta.attr_key else k
                    self._store_pairs.append(
                        (parts + [storage_key], v, store_id))
                    continue
            if isinstance(v, UniqueTreeId):
                res[k] = v.uid_encoded
            elif not isinstance(v, (Undefined, BackendOnlyProp)):
                res[k] = v
        return res


@dataclasses_plain.dataclass
class FieldMeta:
    name: str
    annotype: AnnotatedType
    draft_store_meta: Optional[DraftStoreMetaBase] = None
    contain_nested_map: bool = False
    contain_nested_store: bool = False


def _analysis_model_store_meta(root_annotype: AnnotatedType,
                                exclude_field_ids: set[int],
                               field_meta_dict: dict[int, FieldMeta],
                               parent_metas: list[FieldMeta],
                               all_store_ids: Optional[set[str]] = None):
    """Check a model have splitted KV store.
    All dict type of a nested path must be splitted, don't support splitted store inside a plain container.
    """
    # use resolve_type_hints to resolve generic dataclass
    if root_annotype.child_types:
        # generic dataclass
        type_hints = resolve_type_hints(root_annotype.origin_type[tuple(root_annotype.child_types)])
    else:
        type_hints = resolve_type_hints(root_annotype.origin_type)
    has_splitted_store = False
    for field in dataclasses.fields(root_annotype.origin_type):
        if id(field) in exclude_field_ids:
            continue
        field_type = type_hints[field.name]
        annotype, store_meta = _get_first_store_meta(field_type)
        field_meta = FieldMeta(field.name, annotype)
        if id(field) not in field_meta_dict:
            field_meta_dict[id(field)] = field_meta
        else:
            prev_meta = field_meta_dict[id(field)]
            prev_meta.contain_nested_map = isinstance(store_meta,
                                                      DraftStoreMapMeta)
            prev_meta.contain_nested_store = isinstance(
                store_meta, DraftStoreMetaBase)
            # avoid nested check
            continue
        if isinstance(store_meta, (DraftStoreMapMeta, DraftStoreScalarMeta)):
            for parent_meta in parent_metas:
                # if parent_meta.draft_store_meta is not None:
                parent_meta.contain_nested_map = isinstance(
                    store_meta, DraftStoreMapMeta)
                parent_meta.contain_nested_store = True
            if store_meta.store_id is not None:
                if all_store_ids is not None:
                    assert store_meta.store_id in all_store_ids, f"store id {store_meta.store_id} not exist in {all_store_ids}"
            has_splitted_store = True
            if isinstance(store_meta, DraftStoreMapMeta):
                assert annotype.is_dict_type(
                ) and annotype.get_dict_key_anno_type().origin_type is str
                value_type = annotype.get_dict_value_anno_type()
                if value_type.is_dataclass_type():
                    _analysis_model_store_meta(value_type,
                                                  exclude_field_ids,
                                               field_meta_dict,
                                               parent_metas + [field_meta],
                                               all_store_ids)
                else:
                    # all non-dataclass field type shouldn't contain any store meta
                    for child_type in annotype.child_types:
                        for t in child_type_generator_with_dataclass(
                                child_type):
                            if is_annotated(t):
                                annometa = t.__metadata__
                                for m in annometa:
                                    if isinstance(m, DraftStoreMetaBase):
                                        raise ValueError(
                                            f"subtype of field {field.name} with type {child_type} can't contain any store meta"
                                        )
                field_meta.draft_store_meta = store_meta
            elif isinstance(store_meta, DraftStoreScalarMeta):
                for t in child_type_generator_with_dataclass(
                        annotype.origin_type):
                    if is_annotated(t):
                        annometa = t.__metadata__
                        for m in annometa:
                            if isinstance(m, DraftStoreMetaBase):
                                raise ValueError(
                                    f"subtype of field {field.name} with type {annotype.origin_type} can't contain any store meta"
                                )
                field_meta.draft_store_meta = store_meta
            continue
        if annotype.is_dataclass_type():
            has_splitted_store |= _analysis_model_store_meta(
                annotype, exclude_field_ids, field_meta_dict,
                parent_metas + [field_meta], all_store_ids)
        else:
            # all non-dataclass field type shouldn't contain any store meta
            for t in child_type_generator_with_dataclass(field_type):
                if is_annotated(t):
                    annometa = t.__metadata__
                    for m in annometa:
                        if isinstance(m, DraftStoreMetaBase):
                            raise ValueError(
                                f"subtype of field {field.name} with type {field_type} can't contain any store meta"
                            )
    return has_splitted_store


def analysis_model_store_meta(model_type: type[T],
                                exclude_field_ids: set[int],
                              all_store_ids: Optional[set[str]] = None):
    field_meta_dict: dict[int, FieldMeta] = {}
    res = _analysis_model_store_meta(parse_type_may_optional_undefined(model_type), exclude_field_ids, field_meta_dict, [],
                                     all_store_ids)
    new_field_meta_dict: dict[Optional[int], FieldMeta] = {**field_meta_dict}
    return res, new_field_meta_dict


@dataclasses.dataclass
class SplitNewDeleteOp:
    is_new: bool
    key: str
    value: Any = None
    is_remove_all_childs: bool = False


def _get_relative_store_node_and_path(op: DraftUpdateOp,
                                      field_meta_dict: dict[Optional[int],
                                                            FieldMeta]):
    node = op.node
    relative_node: DraftASTNode = node.clone_tree_only()
    relative_node_cur = relative_node
    relative_node_last = None
    res_relative_node = None
    path: list[str] = []
    # determine is map modificatoin, when target is dict, it is always a modify operation (include scalar inplace on dict value)
    field_meta = field_meta_dict.get(node.field_id)
    is_map_mod = False
    store_id: Optional[str] = None

    if field_meta is not None and field_meta.draft_store_meta and isinstance(
            field_meta.draft_store_meta, DraftStoreMapMeta):
        assert node.type == DraftASTType.GET_ATTR, f"node type should be GET_ATTR, but got {node.type.name}"
        is_map_mod = True
        storage_key = field_meta.draft_store_meta.attr_key if field_meta.draft_store_meta.attr_key else node.value
        assert storage_key != ""
        path.append(storage_key)
        store_id = field_meta.draft_store_meta.store_id
    # determine is direct modify op, include setattr and inplace op (target is object, not attr)
    # for delete/container clear, their target is attr, so they don't need to be handled here
    is_direct_mod = False
    if (op.op == JMESPathOpType.SetAttr or op.op
            == JMESPathOpType.ScalarInplaceOp) and op.field_id is not None:
        field_meta = field_meta_dict.get(op.field_id)
        if field_meta is not None and field_meta.draft_store_meta is not None:
            if op.op == JMESPathOpType.SetAttr:
                key = op.opData["items"][0][0]
            else:
                key = op.opData["key"]
            storage_key = field_meta.draft_store_meta.attr_key if field_meta.draft_store_meta.attr_key else key
            path.append(storage_key)
            assert storage_key != ""
            is_direct_mod = True
            store_id = field_meta.draft_store_meta.store_id

    while relative_node_cur.children:
        next_node = relative_node_cur.children[0]
        field_meta = field_meta_dict.get(next_node.field_id)
        if field_meta is not None:
            if isinstance(field_meta.draft_store_meta, DraftStoreMapMeta):
                assert relative_node_cur.type == DraftASTType.DICT_GET_ITEM, f"relative node type should be DICT_GET_ITEM, but got {relative_node_cur.type.name}|{next_node.type.name}"
                storage_key = field_meta.draft_store_meta.attr_key if field_meta.draft_store_meta.attr_key else next_node.value
                path.extend([
                    field_meta.draft_store_meta.encode_key(
                        relative_node_cur.value), storage_key
                ])
                if store_id is None and field_meta.draft_store_meta.store_id is not None:
                    store_id = field_meta.draft_store_meta.store_id
            elif isinstance(field_meta.draft_store_meta, DraftStoreScalarMeta):
                storage_key = field_meta.draft_store_meta.attr_key if field_meta.draft_store_meta.attr_key else next_node.value
                path.append(storage_key)
                if store_id is None and field_meta.draft_store_meta.store_id is not None:
                    store_id = field_meta.draft_store_meta.store_id
            else:
                path.append(field_meta.name)
            if isinstance(field_meta.draft_store_meta,
                          (DraftStoreMapMeta, DraftStoreScalarMeta)):
                if res_relative_node is None:
                    if relative_node_last is not None:
                        relative_node_last.children = [ROOT_NODE]
                        res_relative_node = relative_node
                    else:
                        relative_node = ROOT_NODE
                        res_relative_node = relative_node
        relative_node_last = relative_node_cur
        relative_node_cur = relative_node_cur.children[0]
    if res_relative_node is None:
        res_relative_node = relative_node

    return relative_node, path[::-1], is_map_mod, is_direct_mod, store_id


def _get_op_modify_dict_with_store(op: DraftUpdateOp, is_map_mod: bool,
                                   field_meta_dict: dict[Optional[int],
                                                         FieldMeta]):
    update_or_setattr_on_dict = False
    contains_nested_store: bool = False
    res_value = None
    res_store_meta: Optional[DraftStoreMapMeta] = None
    if op.op == JMESPathOpType.SetAttr:
        field_id = op.field_id
        if field_id in field_meta_dict:
            field_meta = field_meta_dict[field_id]
            contains_nested_store = field_meta.contain_nested_store
            update_or_setattr_on_dict = isinstance(field_meta.draft_store_meta,
                                                   DraftStoreMapMeta)
            if isinstance(field_meta.draft_store_meta, DraftStoreMapMeta):
                res_store_meta = field_meta.draft_store_meta
            res_value = op.opData["items"][0][1]
    elif op.op == JMESPathOpType.DictUpdate and is_map_mod:
        field_id = op.node.field_id
        assert field_id in field_meta_dict
        field_meta = field_meta_dict[field_id]
        contains_nested_store = field_meta.contain_nested_store

        assert isinstance(field_meta.draft_store_meta, DraftStoreMapMeta)
        res_store_meta = field_meta.draft_store_meta
        update_or_setattr_on_dict = True
        res_value = op.opData["items"]
    return update_or_setattr_on_dict, res_value, res_store_meta, contains_nested_store


def get_splitted_update_model_ops(root_path: str, ops: list[DraftUpdateOp],
                                  main_store_id: str,
                                  field_meta_dict: dict[Optional[int],
                                                        FieldMeta],
                                  exclude_field_ids: Optional[set[int]] = None):
    ops_with_paths: dict[str, tuple[str, list[Union[DraftUpdateOp,
                                                    SplitNewDeleteOp]]]] = {}
    ops_with_paths_batch: list[Union[dict[str, tuple[str, list[Union[
        DraftUpdateOp, SplitNewDeleteOp]]]],
                                     SplitNewDeleteOp]] = [ops_with_paths]
    if exclude_field_ids is None:
        exclude_field_ids = set()
    for op in ops:
        node: DraftASTNode = op.node
        paths: list[str] = []
        # convert absolute path (node) to relative path
        relative_node, paths, is_map_mod, is_direct_mod, store_id = _get_relative_store_node_and_path(
            op, field_meta_dict)
        if store_id is None:
            store_id = main_store_id
        # paths definition:
        # paths contains all parent map store key and field key if its a store
        # -------- Step I: handle nested store (include non-nested map store) ----------
        # handle DictUpdate or SetAttr on a dict with map store
        update_or_setattr_on_dict, map_items, map_store_meta, map_contains_nested = _get_op_modify_dict_with_store(
            op, is_map_mod, field_meta_dict)
        if update_or_setattr_on_dict:
            assert map_store_meta is not None
            if op.op == JMESPathOpType.SetAttr:
                # dict setattr must clear folder.
                all_path = str(Path(root_path, *paths))
                ops_with_paths = {}
                if not ops_with_paths_batch[-1]:
                    ops_with_paths_batch.pop()
                ops_with_paths_batch.append(
                    SplitNewDeleteOp(False,
                                     all_path,
                                     value=store_id,
                                     is_remove_all_childs=True))
                ops_with_paths_batch.append(ops_with_paths)
            if map_items is not None:
                for k, v in map_items.items():
                    all_path = str(
                        Path(root_path, *paths, map_store_meta.encode_key(k)))
                    if all_path not in ops_with_paths:
                        ops_with_paths[all_path] = (store_id, [])
                    if map_contains_nested:
                        asdict_obj = _StoreAsDict()
                        model_dict = asdict_map_trace(
                            v, exclude_field_ids,
                            asdict_obj._asdict_map_trace_factory)
                        for p in asdict_obj._store_pairs:
                            all_path_nested = str(Path(all_path, *p[0]))
                            store_id_nested = p[2]
                            if store_id_nested is None:
                                store_id_nested = store_id
                            if all_path_nested not in ops_with_paths:
                                ops_with_paths[all_path_nested] = (
                                    store_id_nested, [])
                            new_op_item = DraftUpdateOp(
                                JMESPathOpType.RootAssign, p[1], ROOT_NODE)
                            ops_with_paths[all_path_nested][1].append(
                                new_op_item)
                        new_op = DraftUpdateOp(JMESPathOpType.RootAssign,
                                               model_dict, ROOT_NODE)
                    else:
                        new_op = DraftUpdateOp(JMESPathOpType.RootAssign, v,
                                               ROOT_NODE)
                    ops_with_paths[all_path][1].append(new_op)
            if op.op == JMESPathOpType.SetAttr:
                # for dict setattr, we need to remove last path item
                paths.pop()
                k, target = op.opData["items"][0]
                if not _is_none_or_undefined(target):
                    # use empty dict for map store
                    op = dataclasses.replace(op, opData={"items": [(k, {})]})
        # handle SetAttr that set a object contains a nested store or on non-map store
        if op.op == JMESPathOpType.SetAttr:
            field_id = op.field_id
            if field_id in field_meta_dict:
                field_meta = field_meta_dict[field_id]
                if not isinstance(field_meta.draft_store_meta,
                                  DraftStoreMapMeta):
                    # print(field_meta.contain_nested_store, field_meta.annotype)
                    if field_meta.contain_nested_store:
                        # we already handle dict setattr above.
                        # since we don't allow nested store except map store, so draft_store_meta must be None.
                        assert not isinstance(field_meta.draft_store_meta,
                                              DraftStoreMetaBase)
                        k = op.opData["items"][0][0]
                        v = op.opData["items"][0][1]
                        all_path = str(Path(root_path, *paths))
                        all_path_for_clear = str(
                            Path(root_path, *paths, field_meta.name))
                        ops_with_paths = {}
                        if not ops_with_paths_batch[-1]:
                            ops_with_paths_batch.pop()
                        ops_with_paths_batch.append(
                            SplitNewDeleteOp(False,
                                             all_path_for_clear,
                                             value=store_id,
                                             is_remove_all_childs=True))
                        ops_with_paths_batch.append(ops_with_paths)
                        asdict_obj = _StoreAsDict()
                        model_dict = asdict_map_trace(
                            v,
                            exclude_field_ids,
                            asdict_obj._asdict_map_trace_factory,
                            cur_field_type=field_meta.annotype.raw_type)
                        for p in asdict_obj._store_pairs:
                            all_path_nested = str(Path(all_path, *p[0]))
                            store_id_nested = p[2]
                            if store_id_nested is None:
                                store_id_nested = store_id
                            if all_path_nested not in ops_with_paths:
                                ops_with_paths[all_path_nested] = (
                                    store_id_nested, [])
                            new_op_item = DraftUpdateOp(
                                JMESPathOpType.RootAssign, p[1], ROOT_NODE)
                            ops_with_paths[all_path_nested][1].append(
                                new_op_item)
                        op = dataclasses.replace(
                            op, opData={"items": [(k, model_dict)]})
        # after Step I, we only add nested store ops and replace data in original op with data without nested store data.
        # the real assign op of original op isn't appended.

        # -------- Step II: handle op itself ----------
        # in this step, we won't handle nested store. for dict map store,

        if not paths:
            if root_path not in ops_with_paths:
                ops_with_paths[root_path] = (main_store_id, [])
            ops_with_paths[root_path][1].append(op)
        else:
            if is_map_mod:
                field_meta = field_meta_dict.get(node.field_id)
                assert field_meta is not None and field_meta.draft_store_meta is not None
                store_meta = field_meta.draft_store_meta
                assert isinstance(store_meta, DraftStoreMapMeta)
                store_id = store_meta.store_id
                if store_id is None:
                    store_id = main_store_id
                if op.op == JMESPathOpType.DictUpdate:
                    # nested store is handled above.
                    continue
                elif op.op == JMESPathOpType.ScalarInplaceOp:
                    all_path = str(
                        Path(root_path, *paths,
                             store_meta.encode_key(op.opData["key"])))
                    if all_path not in ops_with_paths:
                        ops_with_paths[all_path] = (store_id, [])
                    new_op = dataclasses.replace(
                        op, op=JMESPathOpType.RootInplaceOp, node=ROOT_NODE)
                    ops_with_paths[all_path][1].append(new_op)
                    continue
                elif op.op == JMESPathOpType.Delete:
                    for key in op.opData["keys"]:
                        all_path = str(
                            Path(root_path, *paths,
                                 store_meta.encode_key(key)))
                        if all_path not in ops_with_paths:
                            ops_with_paths[all_path] = (store_id, [])
                        ops_with_paths[all_path][1].append(
                            SplitNewDeleteOp(False, key))
                    continue
                elif op.op == JMESPathOpType.ContainerClear:
                    all_path = str(Path(root_path, *paths))
                    ops_with_paths = {}
                    ops_with_paths_batch.append(
                        SplitNewDeleteOp(False,
                                         all_path,
                                         value=store_id,
                                         is_remove_all_childs=True))
                    ops_with_paths_batch.append(ops_with_paths)
                    continue
                else:
                    raise NotImplementedError
            elif is_direct_mod:
                # setattr op will contains field id of set field.
                field_meta = field_meta_dict.get(op.field_id)
                assert field_meta is not None and field_meta.draft_store_meta is not None
                store_meta = field_meta.draft_store_meta

                if op.op == JMESPathOpType.SetAttr:
                    if isinstance(store_meta, DraftStoreScalarMeta):
                        # k = op.opData["items"][0][0]
                        # storage_key = store_meta.attr_key if store_meta.attr_key else k
                        all_path = str(Path(root_path, *paths))
                        if all_path not in ops_with_paths:
                            ops_with_paths[all_path] = (store_id, [])
                        new_value = op.opData["items"][0][1]
                        new_op = dataclasses.replace(
                            op,
                            op=JMESPathOpType.RootAssign,
                            opData=new_value,
                            node=ROOT_NODE)
                        ops_with_paths[all_path][1].append(new_op)
                        continue
                    # no need to handle map store.
                elif op.op == JMESPathOpType.ScalarInplaceOp:
                    assert isinstance(store_meta, DraftStoreScalarMeta)
                    all_path = str(Path(root_path, *paths))
                    if all_path not in ops_with_paths:
                        ops_with_paths[all_path] = (store_id, [])
                    new_op = dataclasses.replace(
                        op,
                        op=JMESPathOpType.RootInplaceOp,
                        opData=op.opData,
                        node=ROOT_NODE)
                    ops_with_paths[all_path][1].append(new_op)
                    continue
            all_path = str(Path(root_path, *paths))
            if all_path not in ops_with_paths:
                ops_with_paths[all_path] = (store_id, [])
            op = dataclasses.replace(op, node=relative_node)
            ops_with_paths[all_path][1].append(op)
    return ops_with_paths_batch


class DraftFileStorage(Generic[T]):
    """A draft storage for a dataclass model.
    It support splitted storage for different fields in different backend.
    It also support splitted storage for map field, each key-value pair in the map will be stored in different file.
    Args:
        root_path: The root path of the draft storage.
        model: The dataclass model instance.
        store: The draft storage backend or a dict of backend with store id as key.
        main_store_id: The main store id, default is "".
        batch_write_duration: The duration (in seconds) to batch write operations, default is -1 (no batch).
        read_only: If True, the storage is read-only and write operations will be ignored.
            usually used in distributed app that only master node can write draft storage.
    """
    def __init__(self,
                 root_path: str,
                 model: T,
                 store: Union[DraftStoreBackendBase,
                              Mapping[str, DraftStoreBackendBase]],
                 main_store_id: str = "",
                 batch_write_duration: int = -1,
                 read_only: bool = False):
        self._root_path = root_path
        if not isinstance(store, Mapping):
            store = {main_store_id: store}
        self._store = store
        self._model = model
        self._main_store_id = main_store_id
        self._read_only = read_only
        assert dataclasses.is_dataclass(model)
        all_store_ids = set(self._store.keys())
        self._mashumaro_decoder: Optional[BasicDecoder] = None
        self._mashumaro_encoder: Optional[BasicEncoder] = None
        self._exclude_field_ids: set[int] = set()
        model_type_real = type(model)
        self._batch_write_duration = batch_write_duration
        if dataclasses.is_dataclass(model_type_real):
            field_meta_dict = get_dataclass_field_meta_dict(model_type_real)
            for k, v in field_meta_dict.items():
                if v.annotype.annometa is not None:
                    for tt in v.annotype.annometa:
                        if isinstance(tt, DraftFieldMeta):
                            if tt.is_external or tt.is_store_external:
                                if v.field.default is dataclasses.MISSING and v.field.default_factory is dataclasses.MISSING:
                                    raise ValueError(f"external field {v.field.name} must have default value or factory"
                                        " because this field is managed by user, it won't be stored to draft storage.")
                                self._exclude_field_ids.add(v.field_id)
                            break
        self._has_splitted_store, self._field_store_meta = analysis_model_store_meta(
            type(model), self._exclude_field_ids, all_store_ids)

    def _lazy_get_mashumaro_coder(self):
        if self._mashumaro_decoder is None:
            self._mashumaro_decoder = BasicDecoder(type(self._model))
        if self._mashumaro_encoder is None:
            self._mashumaro_encoder = BasicEncoder(type(self._model))
        return self._mashumaro_decoder, self._mashumaro_encoder

    @staticmethod
    async def _write_whole_model(store: Mapping[str, DraftStoreBackendBase],
                                model: T,
                                exclude_field_ids: set[int],
                                path: str,
                                main_store_id: str = ""):
        asdict_obj = _StoreAsDict()
        model_dict = asdict_map_trace(model, exclude_field_ids,
                                      asdict_obj._asdict_map_trace_factory)
        await store[main_store_id].write(path, model_dict)
        for p in asdict_obj._store_pairs:
            store_id = p[2]
            if store_id is None:
                store_id = main_store_id
            await store[store_id].write(str(Path(path, *p[0])), p[1])

    async def _fetch_model_recursive(self, cur_type: type[T], cur_data: Any,
                                     parts: list[str]):
        for field in dataclasses.fields(cur_type):
            field_meta = self._field_store_meta.get(id(field))
            if field_meta is None:
                continue
            annotype = field_meta.annotype
            store_meta = field_meta.draft_store_meta
            if isinstance(store_meta, DraftStoreMapMeta):
                store_id = store_meta.store_id
                if store_id is None:
                    store_id = self._main_store_id
                storage_key = store_meta.attr_key if store_meta.attr_key else field.name
                glob_path_all = Path(*parts, storage_key)
                store = self._store[store_id]
                # TODO better option
                assert isinstance(store, DraftFileStoreBackendBase)
                if not _is_none_or_undefined(cur_data[field.name]):
                    real_data = await store.read_all_childs(str(glob_path_all))
                    # real_data = {store_meta.decode_key(Path(k).stem): v["value"] for k, v in real_data.items()}
                    real_data = {
                        store_meta.decode_key(Path(k).stem): v
                        for k, v in real_data.items()
                    }
                    cur_data[field.name] = real_data
                    value_type = annotype.get_dict_value_anno_type()
                    if value_type.is_dataclass_type():
                        for k, vv in real_data.items():
                            await self._fetch_model_recursive(
                                value_type.origin_type, vv, parts +
                                [storage_key,
                                 store_meta.encode_key(k)])
                continue
            elif isinstance(store_meta, DraftStoreScalarMeta):
                store_id = store_meta.store_id
                if store_id is None:
                    store_id = self._main_store_id
                storage_key = store_meta.attr_key if store_meta.attr_key else field.name
                glob_path_all = Path(*parts, storage_key)
                store = self._store[store_id]
                real_data = await store.read(str(glob_path_all))
                cur_data[field.name] = real_data
                continue
            if annotype.is_dataclass_type():
                cur_data_next = cur_data[field.name]
                if not annotype.is_optional and not annotype.is_undefined:
                    assert isinstance(cur_data_next, dict)
                if isinstance(cur_data_next, dict):
                    await self._fetch_model_recursive(annotype.origin_type,
                                                      cur_data_next, parts)

    @property
    def has_splitted_store(self):
        return self._has_splitted_store

    async def write_whole_model(self, new_model: T):
        if self._read_only:
            return
        assert type(new_model) == type(self._model)
        self._model = new_model
        await self._write_whole_model(self._store, new_model, self._exclude_field_ids,
                                      self._root_path,
                                      main_store_id=self._main_store_id)

    async def fetch_model(self) -> T:
        data = await self._store[self._main_store_id].read(self._root_path)
        if data is None:
            if not self._read_only:
                # not exist, create new
                if self._has_splitted_store:
                    await self._write_whole_model(self._store,
                                                self._model,
                                                self._exclude_field_ids,
                                                self._root_path,
                                                main_store_id=self._main_store_id)
                else:
                    await self._store[self._main_store_id
                                    ].write(self._root_path,
                                            as_dict_no_undefined(self._model))
            return self._model
        if self._has_splitted_store:
            await self._fetch_model_recursive(type(self._model), data,
                                              [self._root_path])
        if dataclasses.is_pydantic_dataclass(type(self._model)):
            self._model: T = type(self._model)(**data)  # type: ignore
        else:
            # plain dataclass don't support create from dict, so we use `mashumaro` decoder here. it's fast.
            dec, _ = self._lazy_get_mashumaro_coder()
            self._model: T = dec.decode(data)  # type: ignore
        # write whole model to clean unused (ignored, external) fields
        if not self._read_only:
            await self._write_whole_model(self._store,
                                        self._model,
                                        self._exclude_field_ids,
                                        self._root_path,
                                        main_store_id=self._main_store_id)
        return self._model

    async def update_model(self, root_draft: Any, ops: list[DraftUpdateOp]):
        if not ops:
            return 
        if self._read_only:
            return 
        assert isinstance(root_draft, DraftBase)
        # convert dynamic node to static in op
        ops = ops.copy()
        for i in range(len(ops)):
            op = ops[i]
            if op.has_dynamic_node_in_main_path():
                ops[i] = stabilize_getitem_path_in_op_main_path(
                    op, root_draft, self._model)
        # remove all op with external (exclude) modify target
        ops = list(filter(lambda o: not o.is_external and not o.is_store_external, ops))
        if not self._has_splitted_store:
            ops = [o.to_json_update_op().to_userdata_removed() for o in ops]
            await self._store[self._main_store_id].update(self._root_path, ops)
            return
        ops_with_paths_batch = get_splitted_update_model_ops(
            self._root_path, ops, self._main_store_id, self._field_store_meta,
            self._exclude_field_ids)
        for ops_with_paths in ops_with_paths_batch:
            if isinstance(ops_with_paths, SplitNewDeleteOp):
                assert ops_with_paths.is_remove_all_childs
                store_id = ops_with_paths.value
                if store_id is None:
                    store_id = self._main_store_id
                store = self._store[store_id]
                await store.remove_folder(ops_with_paths.key)
            else:
                for path, (store_id, ops_mixed) in ops_with_paths.items():
                    batch_ops: list[StoreBackendOp] = []
                    cur_update_ops: list[DraftUpdateOp] = []
                    store = self._store[store_id]
                    for op_mixed in ops_mixed:
                        if isinstance(op_mixed, DraftUpdateOp):
                            cur_update_ops.append(op_mixed)
                        else:
                            if cur_update_ops:
                                cur_update_ops = [
                                    o.to_json_update_op().to_userdata_removed(
                                    ) for o in cur_update_ops
                                ]
                                batch_ops.append(
                                    StoreBackendOp(path,
                                                   StoreWriteOpType.UPDATE,
                                                   cur_update_ops))
                                cur_update_ops = []
                            if op_mixed.is_new:
                                batch_ops.append(
                                    StoreBackendOp(path,
                                                   StoreWriteOpType.WRITE,
                                                   op_mixed.value))
                            else:
                                rm_type = StoreWriteOpType.REMOVE if op_mixed.is_remove_all_childs else StoreWriteOpType.REMOVE_FOLDER
                                batch_ops.append(
                                    StoreBackendOp(path, rm_type, None))
                    if cur_update_ops:
                        cur_update_ops = [
                            o.to_json_update_op().to_userdata_removed()
                            for o in cur_update_ops
                        ]
                        batch_ops.append(
                            StoreBackendOp(path, StoreWriteOpType.UPDATE,
                                           cur_update_ops))
                    await store.batch_update(batch_ops)
