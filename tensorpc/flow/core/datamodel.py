import contextlib
from collections.abc import Mapping, Sequence
from functools import partial
from typing import (Any, Callable, Coroutine, Generic, Optional, TypeVar,
                    Union, cast)

from mashumaro.codecs.basic import BasicDecoder, BasicEncoder
from pydantic import field_validator
from typing_extensions import Self, TypeAlias

import tensorpc.core.datamodel.jmes as jmespath
from tensorpc.core import dataclass_dispatch as dataclasses
from tensorpc.core.datamodel.draft import (
    DraftBase, DraftUpdateOp, apply_draft_update_ops, capture_draft_update,
    create_draft, create_draft_type_only, enter_op_process_ctx,
    evaluate_draft_ast_noexcept, get_draft_ast_node, get_draft_jmespath,
    insert_assign_draft_op)
from tensorpc.core.datamodel.events import (DraftChangeEvent,
                                            DraftChangeEventHandler,
                                            DraftEventType,
                                            update_model_with_change_event)
from tensorpc.flow.core.component import (Component, ContainerBase,
                                          ContainerBaseProps, DraftOpUserData,
                                          EventSlotEmitter,
                                          EventSlotNoArgEmitter, UIType)
from tensorpc.flow.coretypes import StorageType

from ..jsonlike import Undefined, as_dict_no_undefined, undefined
from .appcore import Event

T = TypeVar("T")
_T = TypeVar("_T")
_CORO_NONE: TypeAlias = Union[Coroutine[None, None, None], None]


@dataclasses.dataclass
class DataModelProps(ContainerBaseProps):
    dataObject: Any = dataclasses.field(default_factory=dict)


class DataModel(ContainerBase[DataModelProps, Component], Generic[_T]):
    """DataModel is the model part of classic MVC pattern, child components can use `bind_fields` to query data from
    this component.
    
    Pitfalls:
        model may be replaced when you connect it to a storage, you should always access model instance via property instead of 
            keep it by user.
    """

    def __init__(
        self,
        model: _T,
        children: Union[Sequence[Component], Mapping[str, Component]],
    ) -> None:
        if children is not None and isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.DataModel,
                         DataModelProps,
                         children,
                         allowed_events=[])
        self.prop(dataObject=model)
        self._model: _T = model
        self._backend_draft_update_event_key = "__backend_draft_update"
        self._backend_storage_fetched_event_key = "__backend_storage_fetched"

        self.event_draft_update: EventSlotEmitter[
            list[DraftUpdateOp]] = self._create_emitter_event_slot(
                self._backend_draft_update_event_key)
        self.event_storage_fetched: EventSlotNoArgEmitter = self._create_emitter_event_slot_noarg(
            self._backend_storage_fetched_event_key)

        self._app_storage_handler_registered: bool = False

        self._is_model_dataclass = dataclasses.is_dataclass(type(model))
        self._is_model_pydantic_dataclass = dataclasses.is_pydantic_dataclass(
            type(model))

        self._mashumaro_decoder: Optional[BasicDecoder] = None
        self._mashumaro_encoder: Optional[BasicEncoder] = None

        self._draft_change_event_handlers: dict[str, dict[
            Callable, DraftChangeEventHandler]] = {}

    def _draft_change_handler_effect(self, path: str,
                                     handler: DraftChangeEventHandler):
        if path not in self._draft_change_event_handlers:
            self._draft_change_event_handlers[path] = {}
        self._draft_change_event_handlers[path][handler.handler] = handler

        def unmount():
            self._draft_change_event_handlers[path].pop(handler.handler)

        return unmount

    def install_draft_change_handler(
            self,
            comp: Component,
            draft: Any,
            handler: Callable[[DraftChangeEvent], _CORO_NONE],
            equality_fn: Optional[Callable[[Any, Any], bool]] = None,
            handle_child_change: bool = False,
            ):
        assert isinstance(draft, DraftBase)
        node = get_draft_ast_node(draft)
        path = node.get_jmes_path()
        handler_obj = DraftChangeEventHandler(node, handler, equality_fn,
                                              handle_child_change)
        effect_fn = partial(self._draft_change_handler_effect, path,
                            handler_obj)
        comp.use_effect(effect_fn)
        # return effect_fn to let user remove the effect.
        return handler_obj, effect_fn

    def _lazy_get_mashumaro_coder(self):
        if self._mashumaro_decoder is None:
            self._mashumaro_decoder = BasicDecoder(type(self.model))
        if self._mashumaro_encoder is None:
            self._mashumaro_encoder = BasicEncoder(type(self.model))
        return self._mashumaro_decoder, self._mashumaro_encoder

    @property
    def model(self) -> _T:
        return self._model

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    async def sync_model(self):
        await self.send_and_wait(self.update_event(dataObject=self.model))

    def get_draft(self) -> _T:
        """Create draft object, the generated draft AST is depend on real object.
        All opeartion you appied to draft object must be valid for real object.
        """
        return cast(_T, create_draft(self.model,
                                     userdata=DraftOpUserData(self)))

    def get_draft_type_only(self) -> _T:
        """Create draft object, but the generated draft AST is depend on annotation type instead of real object.
        useful when your draft ast contains optional/undefined path, this kind of path produce undefined in frontend,
        but raise error if we use real object.

        We also enable method support in this mode, which isn't allowed in object mode.
        """
        return cast(
            _T,
            create_draft_type_only(type(self.model),
                                   userdata=DraftOpUserData(self)))

    async def _update_with_jmes_ops_event(self, ops: list[DraftUpdateOp]):
        if not self._draft_change_event_handlers:
            apply_draft_update_ops(self.model, ops)
        else:
            all_disabled_ev_handlers: set[DraftChangeEventHandler] = set()
            for op in ops:
                userdata = op.get_userdata_typed(DraftOpUserData)
                if userdata is not None:
                    all_disabled_ev_handlers.update(userdata.disabled_handlers)
            all_ev_handlers: list[DraftChangeEventHandler] = []
            for path, handlers in self._draft_change_event_handlers.items():
                for handler in handlers.values():
                    if handler not in all_disabled_ev_handlers:
                        all_ev_handlers.append(handler)
            event_handler_changes = update_model_with_change_event(
                self.model, ops, all_ev_handlers)
            cbs: list[Callable[[], _CORO_NONE]] = []
            for change, handler in zip(event_handler_changes, all_ev_handlers):
                if change != DraftEventType.NoChange:
                    draft_change_ev = DraftChangeEvent(change[0], change[1])
                    if handler.user_eval_vars:
                        # user can define custom evaluates to get new model value.
                        user_vars = {}
                        for k, expr in handler.user_eval_vars.items():
                            obj = evaluate_draft_ast_noexcept(expr, self.model)
                            user_vars[k] = obj
                        draft_change_ev.user_eval_vars = user_vars
                    cbs.append(partial(handler.handler, draft_change_ev))
            # TODO should we allow user change draft inside draft change handler (may cause infinite loop)?
            await self.run_callbacks(cbs,
                                     change_status=False,
                                     capture_draft=True)
        # apply_draft_update_ops(self.model, ops)
        # import rich
        # rich.print(as_dict_no_undefined(self.model))
        frontend_ops = [op.to_jmes_path_op().to_dict() for op in ops]
        return self.create_comp_event({
            "type": 0,
            "ops": frontend_ops,
        })

    async def _update_with_jmes_ops(self, ops: list[DraftUpdateOp]):
        if ops:
            await self.flow_event_emitter.emit_async(
                self._backend_draft_update_event_key,
                Event(self._backend_draft_update_event_key, ops))
            return await self.send_and_wait(
                await self._update_with_jmes_ops_event(ops))

    async def _update_with_jmes_ops_event_for_internal(
            self, ops: list[DraftUpdateOp]):
        # internal event handle system will call this function
        await self.flow_event_emitter.emit_async(
            self._backend_draft_update_event_key,
            Event(self._backend_draft_update_event_key, ops))
        return await self._update_with_jmes_ops_event(ops)

    def bind_fields_unchecked(
        self, **kwargs: Union[str, tuple["Component", Union[str, DraftBase]],
                              DraftBase]
    ) -> Self:
        raise NotImplementedError("you can't bind fields on DataModel")

    @contextlib.asynccontextmanager
    async def draft_update(self):
        """Do draft update immediately after this context.
        We won't perform real update during draft operation because we need to keep state same between
        frontend and backend. if your update code raise error during draft operation, the real model in backend won't 
        be updated, so the state is still same. if we do backend update in each draft update, the state
        will be different between frontend and backend when exception happen.

        If your code after draft depends on the updated model, you can use this ctx to perform
        update immediately.
        """
        draft = self.get_draft()
        with capture_draft_update() as ctx:
            yield draft
        await self._update_with_jmes_ops(ctx._ops)

    @staticmethod
    def get_draft_external(model: T) -> T:
        return cast(T, create_draft(model, userdata=None))

    def connect_app_storage(self,
                            path: str,
                            storage_type: StorageType = StorageType.JSON):
        """Register event handler that store and send update info to app storage.
        WARNING: this function must be called in layout function.
        """
        assert self._is_model_dataclass, "only support dataclass model when use app storage"
        assert not self._app_storage_handler_registered, "only support connect once if you want to change path/type, create a new component."
        assert dataclasses.is_dataclass(
            self.model), "only support dataclass model"
        self.event_after_mount.on(
            partial(self._fetch_internal_data_from_app_storage,
                    path=path,
                    storage_type=storage_type))
        self.event_draft_update.on(
            partial(self._handle_app_storage_update,
                    path=path,
                    storage_type=storage_type))
        self._app_storage_handler_registered = True

    async def _fetch_internal_data_from_app_storage(self, path: str,
                                                    storage_type: StorageType):
        from tensorpc.flow import appctx
        assert dataclasses.is_dataclass(
            self.model), "only support dataclass model"
        storage = appctx.get_app_storage()
        data = await storage.read_data_storage(path, raise_if_not_found=False)
        if data is None:
            # not exist, create new
            data = as_dict_no_undefined(self.model)
            await storage.save_data_storage(path,
                                            data,
                                            storage_type=storage_type)
            return
        if self._is_model_dataclass:
            if dataclasses.is_pydantic_dataclass(type(self.model)):
                self._model = type(self.model)(**data)  # type: ignore
            else:
                # plain dataclass don't support create from dict, so we use `mashumaro` decoder here. it's fast.
                dec, _ = self._lazy_get_mashumaro_coder()
                self._model = dec.decode(data)  # type: ignore
        else:
            self._model = data
        self.props.dataObject = self._model
        await self.flow_event_emitter.emit_async(
            self._backend_storage_fetched_event_key,
            Event(self._backend_storage_fetched_event_key, None))
        # finally sync the model.
        await self.sync_model()

    async def _handle_app_storage_update(self, ops: list[DraftUpdateOp],
                                         path: str, storage_type: StorageType):
        from tensorpc.flow import appctx
        storage = appctx.get_app_storage()
        success = await storage.update_data_storage(
            path, [o.to_json_update_op().to_userdata_removed() for o in ops])
        if not success:
            # path not exist.
            data = as_dict_no_undefined(self.model)
            await storage.save_data_storage(path,
                                            data,
                                            storage_type=storage_type)

    @staticmethod
    def _op_proc(op: DraftUpdateOp, handlers: list[DraftChangeEventHandler]):
        userdata = op.get_userdata_typed(DraftOpUserData)
        if userdata is None:
            return op
        userdata.disabled_handlers.extend(handlers)
        return op

    @staticmethod
    @contextlib.contextmanager
    def add_disabled_handler_ctx(handlers: list[DraftChangeEventHandler]):
        with enter_op_process_ctx(
                partial(DataModel._op_proc, handlers=handlers)):
            yield


@dataclasses.dataclass
class DataPortalProps(ContainerBaseProps):
    comps: list[Component] = dataclasses.field(default_factory=list)
    query: Union[Undefined, str] = undefined


class DataPortal(ContainerBase[DataPortalProps, Component]):
    """DataPortal is used to forward multiple container that isn't direct parent.
    can only be used with DataModel and resource loaders.
    """

    def __init__(
        self,
        sources: list[Component],
        children: Optional[Union[Sequence[Component],
                                 Mapping[str, Component]]] = None
    ) -> None:
        if children is not None and isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        allowed_comp_types = {
            UIType.DataModel, UIType.ThreeURILoaderContext,
            UIType.ThreeCubeCamera
        }
        for comp in sources:
            assert comp._flow_comp_type in allowed_comp_types, "DataPortal only support DataModel and resource loaders."
        assert len(sources) > 0, "DataPortal must have at least one source"
        super().__init__(UIType.DataPortal,
                         DataPortalProps,
                         children,
                         allowed_events=[])
        self.prop(comps=sources)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    def bind_fields_unchecked(
        self, **kwargs: Union[str, tuple["Component", Union[str, DraftBase]],
                              DraftBase]
    ) -> Self:
        if "comps" not in kwargs and "query" not in kwargs:
            return super().bind_fields_unchecked(**kwargs)
        raise NotImplementedError(
            "you can't bind `comps` and `query` on DataModel")


@dataclasses.dataclass
class DataSubQueryProps(ContainerBaseProps):
    query: Union[Undefined, str] = undefined
    enable: Union[Undefined, bool] = undefined

    @field_validator('query')
    def jmes_query_validator(cls, v: Union[str, Undefined]):
        assert isinstance(v, str), "query must be string"
        # compile test
        jmespath.compile(v)


class DataSubQuery(ContainerBase[DataSubQueryProps, Component]):

    def __init__(
        self,
        query: str,
        children: Optional[Union[Sequence[Component],
                                 Mapping[str, Component]]] = None
    ) -> None:
        # compile test
        jmespath.compile(query)
        if children is not None and isinstance(children, Sequence):
            children = {str(i): v for i, v in enumerate(children)}
        super().__init__(UIType.DataSubQuery,
                         DataSubQueryProps,
                         children,
                         allowed_events=[])
        self.prop(query=query)

    @property
    def prop(self):
        propcls = self.propcls
        return self._prop_base(propcls, self)

    @property
    def update_event(self):
        propcls = self.propcls
        return self._update_props_base(propcls)

    def bind_fields_unchecked(
        self, **kwargs: Union[str, tuple["Component", Union[str, DraftBase]],
                              DraftBase]
    ) -> Self:
        if "query" not in kwargs:
            return super().bind_fields_unchecked(**kwargs)
        raise NotImplementedError(
            "you can't bind `comps` and `query` on DataModel")
