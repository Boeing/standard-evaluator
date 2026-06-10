"""Module to combine multiple Pydantic v2 BaseModel instances into a single
combined model and validated instance while preserving and reconciling
annotations, metadata, and numeric constraints.

This is the "robust patch" version: appearance collection and bounds/optionality
extraction are defensive and inspect nested ``default``, ``metadata``, and ``extra``
places where pydantic may store Field/constraint information across versions.
"""

from typing import Any, Dict, List, Tuple, Type, Optional, get_origin, get_args, Union
from pydantic import BaseModel, create_model, Field
from pydantic.fields import FieldInfo
from annotated_types import Interval
from pydantic import conint, confloat
import math

NumericBounds = Dict[str, Optional[float]]  # keys: 'ge','gt','le','lt'


def _is_optional_annotation(annotation: Any) -> bool:
    """Determines whether an annotation allows None.

    Args:
        annotation: A type annotation which may be a Union, PEP 604 union,
            or Annotated.

    Returns:
        True if the annotation permits None, otherwise False.
    """
    if annotation is None:
        return False
    try:
        args = get_args(annotation)
        if args:
            return any(a is type(None) for a in args)
    except Exception:
        pass
    # fallback: check typing.Union origin
    try:
        origin = get_origin(annotation)
        if origin is Union:
            return type(None) in get_args(annotation)
    except Exception:
        pass
    return False


def _strip_optional(annotation: Any) -> Any:
    """Removes NoneType from union-style annotations.

    Args:
        annotation: A type annotation, possibly a Union or PEP 604 union.

    Returns:
        The annotation with None removed. If only None present, returns type(None).
    """
    try:
        args = get_args(annotation)
        if args:
            non_none = tuple(a for a in args if a is not type(None))
            if len(non_none) == 0:
                return type(None)
            if len(non_none) == 1:
                return non_none[0]
            return Union[non_none]
    except Exception:
        pass
    return annotation


def _choose_numeric_kind(annotations: List[Any]) -> str:
    """Decides whether numeric fields should be treated as 'int' or 'float'.

    Args:
        annotations: A list of annotations from the various appearances of a field.

    Returns:
        'float' if any annotation indicates float-like behavior, otherwise 'int'.
    """
    for ann in annotations:
        if ann is None:
            continue
        ann_stripped = _strip_optional(ann)
        # direct float type
        if ann_stripped is float:
            return "float"
        # constrained float types often have names containing 'float' or start with 'confloat'
        name = getattr(ann_stripped, "__name__", "")
        if isinstance(name, str) and "float" in name.lower():
            return "float"
        # Annotated types: check metadata for Interval or FieldInfo suggesting floats
        try:
            meta = getattr(ann_stripped, "__metadata__", None)
            if meta:
                for m in meta:
                    if isinstance(m, Interval):
                        return "float"
                    if isinstance(m, FieldInfo):
                        for k in ("ge", "gt", "le", "lt"):
                            val = getattr(m, k, None)
                            if val is not None and (not float(val).is_integer()):
                                return "float"
        except Exception:
            pass
        # origins and args
        try:
            origin = get_origin(ann_stripped)
            if origin is float:
                return "float"
            args = get_args(ann_stripped)
            if args:
                for a in args:
                    if a is float:
                        return "float"
        except Exception:
            pass
    return "int"


def _extract_bounds_from_field_info(finfo: Any) -> NumericBounds:
    """Defensively extracts numeric bounds from many possible finfo shapes.

    Inspects FieldInfo instances, dict-style finfo, nested defaults, and
    annotated Interval metadata.

    Args:
        finfo: A FieldInfo instance, dict, or None.

    Returns:
        A dict with keys 'ge', 'gt', 'le', 'lt' mapped to float or None.
    """
    bounds: NumericBounds = {"ge": None, "gt": None, "le": None, "lt": None}
    if finfo is None:
        return bounds

    def _try_set(k: str, val: Any) -> None:
        """Tries to coerce a value to float and set it into bounds[k]."""
        if val is None:
            return
        try:
            bounds[k] = float(val)
        except Exception:
            # ignore values that can't be coerced
            pass

    def _extract_from_obj(obj: Any) -> None:
        """Extracts bounds values from an object that may expose attributes,
        metadata, default, or extra mapping.

        Args:
            obj: Object to inspect for numeric bound attributes.
        """
        if obj is None:
            return
        # attributes
        for k in ("ge", "gt", "le", "lt"):
            _try_set(k, getattr(obj, k, None))
        # metadata if present as attribute
        meta = getattr(obj, "metadata", None)
        if meta:
            if isinstance(meta, (list, tuple)):
                for m in meta:
                    # try to extract ge/gt/le/lt from any metadata object
                    for k in ("ge", "gt", "le", "lt"):
                        _try_set(k, getattr(m, k, None))
        # default as attribute
        default = getattr(obj, "default", None)
        if isinstance(default, FieldInfo) or isinstance(default, dict):
            _extract_from_obj(default)
        # extra mapping if present
        extra = getattr(obj, "extra", None)
        if isinstance(extra, dict):
            for k in ("ge", "gt", "le", "lt"):
                _try_set(k, extra.get(k))

    # If dict-style finfo (pydantic v2 model_fields entry)
    if isinstance(finfo, dict):
        # direct keys
        for k in ("ge", "gt", "le", "lt"):
            if k in finfo:
                _try_set(k, finfo.get(k))
        # metadata key
        meta = finfo.get("metadata", None)
        if meta:
            if isinstance(meta, (list, tuple)):
                for m in meta:
                    for k in ("ge", "gt", "le", "lt"):
                        _try_set(k, getattr(m, k, None))
        # default key (may be FieldInfo or dict with keys)
        default = finfo.get("default", None)
        if isinstance(default, FieldInfo) or isinstance(default, dict):
            _extract_from_obj(default)
        # extra key
        extra = finfo.get("extra", None)
        if isinstance(extra, dict):
            for k in ("ge", "gt", "le", "lt"):
                _try_set(k, extra.get(k))
        # also check nested annotation-level Interval via annotation metadata if present
        ann = finfo.get("annotation", None)
        if ann is not None:
            ann_bounds = _extract_bounds_from_annotation(ann)
            for k in ann_bounds:
                if ann_bounds[k] is not None:
                    _try_set(k, ann_bounds[k])
        return bounds

    # FieldInfo or arbitrary object
    _extract_from_obj(finfo)

    return bounds


def _extract_bounds_from_annotation(annotation: Any) -> NumericBounds:
    bounds: NumericBounds = {"ge": None, "gt": None, "le": None, "lt": None}
    if annotation is None:
        return bounds

    # PEP 593 Annotated metadata available on __metadata__ (tuple/list)
    try:
        metadata = getattr(annotation, "__metadata__", None)
        if metadata:
            for meta in metadata:
                for k in ("ge", "gt", "le", "lt"):
                    try:
                        val = getattr(meta, k, None)
                    except Exception:
                        val = None
                    if val is not None:
                        try:
                            bounds[k] = float(val)
                        except Exception:
                            pass
    except Exception:
        pass

    # annotation attributes (conint/confloat types expose these)
    for k in ("ge", "gt", "le", "lt"):
        val = getattr(annotation, k, None)
        if val is not None:
            try:
                bounds[k] = float(val)
            except Exception:
                pass

    return bounds


def _merge_bounds(bounds_list: List[NumericBounds]) -> NumericBounds:
    best_lower_val = -math.inf
    best_lower_strict = False
    for b in bounds_list:
        for kind in ("gt", "ge"):
            val = b.get(kind)
            if val is None:
                continue
            strict = kind == "gt"
            if val > best_lower_val or (
                val == best_lower_val and strict and not best_lower_strict
            ):
                best_lower_val = val
                best_lower_strict = strict

    best_upper_val = math.inf
    best_upper_strict = False
    for b in bounds_list:
        for kind in ("lt", "le"):
            val = b.get(kind)
            if val is None:
                continue
            strict = kind == "lt"
            if val < best_upper_val or (
                val == best_upper_val and strict and not best_upper_strict
            ):
                best_upper_val = val
                best_upper_strict = strict

    result: NumericBounds = {"ge": None, "gt": None, "le": None, "lt": None}
    if best_lower_val != -math.inf:
        if best_lower_strict:
            result["gt"] = best_lower_val
        else:
            result["ge"] = best_lower_val
    if best_upper_val != math.inf:
        if best_upper_strict:
            result["lt"] = best_upper_val
        else:
            result["le"] = best_upper_val
    return result


def _collect_first_metadata(finfo_list: List[Any]) -> Optional[FieldInfo]:
    """Selects the first finfo entry that provides descriptive metadata.

    Args:
        finfo_list: List of FieldInfo or dict-style finfo entries.

    Returns:
        A FieldInfo-like object or None.
    """
    for finfo in finfo_list:
        if isinstance(finfo, FieldInfo):
            if any(
                getattr(finfo, k, None) is not None
                for k in ("description", "title", "example")
            ):
                return finfo
        elif isinstance(finfo, dict):
            default = finfo.get("default", None)
            if isinstance(default, FieldInfo):
                if any(
                    getattr(default, k, None) is not None
                    for k in ("description", "title", "example")
                ):
                    return default
            if any(
                k in finfo and finfo[k] is not None
                for k in ("description", "title", "example")
            ):
                meta_kwargs = {
                    k: finfo[k]
                    for k in ("description", "title", "example")
                    if k in finfo and finfo[k] is not None
                }
                return Field(**meta_kwargs)
    return None


def _validate_merged_bounds(merged: NumericBounds) -> None:
    """Validates that merged bounds are not contradictory.

    Args:
        merged: The merged bounds to validate.

    Raises:
        ValueError: If the merged bounds are contradictory (empty feasible set).
    """
    lower_val = None
    lower_strict = False
    if merged.get("gt") is not None:
        lower_val = merged["gt"]
        lower_strict = True
    elif merged.get("ge") is not None:
        lower_val = merged["ge"]
        lower_strict = False

    upper_val = None
    upper_strict = False
    if merged.get("lt") is not None:
        upper_val = merged["lt"]
        upper_strict = True
    elif merged.get("le") is not None:
        upper_val = merged["le"]
        upper_strict = False

    if lower_val is not None and upper_val is not None:
        if lower_val > upper_val:
            raise ValueError(
                f"Contradictory bounds: lower bound {('>' if lower_strict else '>=')}{lower_val} "
                f"is greater than upper bound {('<' if upper_strict else '<=')}{upper_val}."
            )
        if lower_val == upper_val and (lower_strict or upper_strict):
            raise ValueError(
                f"Contradictory bounds: lower bound {('>' if lower_strict else '>=')}{lower_val} "
                f"and upper bound {('<' if upper_strict else '<=')}{upper_val} produce empty set."
            )


def _appearance_is_optional(annotation: Any, finfo: Any) -> bool:
    """Determines if an appearance is optional.

    An appearance is optional if the annotation explicitly allows None,
    or the finfo/default explicitly has default None.

    Args:
        annotation: The type annotation for the field appearance.
        finfo: The FieldInfo or dict-style field info.

    Returns:
        True if the appearance is optional, False otherwise.
    """
    try:
        if _is_optional_annotation(annotation):
            return True
    except Exception:
        pass

    # FieldInfo case
    if isinstance(finfo, FieldInfo):
        if getattr(finfo, "default", Ellipsis) is None:
            return True
        # nested default
        nested = getattr(finfo, "default", None)
        if (
            isinstance(nested, FieldInfo)
            and getattr(nested, "default", Ellipsis) is None
        ):
            return True
    # dict-style
    if isinstance(finfo, dict):
        if "default" in finfo and finfo.get("default", Ellipsis) is None:
            return True
        nested = finfo.get("default", None)
        if (
            isinstance(nested, FieldInfo)
            and getattr(nested, "default", Ellipsis) is None
        ):
            return True

    return False


def combine_instances(instances: List[BaseModel]) -> Tuple[Type[BaseModel], BaseModel]:
    """Combines multiple Pydantic v2 BaseModel instances into one.

    Args:
        instances: List of instantiated pydantic BaseModel objects to combine.

    Returns:
        A tuple of (CombinedModelClass, combined_instance).

    Raises:
        ValueError: If no instances are provided or if contradictory numeric
            bounds are found.
    """
    if not instances:
        raise ValueError("No instances provided")

    # collect appearances: Dict[field_name, List[ (annotation, finfo, cls) ]]
    field_appearances: Dict[str, List[Tuple[Any, Any, Type[BaseModel]]]] = {}
    for inst in instances:
        cls = inst.__class__
        model_fields = getattr(cls, "model_fields", None)
        if model_fields is not None:
            for fname, finfo in model_fields.items():
                if isinstance(finfo, dict):
                    ann = finfo.get("annotation")
                    field_appearances.setdefault(fname, []).append((ann, finfo, cls))
                    # if default nested is FieldInfo/dict, also append that appearance so nested bounds are discovered
                    default = finfo.get("default", None)
                    if isinstance(default, FieldInfo) or isinstance(default, dict):
                        # keep the same annotation (ann) but provide the nested finfo for extraction
                        field_appearances.setdefault(fname, []).append(
                            (ann, default, cls)
                        )
                else:
                    ann = getattr(finfo, "annotation", None)
                    field_appearances.setdefault(fname, []).append((ann, finfo, cls))
                    # if finfo has .default that is FieldInfo/dict, append it too
                    default = getattr(finfo, "default", None)
                    if isinstance(default, FieldInfo) or isinstance(default, dict):
                        field_appearances.setdefault(fname, []).append(
                            (ann, default, cls)
                        )
        else:
            # fallback: use __annotations__ and class attribute defaults
            annotations = getattr(cls, "__annotations__", {})
            for fname, ann in annotations.items():
                default = getattr(cls, fname, ...)
                finfo = {"annotation": ann, "default": default}
                field_appearances.setdefault(fname, []).append((ann, finfo, cls))
                # if default is FieldInfo/dict, append as separate appearance
                if isinstance(default, FieldInfo) or isinstance(default, dict):
                    field_appearances.setdefault(fname, []).append((ann, default, cls))

    combined_fields_for_create: Dict[str, Tuple[Any, Any]] = {}

    for fname, appearances in field_appearances.items():
        anns = [a for a, _, _ in appearances]
        finfos = [f for _, f, _ in appearances]

        # Optional only if optional in ALL appearances
        is_optional_all: bool = all(
            _appearance_is_optional(a, f) for a, f, _ in appearances
        )

        # Gather bounds from all appearances (iterate appearances to avoid zip truncation)
        bounds_list: List[NumericBounds] = []
        for a, f, _ in appearances:
            bounds_list.append(_extract_bounds_from_field_info(f))
            bounds_list.append(_extract_bounds_from_annotation(a))
        merged_bounds = _merge_bounds(bounds_list)
        try:
            _validate_merged_bounds(merged_bounds)
        except ValueError as exc:
            raise ValueError(f"Field '{fname}': {exc}") from exc

        bounds_present = any(v is not None for v in merged_bounds.values())
        numeric_kind = _choose_numeric_kind(anns)
        candidate_meta = _collect_first_metadata(finfos)

        ann_with_bounds = next(
            (a for a in anns if any(_extract_bounds_from_annotation(a).values())), None
        )
        first_non_none_ann = next((a for a in anns if a is not None), None)
        base_ann: Any = (
            ann_with_bounds
            if ann_with_bounds is not None
            else (first_non_none_ann or Any)
        )
        base_ann = _strip_optional(base_ann)

        if bounds_present:
            ge = merged_bounds.get("ge")
            gt = merged_bounds.get("gt")
            le = merged_bounds.get("le")
            lt = merged_bounds.get("lt")
            kwargs: Dict[str, Any] = {}
            if ge is not None:
                kwargs["ge"] = int(ge) if numeric_kind == "int" else ge
            if gt is not None:
                kwargs["gt"] = int(gt) if numeric_kind == "int" else gt
            if le is not None:
                kwargs["le"] = int(le) if numeric_kind == "int" else le
            if lt is not None:
                kwargs["lt"] = int(lt) if numeric_kind == "int" else lt

            if numeric_kind == "int":
                constrained = conint(**kwargs) if kwargs else int
            else:
                constrained = confloat(**kwargs) if kwargs else float

            final_ann: Any = Optional[constrained] if is_optional_all else constrained

            meta_kwargs: Dict[str, Any] = {}
            if candidate_meta is not None and isinstance(candidate_meta, FieldInfo):
                for k in ("description", "title", "example"):
                    val = getattr(candidate_meta, k, None)
                    if val is not None:
                        meta_kwargs[k] = val

            bound_kwargs: Dict[str, Any] = {}
            if ge is not None:
                bound_kwargs["ge"] = int(ge) if numeric_kind == "int" else ge
            if gt is not None:
                bound_kwargs["gt"] = int(gt) if numeric_kind == "int" else gt
            if le is not None:
                bound_kwargs["le"] = int(le) if numeric_kind == "int" else le
            if lt is not None:
                bound_kwargs["lt"] = int(lt) if numeric_kind == "int" else lt

            chosen_default = (
                Field(default=..., **{**bound_kwargs, **meta_kwargs})
                if (bound_kwargs or meta_kwargs)
                else ...
            )
        else:
            final_ann = Optional[base_ann] if is_optional_all else base_ann

            # If the field is required in any appearance, ensure the combined model marks it required (no default None).
            # But preserve descriptive metadata (description/title/example) by attaching them to Field(...).
            chosen_default: Any = ...
            meta_kwargs: Dict[str, Any] = {}
            if candidate_meta is not None and isinstance(candidate_meta, FieldInfo):
                for k in ("description", "title", "example"):
                    val = getattr(candidate_meta, k, None)
                    if val is not None:
                        meta_kwargs[k] = val

            if is_optional_all:
                # if truly optional in all appearances, prefer any explicit default or FieldInfo
                for finfo in finfos:
                    if isinstance(finfo, FieldInfo):
                        chosen_default = finfo
                        break
                    if isinstance(finfo, dict):
                        default = finfo.get("default", None)
                        if isinstance(default, FieldInfo):
                            chosen_default = default
                            break
                        if "default" in finfo and finfo["default"] is not Ellipsis:
                            chosen_default = finfo["default"]
                            break
                # if metadata present but no explicit default, create a Field(default=None, **meta_kwargs)
                if chosen_default is ... and meta_kwargs:
                    chosen_default = Field(default=None, **meta_kwargs)
            else:
                # required: attach metadata to required Field(...) if present
                if meta_kwargs:
                    chosen_default = Field(..., **meta_kwargs)

        combined_fields_for_create[fname] = (final_ann, chosen_default)

    Combined: Type[BaseModel] = create_model("Combined", **combined_fields_for_create)

    # Build combined instance values using first non-None from instances
    combined_values: Dict[str, Any] = {}
    for fname in combined_fields_for_create.keys():
        chosen_value: Any = None
        found_non_none: bool = False
        any_attr: bool = False
        for inst in instances:
            if hasattr(inst, fname):
                any_attr = True
                val = getattr(inst, fname)
                if val is not None:
                    chosen_value = val
                    found_non_none = True
                    break
                if chosen_value is None:
                    chosen_value = None
        if found_non_none:
            combined_values[fname] = chosen_value
        else:
            if any_attr:
                combined_values[fname] = None
            else:
                # omit and allow model default
                pass

    combined_instance: BaseModel = Combined(**combined_values)
    return Combined, combined_instance
