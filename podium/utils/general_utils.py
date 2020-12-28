"""
General utilities used across the codebase.
"""

import inspect


def add_repr(*, inspect_init=False, include_attrs=None, exclude_attrs=None):
    def repr_decorator(cls):
        def attrs_type(x):
            x = x or []
            if isinstance(x, str):
                x = [x]
            return x

        nonlocal include_attrs
        nonlocal exclude_attrs

        include_attrs = attrs_type(include_attrs)
        exclude_attrs = attrs_type(exclude_attrs)

        if inspect_init:
            attrs = list(inspect.signature(cls.__init__).parameters)[1:]
            if include_attrs:
                extra_attrs = [attr for attr in include_attrs if attr not in attrs]
                attrs = attrs + extra_attrs
        else:
            attrs = include_attrs

        attrs = [attr for attr in attrs if attr not in exclude_attrs]

        def _repr(self):
            attrs_dict = {}
            for attr in attrs:
                attr_name = attr
                if not hasattr(self, attr):
                    if attr.startswith("_"):
                        attr = attr.lstrip("_")
                    else:
                        attr = "_" + attr
                attrs_dict[attr_name] = getattr(self, attr)

            return (
                self.__class__.__qualname__
                + "("
                + ", ".join(f"{attr}={val!r}" for attr, val in attrs_dict.items())
                + ")"
            )

        cls.__repr__ = _repr
        return cls

    return repr_decorator
