from __future__ import annotations

from copy import deepcopy
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class CopyUtil:
    """Utility class for copy operations.

    Provides deep copy functionality for both Pydantic models and regular Python objects.
    """

    @staticmethod
    def deep_copy(obj: T) -> T:
        """Create a deep copy of an object.

        Uses model_copy(deep=True) for Pydantic models,
        and copy.deepcopy for other objects.

        Args:
            obj: Object to be copied

        Returns:
            Deep copied object

        Examples:
            >>> # Deep copy of a dictionary
            >>> original = {"key": {"nested": "value"}}
            >>> copied = CopyUtil.deep_copy(original)
            >>> copied["key"]["nested"] = "modified"
            >>> original["key"]["nested"]  # "value" (original data is unchanged)

            >>> # Deep copy of a Pydantic model
            >>> from pydantic import BaseModel
            >>> class MyModel(BaseModel):
            ...     data: dict
            >>> original_model = MyModel(data={"nested": "value"})
            >>> copied_model = CopyUtil.deep_copy(original_model)
            >>> copied_model.data["nested"] = "modified"
            >>> original_model.data["nested"]  # "value" (original data is unchanged)

        """
        if isinstance(obj, BaseModel):
            return obj.model_copy(deep=True)
        return deepcopy(obj)
