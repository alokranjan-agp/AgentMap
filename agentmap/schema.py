
from __future__ import annotations
from typing import Any, Dict
from jsonschema import Draft202012Validator

def validate_inputs(schema: Dict[str, Any], data: Dict[str, Any]) -> None:
    if not schema:
        return
    Draft202012Validator(schema).validate(data)
