from typing import Any
import runpod

from app.generator import process
from app.schema import validate_input


def handler(job: dict[str, Any]) -> dict[str, str]:
    input_ = job["input"]
    if error := validate_input(data=input_):
        return error
    return process(
        job=job,
    )


runpod.serverless.start(
    {
        "handler": handler,
    }
)
