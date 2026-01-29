from dataclasses import dataclass, fields, MISSING


@dataclass
class InputData:
    images: list[str]
    prompt: str
    negative_prompt: str | None = None
    steps: int | None = None
    guidance_scale: float | None = None

def validate_input(data: dict) -> dict[str, str] | None:
    field_names = {f.name for f in fields(InputData)}
    required = {
        f.name for f in fields(InputData)
        if f.default is MISSING and f.default_factory is MISSING
    }
    missing = required - data.keys()
    extra = data.keys() - field_names

    if missing or extra:
        errors = []
        if missing:
            errors.append(f"missing fields: {sorted(missing)}")
        if extra:
            errors.append(f"extra fields: {sorted(extra)} is not allowed")

        return {"error": f'{"; ".join(errors)}'}
    else:
        return None
