from typing import List
from pydantic import BaseModel

class RequestDto(BaseModel):
    camera_id: int
    input: str
    stream_url: List[str]
    algorithm: str
    device: str