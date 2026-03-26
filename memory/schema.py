from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any
from enum import StrEnum
from datetime import datetime, timezone

class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"

class MemoryRecord(BaseModel):
    memory_id: str = Field(min_length=1)
    persona_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0, description="The sequence number of the turn, starting at 0")
    status: MemoryStatus = MemoryStatus.ACTIVE
    normalized_mem_text: str = Field(min_length=1)
    memory_type: str = Field(default="general_preference", description="semantic category of memory for filtering and versioning")
    time_marker: str = Field(default_factory=lambda:datetime.now(timezone.utc).isoformat())
    supersession_link: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(use_enum_values=True)


