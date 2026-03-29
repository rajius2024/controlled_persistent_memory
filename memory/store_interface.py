from abc import ABC, abstractmethod
from typing import Optional

from memory.schema import MemoryRecord


class MemoryStore(ABC):
    @abstractmethod
    def put_memory(self, record: MemoryRecord) -> MemoryRecord:
        pass

    @abstractmethod
    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        pass

    @abstractmethod
    def list_memories(
        self,
        persona_id: str,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> list[MemoryRecord]:
        pass

    @abstractmethod
    def query_memories(
        self,
        persona_id: str,
        query_text: str,
        k: int,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> list[MemoryRecord]:
        pass

    @abstractmethod
    def supersede_memory(
        self,
        memory_id: str,
        replacement_id: Optional[str] = None,
    ) -> None:
        pass
