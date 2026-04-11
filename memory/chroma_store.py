from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import chromadb

from memory.schema import MemoryRecord, MemoryStatus
from memory.store_interface import MemoryStore


class ChromaMemoryStore(MemoryStore):
    def __init__(
        self,
        run_dir: str,
        collection_name: str = "memory_records",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.persist_dir = self.run_dir / "chromadb"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def put_memory(self, record: MemoryRecord) -> MemoryRecord:
        payload = self._record_to_payload(record)
        self.collection.add(
            ids=[payload["id"]],
            documents=[payload["document"]],
            metadatas=[payload["metadata"]],
        )
        return record

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        result = self.collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"],
        )
        ids = result.get("ids", [])
        if not ids:
            return None

        return self._from_get_result(result, index=0)

    def list_memories(
        self,
        persona_id: str,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> list[MemoryRecord]:
        where = self._build_where_filter(
            persona_id=persona_id,
            status=status,
            memory_type=memory_type,
        )
        result = self.collection.get(
            where=where,
            include=["documents", "metadatas"],
        )
        return self._from_multi_get_result(result)

    def query_memories(
        self,
        persona_id: str,
        query_text: str,
        k: int,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> list[MemoryRecord]:
        where = self._build_where_filter(
            persona_id=persona_id,
            status=status,
            memory_type=memory_type,
        )
        result = self.collection.query(
            query_texts=[query_text],
            n_results=k,
            where=where,
            include=["documents", "metadatas"],
        )
        return self._from_query_result(result)

    def supersede_memory(
        self,
        memory_id: str,
        replacement_id: Optional[str] = None,
    ) -> None:
        existing = self.get_memory_by_id(memory_id)
        if existing is None:
            raise ValueError(f"Cannot supersede missing memory_id={memory_id}")

        updated_metadata = self._record_to_metadata(existing)
        updated_metadata["status"] = MemoryStatus.SUPERSEDED.value

        if replacement_id is not None:
            updated_metadata["replacement_id"] = replacement_id

        self.collection.update(
            ids=[memory_id],
            metadatas=[updated_metadata],
        )

    def _record_to_payload(self, record: MemoryRecord) -> dict[str, Any]:
        return {
            "id": record.memory_id,
            "document": record.normalized_mem_text,
            "metadata": self._record_to_metadata(record),
        }

    def _record_to_metadata(self, record: MemoryRecord) -> dict[str, Any]:
        status_value = record.status.value if isinstance(record.status, MemoryStatus) else str(record.status)

        metadata = {
            "persona_id": record.persona_id,
            "turn_index": record.turn_index,
            "status": status_value,
            "memory_type": record.memory_type,
            "time_marker": record.time_marker,
        }

        if record.supersession_link is not None:
            metadata["supersession_link"] = record.supersession_link

        for key, value in record.metadata.items():
            metadata[key] = value

        return metadata

    def _row_to_record(
        self,
        memory_id: str,
        document: str,
        metadata: dict[str, Any],
    ) -> MemoryRecord:
        base_metadata = dict(metadata)

        return MemoryRecord(
            memory_id=memory_id,
            persona_id=metadata["persona_id"],
            turn_index=metadata["turn_index"],
            status=metadata["status"],
            normalized_mem_text=document,
            memory_type=metadata["memory_type"],
            time_marker=metadata["time_marker"],
            supersession_link=metadata.get("supersession_link"),
            metadata=base_metadata,
        )

    def _build_where_filter(
        self,
        persona_id: str,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> dict[str, Any]:
        conditions: list[dict[str, Any]] = [{"persona_id": persona_id}]

        if status is not None:
            conditions.append({"status": status})
        if memory_type is not None:
            conditions.append({"memory_type": memory_type})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _from_get_result(self, result: dict[str, Any], index: int) -> MemoryRecord:
        memory_id = result["ids"][index]
        document = result["documents"][index]
        metadata = result["metadatas"][index]
        return self._row_to_record(memory_id, document, metadata)

    def _from_multi_get_result(self, result: dict[str, Any]) -> list[MemoryRecord]:
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        records: list[MemoryRecord] = []
        for memory_id, document, metadata in zip(ids, documents, metadatas):
            records.append(self._row_to_record(memory_id, document, metadata))
        return records

    def _from_query_result(self, result: dict[str, Any]) -> list[MemoryRecord]:
        ids_nested = result.get("ids", [])
        documents_nested = result.get("documents", [])
        metadatas_nested = result.get("metadatas", [])

        if not ids_nested:
            return []

        ids = ids_nested[0]
        documents = documents_nested[0] if documents_nested else []
        metadatas = metadatas_nested[0] if metadatas_nested else []

        records: list[MemoryRecord] = []
        for memory_id, document, metadata in zip(ids, documents, metadatas):
            records.append(self._row_to_record(memory_id, document, metadata))
        return records
