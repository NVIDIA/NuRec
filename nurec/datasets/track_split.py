# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass(frozen=True, slots=True)
class TrackSplitChild:
    parent_track_id: str
    child_track_id: str
    start_timestamp_us: int
    end_timestamp_us: int
    child_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_track_id": self.parent_track_id,
            "child_track_id": self.child_track_id,
            "start_timestamp_us": self.start_timestamp_us,
            "end_timestamp_us": self.end_timestamp_us,
            "child_index": self.child_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackSplitChild:
        return cls(
            parent_track_id=str(data["parent_track_id"]),
            child_track_id=str(data["child_track_id"]),
            start_timestamp_us=int(data["start_timestamp_us"]),
            end_timestamp_us=int(data["end_timestamp_us"]),
            child_index=int(data["child_index"]),
        )


@dataclass(frozen=True, slots=True)
class TrackSplitMetadata:
    children: list[TrackSplitChild] = field(default_factory=list)
    version: int = 1

    @property
    def is_empty(self) -> bool:
        return not self.children

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "children": [child.to_dict() for child in self.children]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrackSplitMetadata:
        if data is None:
            return cls()
        raw_children = data.get("children", [])
        if not isinstance(raw_children, list):
            raise ValueError("track split metadata 'children' must be a list")
        return cls(
            version=int(data.get("version", 1)),
            children=[TrackSplitChild.from_dict(child) for child in raw_children],
        )

    def map_track_ids(self, mapper: Callable[[str], str]) -> TrackSplitMetadata:
        return TrackSplitMetadata(
            version=self.version,
            children=[
                TrackSplitChild(
                    parent_track_id=mapper(child.parent_track_id),
                    child_track_id=mapper(child.child_track_id),
                    start_timestamp_us=child.start_timestamp_us,
                    end_timestamp_us=child.end_timestamp_us,
                    child_index=child.child_index,
                )
                for child in self.children
            ],
        )


class TrackSplitResolver:
    """Maps external parent track ids to internal split child track ids."""

    def __init__(self, metadata: TrackSplitMetadata | dict[str, Any] | None = None) -> None:
        if isinstance(metadata, dict) or metadata is None:
            metadata = TrackSplitMetadata.from_dict(metadata)
        self._metadata = metadata
        self._children_by_parent: dict[str, list[TrackSplitChild]] = {}
        self._parent_by_child: dict[str, str] = {}
        for child in metadata.children:
            self._children_by_parent.setdefault(child.parent_track_id, []).append(child)
            self._parent_by_child[child.child_track_id] = child.parent_track_id
        for children in self._children_by_parent.values():
            children.sort(key=lambda child: (child.start_timestamp_us, child.end_timestamp_us, child.child_index))

    @property
    def metadata(self) -> TrackSplitMetadata:
        return self._metadata

    @property
    def has_splits(self) -> bool:
        return not self._metadata.is_empty

    def to_parent_track_id(self, track_id: str) -> str:
        return self._parent_by_child.get(track_id, track_id)

    def child_ids_for_parent(self, parent_track_id: str) -> list[str]:
        children = self._children_by_parent.get(parent_track_id)
        if children is None:
            return [parent_track_id]
        return [child.child_track_id for child in children]

    def child_ids_overlapping_interval(
        self, parent_track_id: str, start_timestamp_us: int, end_timestamp_us: int
    ) -> list[str]:
        children = self._children_by_parent.get(parent_track_id)
        if children is None:
            return [parent_track_id]

        overlapping_children = [
            child.child_track_id
            for child in children
            if not (end_timestamp_us < child.start_timestamp_us or start_timestamp_us > child.end_timestamp_us)
        ]
        if overlapping_children:
            return overlapping_children
        return [self.active_child_id(parent_track_id, (start_timestamp_us + end_timestamp_us) // 2)]

    def external_track_ids(self, internal_track_ids: Sequence[str]) -> list[str]:
        external_track_ids: list[str] = []
        seen: set[str] = set()
        for track_id in internal_track_ids:
            external_track_id = self.to_parent_track_id(track_id)
            if external_track_id in seen:
                continue
            seen.add(external_track_id)
            external_track_ids.append(external_track_id)
        return external_track_ids

    def active_child_id(self, parent_track_id: str, timestamp_us: int) -> str:
        children = self._children_by_parent.get(parent_track_id)
        if children is None:
            return parent_track_id

        containing_children = [
            child for child in children if child.start_timestamp_us <= timestamp_us <= child.end_timestamp_us
        ]
        if containing_children:
            return max(containing_children, key=lambda child: child.start_timestamp_us).child_track_id

        closest_child = min(
            children,
            key=lambda child: min(
                abs(timestamp_us - child.start_timestamp_us),
                abs(timestamp_us - child.end_timestamp_us),
            ),
        )
        return closest_child.child_track_id
