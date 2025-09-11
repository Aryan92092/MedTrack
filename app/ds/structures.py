from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple


@dataclass(order=True)
class MedicineNode:
    expiry_date: date
    id: int
    name: str
    quantity: int


class MinExpiryHeap:
    def __init__(self) -> None:
        self._heap: List[MedicineNode] = []
        self._id_to_index: Dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._heap)

    def peek(self) -> Optional[MedicineNode]:
        return self._heap[0] if self._heap else None

    def push(self, node: MedicineNode) -> None:
        self._heap.append(node)
        idx = len(self._heap) - 1
        self._id_to_index[node.id] = idx
        self._sift_up(idx)

    def pop(self) -> Optional[MedicineNode]:
        if not self._heap:
            return None
        last = len(self._heap) - 1
        self._swap(0, last)
        node = self._heap.pop()
        self._id_to_index.pop(node.id, None)
        if self._heap:
            self._sift_down(0)
        return node

    def remove(self, medicine_id: int) -> Optional[MedicineNode]:
        idx = self._id_to_index.get(medicine_id)
        if idx is None:
            return None
        last = len(self._heap) - 1
        self._swap(idx, last)
        node = self._heap.pop()
        self._id_to_index.pop(node.id, None)
        if idx < len(self._heap):
            self._sift_down(idx)
            self._sift_up(idx)
        return node

    def update(self, node: MedicineNode) -> None:
        idx = self._id_to_index.get(node.id)
        if idx is None:
            self.push(node)
            return
        self._heap[idx] = node
        self._sift_down(idx)
        self._sift_up(idx)

    def as_sorted_list(self) -> List[MedicineNode]:
        # Non-destructive: copy then pop all
        temp = MinExpiryHeap()
        for n in self._heap:
            temp.push(n)
        result: List[MedicineNode] = []
        node = temp.pop()
        while node is not None:
            result.append(node)
            node = temp.pop()
        return result

    def _sift_up(self, idx: int) -> None:
        while idx > 0:
            parent = (idx - 1) // 2
            if self._heap[idx] < self._heap[parent]:
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _sift_down(self, idx: int) -> None:
        size = len(self._heap)
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx
            if left < size and self._heap[left] < self._heap[smallest]:
                smallest = left
            if right < size and self._heap[right] < self._heap[smallest]:
                smallest = right
            if smallest != idx:
                self._swap(idx, smallest)
                idx = smallest
            else:
                break

    def _swap(self, i: int, j: int) -> None:
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._id_to_index[self._heap[i].id] = i
        self._id_to_index[self._heap[j].id] = j


class NameHashMap:
    def __init__(self) -> None:
        self._name_to_ids: Dict[str, List[int]] = {}

    def add(self, name: str, medicine_id: int) -> None:
        key = name.lower()
        self._name_to_ids.setdefault(key, []).append(medicine_id)

    def remove(self, name: str, medicine_id: int) -> None:
        key = name.lower()
        ids = self._name_to_ids.get(key)
        if not ids:
            return
        self._name_to_ids[key] = [i for i in ids if i != medicine_id]
        if not self._name_to_ids[key]:
            self._name_to_ids.pop(key, None)

    def get_ids(self, name: str) -> List[int]:
        return list(self._name_to_ids.get(name.lower(), []))


