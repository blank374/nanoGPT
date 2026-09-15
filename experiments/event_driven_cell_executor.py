"""Event-driven packed execution for Full-Free Compute Cells.

The executor is deliberately independent from routing.  A router (or a replayed
graph) submits ready token events, grouped here as ``CellEventBatch`` objects.
Events targeting the same Cell are coalesced into one packed invocation and
scattered back to their original request/token slots afterwards.

This is the correctness-first executor.  It exposes the queue and dependency
semantics needed by a future persistent GPU scheduler without changing the
fixed 128->64->128 Cell atom.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import time

import torch


@dataclass
class CellEventBatch:
    """Ready events for one logical request shard and one target Cell."""

    cell_id: int
    fused: torch.Tensor
    active: torch.Tensor
    dependency_count: Optional[torch.Tensor] = None


class EventDrivenCellExecutor:
    """Coalesce ready events by Cell, execute packed queues, then scatter."""

    def __init__(self):
        self.last_stats = None

    @staticmethod
    def aggregate_stats(records):
        if not records:
            return None
        launches = sum(row["cell_launches"] for row in records)
        events = sum(row["ready_events"] for row in records)
        return {
            "steps": len(records),
            "submitted_jobs": sum(row["submitted_jobs"] for row in records),
            "ready_events": events,
            "nonempty_cell_queues": sum(row["nonempty_cell_queues"] for row in records),
            "cell_launches": launches,
            "coalesced_jobs": sum(row["coalesced_jobs"] for row in records),
            "mean_queue_size": float(events / launches) if launches else 0.0,
            "max_queue_size": max(row["max_queue_size"] for row in records),
            "mean_fan_in": float(sum(
                row["mean_fan_in"] * row["ready_events"] for row in records
            ) / events) if events else 0.0,
            "parallel_streams": any(row["parallel_streams"] for row in records),
            "per_step": records,
        }

    @staticmethod
    def dependency_waves(node_mask: torch.Tensor, edge_mask: torch.Tensor):
        """Return topological ready waves for one hard Cell graph.

        ``node_mask`` has shape [N]. ``edge_mask`` has shape [N, 1+N], where
        source zero is the already-ready current state and source k+1 is Cell k.
        Inactive predecessors do not create dependencies.  A cycle or a missing
        active predecessor is rejected rather than silently deadlocking.
        """
        nodes = node_mask.bool().flatten().cpu()
        edges = edge_mask.bool().cpu()
        if edges.shape != (nodes.numel(), nodes.numel() + 1):
            raise ValueError("edge_mask must have shape [N, 1+N]")
        active_ids = nodes.nonzero().flatten().tolist()
        remaining = set(active_ids)
        completed = set()
        waves = []
        while remaining:
            ready = []
            for node in sorted(remaining):
                parents = edges[node, 1:].nonzero().flatten().tolist()
                active_parents = [parent for parent in parents if bool(nodes[parent])]
                if all(parent in completed for parent in active_parents):
                    ready.append(node)
            if not ready:
                raise RuntimeError("Cell graph contains a cycle or unresolved dependency")
            waves.append(ready)
            completed.update(ready)
            remaining.difference_update(ready)
        return waves

    @staticmethod
    def _run_queue(cell_id, packed, input_norms, cells):
        return cells[cell_id](input_norms[cell_id](packed))

    def execute(
        self,
        jobs: Sequence[CellEventBatch],
        input_norms,
        cells,
        parallel_streams: bool = False,
    ) -> List[torch.Tensor]:
        """Execute all ready events and return one dense delta per input job."""
        if not jobs:
            self.last_stats = {
                "submitted_jobs": 0,
                "ready_events": 0,
                "nonempty_cell_queues": 0,
                "cell_launches": 0,
                "coalesced_jobs": 0,
                "mean_queue_size": 0.0,
                "max_queue_size": 0,
                "mean_fan_in": 0.0,
            }
            return []

        outputs = [torch.zeros_like(job.fused) for job in jobs]
        queues: Dict[int, list] = {}
        fan_in_sum = 0.0
        fan_in_events = 0
        for job_index, job in enumerate(jobs):
            if job.fused.shape[:-1] != job.active.shape:
                raise ValueError("active mask must match fused tensor except hidden dimension")
            active_flat = job.active.bool().reshape(-1)
            indices = active_flat.nonzero().flatten()
            if job.dependency_count is not None:
                dependencies = job.dependency_count.reshape(-1)[indices]
                if dependencies.numel() and bool((dependencies <= 0).any()):
                    raise RuntimeError("active Cell event was submitted before dependencies were ready")
                fan_in_sum += dependencies.float().sum().item()
                fan_in_events += dependencies.numel()
            if indices.numel() == 0:
                continue
            queues.setdefault(int(job.cell_id), []).append((job_index, indices))

        prepared = []
        for cell_id, parts in queues.items():
            packed_parts = [
                jobs[job_index].fused.reshape(-1, jobs[job_index].fused.size(-1))[indices]
                for job_index, indices in parts
            ]
            prepared.append((cell_id, parts, torch.cat(packed_parts, dim=0)))
        # Largest-ready-first improves useful work per launch and is deterministic.
        prepared.sort(key=lambda item: (-item[2].size(0), item[0]))

        queue_results = {}
        use_streams = (
            parallel_streams
            and prepared
            and prepared[0][2].is_cuda
            and len(prepared) > 1
        )
        if use_streams:
            producer = torch.cuda.current_stream(prepared[0][2].device)
            streams = [torch.cuda.Stream(device=prepared[0][2].device) for _ in prepared]
            for stream, (cell_id, _, packed) in zip(streams, prepared):
                stream.wait_stream(producer)
                with torch.cuda.stream(stream):
                    queue_results[cell_id] = self._run_queue(
                        cell_id, packed, input_norms, cells
                    )
            for stream in streams:
                producer.wait_stream(stream)
        else:
            for cell_id, _, packed in prepared:
                queue_results[cell_id] = self._run_queue(
                    cell_id, packed, input_norms, cells
                )

        queue_sizes = []
        for cell_id, parts, packed in prepared:
            result = queue_results[cell_id]
            queue_sizes.append(packed.size(0))
            cursor = 0
            for job_index, indices in parts:
                count = indices.numel()
                flat = outputs[job_index].reshape(-1, outputs[job_index].size(-1))
                values = result[cursor:cursor + count].to(flat.dtype)
                flat = flat.index_copy(0, indices, values)
                outputs[job_index] = flat.view_as(outputs[job_index])
                cursor += count

        submitted_nonempty = sum(len(parts) for _, parts, _ in prepared)
        self.last_stats = {
            "submitted_jobs": len(jobs),
            "ready_events": int(sum(queue_sizes)),
            "nonempty_cell_queues": len(prepared),
            "cell_launches": len(prepared),
            "coalesced_jobs": int(submitted_nonempty - len(prepared)),
            "mean_queue_size": (
                float(sum(queue_sizes) / len(queue_sizes)) if queue_sizes else 0.0
            ),
            "max_queue_size": int(max(queue_sizes, default=0)),
            "mean_fan_in": (
                float(fan_in_sum / fan_in_events) if fan_in_events else 0.0
            ),
            "parallel_streams": bool(use_streams),
            "dispatch_order": [cell_id for cell_id, _, _ in prepared],
            "queue_sizes": {str(cell_id): packed.size(0) for cell_id, _, packed in prepared},
        }
        return outputs


class PersistentCellEventScheduler:
    """Cross-request ready queues for an asynchronous serving loop.

    ``submit`` does not execute work. ``flush_ready`` dispatches a Cell when its
    accumulated active-event count reaches ``min_events``, its oldest shard has
    waited ``max_wait_seconds``, or the caller requests a forced drain.  Results
    are returned by ticket so a serving runtime can resume the corresponding
    request state machine.
    """

    def __init__(self):
        self._queues = {}
        self._next_ticket = 0
        self.last_flush_stats = None

    def submit(self, job: CellEventBatch):
        ticket = self._next_ticket
        self._next_ticket += 1
        entry = (ticket, time.monotonic(), job)
        self._queues.setdefault(int(job.cell_id), []).append(entry)
        return ticket

    def pending_stats(self):
        per_cell = {}
        for cell_id, entries in self._queues.items():
            events = sum(int(job.active.bool().sum().item()) for _, _, job in entries)
            per_cell[str(cell_id)] = {"shards": len(entries), "events": events}
        return {
            "pending_cells": len(per_cell),
            "pending_shards": sum(row["shards"] for row in per_cell.values()),
            "pending_events": sum(row["events"] for row in per_cell.values()),
            "per_cell": per_cell,
        }

    def flush_ready(
        self,
        input_norms,
        cells,
        min_events=32,
        max_wait_seconds=0.00005,
        force=False,
        parallel_streams=False,
    ):
        now = time.monotonic()
        chosen = []
        for cell_id, entries in self._queues.items():
            events = sum(int(job.active.bool().sum().item()) for _, _, job in entries)
            oldest_wait = now - entries[0][1]
            if force or events >= min_events or oldest_wait >= max_wait_seconds:
                chosen.append(cell_id)
        selected = []
        tickets = []
        for cell_id in chosen:
            entries = self._queues.pop(cell_id)
            for ticket, _, job in entries:
                tickets.append(ticket)
                selected.append(job)
        outputs = EventDrivenCellExecutor().execute(
            selected, input_norms, cells, parallel_streams=parallel_streams
        )
        completed = dict(zip(tickets, outputs))
        self.last_flush_stats = {
            "completed_tickets": tickets,
            "dispatched_cells": chosen,
            "completed_shards": len(tickets),
            "remaining": self.pending_stats(),
        }
        return completed
