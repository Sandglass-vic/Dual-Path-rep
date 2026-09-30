#!/usr/bin/env python3
"""Add the DUAL_PATH_KVTRANS=layer|full switch to SGLang PD disaggregation.

Installs scripts/infra/dualpath_kvtrans.py as sglang/srt/utils/ and patches:

* Prefill path, ``full``: HiRadixCache preloads a request's whole host-hit
  prefix into HBM outside the layer-wise load queue and the request is only
  admitted into a batch once every TP rank finished (one gloo MIN per
  scheduler iteration covers all pending preloads).
* Decode path, ``layer``: Decode RDMA-writes the prefix into Prefill HBM one
  layer at a time, flushes it with a 4-byte RDMA read, then writes a
  per-request flag; the Prefill forward waits on that flag with
  cuStreamWaitValue32 before attention reads each layer.

Must run after patch_sglang_benchmark_timing.py. Idempotent.
"""

from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path

MARKER = "_dualpath_kvtrans"
IMPORT_LINE = "from sglang.srt.utils import dualpath_kvtrans as _dualpath_kvtrans\n"
LOGGER_LINE = "logger = logging.getLogger(__name__)\n"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return source.replace(old, new, 1)


def add_import(source: str, label: str) -> str:
    return replace_once(
        source, LOGGER_LINE, LOGGER_LINE + IMPORT_LINE, f"{label} import"
    )


def patch_cache_controller(source: str) -> str:
    source = add_import(source, "cache_controller")
    source = replace_once(
        source,
        "        self.producer_index = -1\n"
        "        self.consumer_index = -1\n"
        "\n"
        "    def update_producer(self):\n",
        "        self.producer_index = -1\n"
        "        self.consumer_index = -1\n"
        "        # Device addresses of Decode-written layer flags for the batch\n"
        "        # being forwarded (DUAL_PATH_KVTRANS=layer).\n"
        "        self.dualpath_remote_flags = []\n"
        "        self.dualpath_remote_waited = -1\n"
        "\n"
        "    def update_producer(self):\n",
        "LayerDoneCounter remote flag state",
    )
    source = replace_once(
        source,
        "    def wait_until(self, threshold: int):\n"
        "        if self.consumer_index < 0:\n"
        "            return\n"
        "        self.events[self.consumer_index].wait(threshold)\n",
        "    def wait_until(self, threshold: int):\n"
        "        if self.consumer_index >= 0:\n"
        "            self.events[self.consumer_index].wait(threshold)\n"
        "        # K and V of a layer both call in; one GPU wait per layer is enough.\n"
        "        if self.dualpath_remote_flags and threshold > self.dualpath_remote_waited:\n"
        "            _dualpath_kvtrans.wait_remote_layer(\n"
        "                self.dualpath_remote_flags, threshold\n"
        "            )\n"
        "            self.dualpath_remote_waited = threshold\n",
        "LayerDoneCounter remote wait",
    )
    source = replace_once(
        source,
        "        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)\n",
        "        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)\n"
        "        _dualpath_kvtrans.mark_remote_wait_ready()\n",
        "mark remote wait ready",
    )
    source = replace_once(
        source,
        "    def dualpath_loading_timing_events(self, producer_id: int):\n",
        "    def dualpath_full_load(self, host_indices: torch.Tensor, node_id: int):\n"
        "        \"\"\"Issue a whole-prefix H2D load outside the layer-wise queue.\n"
        "\n"
        "        It neither takes a LayerDoneCounter slot nor joins load_queue, so\n"
        "        it cannot be merged into, or block, the next batch's layer-wise\n"
        "        load. Completion is acked through ack_load_queue as usual.\n"
        "        \"\"\"\n"
        "        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))\n"
        "        if device_indices is None:\n"
        "            return None\n"
        "        op = CacheOperation(host_indices, device_indices, node_id)\n"
        "        io_host_indices, io_device_indices = self.move_indices(op)\n"
        "        start_event = device_module.Event()\n"
        "        finish_event = device_module.Event()\n"
        "        timing_start = dualpath_new_cuda_event(device_module)\n"
        "        timing_finish = dualpath_new_cuda_event(device_module)\n"
        "        start_event.record()\n"
        "        with device_module.stream(self.load_stream):\n"
        "            start_event.wait(self.load_stream)\n"
        "            if timing_start is not None:\n"
        "                timing_start.record(self.load_stream)\n"
        "            for i in range(self.layer_num):\n"
        "                self.mem_pool_host.load_to_device_per_layer(\n"
        "                    self.mem_pool_device,\n"
        "                    io_host_indices,\n"
        "                    io_device_indices,\n"
        "                    i,\n"
        "                    self.io_backend,\n"
        "                )\n"
        "            finish_event.record(self.load_stream)\n"
        "            if timing_finish is not None:\n"
        "                timing_finish.record(self.load_stream)\n"
        "            if io_host_indices.is_cuda:\n"
        "                io_host_indices.record_stream(self.load_stream)\n"
        "            if io_device_indices.is_cuda:\n"
        "                io_device_indices.record_stream(self.load_stream)\n"
        "        self.ack_load_queue.append(\n"
        "            HiCacheAck(\n"
        "                start_event=start_event,\n"
        "                finish_event=finish_event,\n"
        "                node_ids=op.node_ids,\n"
        "            )\n"
        "        )\n"
        "        return device_indices, finish_event, (timing_start, timing_finish)\n"
        "\n"
        "    def dualpath_loading_timing_events(self, producer_id: int):\n",
        "controller full load",
    )
    return source


HIRADIX_METHODS = '''    def dualpath_full_preload(self, req, token_budget: int) -> bool:
        """DUAL_PATH_KVTRANS=full admission gate for one waiting request.

        Returns True when the request may be scheduled: either its host-hit
        prefix is already resident on every TP rank, or there is nothing
        worth preloading (the layer-wise path then handles any remainder).
        """
        entry = self.dualpath_full_loads.get(req.rid)
        if entry is not None:
            return entry[2]
        req.init_next_round_input(self)
        node = req.last_host_node
        if req.host_hit_length <= 0 or not node.evicted:
            return True
        # Preloaded prefixes stay locked until admission, so bound them to
        # one batch worth of tokens to keep the queue from pinning HBM.
        held = sum(e[3] for e in self.dualpath_full_loads.values())
        if held and held + req.host_hit_length > token_budget:
            return False
        loaded = self._dualpath_full_load_back(node)
        if loaded is None:
            return True
        finish_event, timing_events, num_tokens = loaded
        self.inc_lock_ref(node)
        self.dualpath_full_loads[req.rid] = [
            node, finish_event, False, num_tokens, timing_events
        ]
        return False

    def _dualpath_full_load_back(self, node: TreeNode):
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert node.backuped, "No backup available on evicted nodes"
            nodes_to_load.insert(0, node)
            node = node.parent
        ancestor_node = node
        self.inc_lock_ref(ancestor_node)
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold:
            self.dec_lock_ref(ancestor_node)
            return None
        loaded = self.cache_controller.dualpath_full_load(
            host_indices, last_hit_node.id
        )
        if loaded is None:
            self.evict(len(host_indices))
            loaded = self.cache_controller.dualpath_full_load(
                host_indices, last_hit_node.id
            )
        self.dec_lock_ref(ancestor_node)
        if loaded is None:
            return None
        device_indices, finish_event, timing_events = loaded
        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for n in nodes_to_load:
            n.value = device_indices[offset : offset + len(n.host_value)]
            offset += len(n.host_value)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)
        return finish_event, timing_events, len(device_indices)

    def dualpath_full_poll(self):
        pending = [e for e in self.dualpath_full_loads.values() if not e[2]]
        if not pending:
            return
        done = torch.tensor(
            [int(e[1].query()) for e in pending], dtype=torch.int32
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                done, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        for entry, ok in zip(pending, done.tolist()):
            if ok:
                entry[2] = True

    def dualpath_full_admitted(self, req, admitted: bool):
        if not admitted:
            return
        entry = self.dualpath_full_loads.pop(req.rid, None)
        if entry is None:
            return
        # add_one_req took its own lock on the request's last node.
        self.dec_lock_ref(entry[0])
        timing_events = entry[4]
        if timing_events[0] is not None:
            req.dualpath_h2d_event_pairs = getattr(
                req, "dualpath_h2d_event_pairs", []
            )
            req.dualpath_h2d_event_pairs.append(timing_events)

    def dualpath_full_release(self, rid: str):
        entry = self.dualpath_full_loads.pop(rid, None)
        if entry is not None:
            self.dec_lock_ref(entry[0])

'''


def patch_hiradix(source: str) -> str:
    source = replace_once(
        source,
        "        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)\n",
        "        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)\n"
        "        # rid -> [node, finish_event, ready_on_all_ranks, tokens, timing]\n"
        "        self.dualpath_full_loads = {}\n",
        "hiradix full-load state",
    )
    source = replace_once(
        source,
        "    def ready_to_load_host_cache(self) -> int:\n",
        HIRADIX_METHODS + "    def ready_to_load_host_cache(self) -> int:\n",
        "hiradix full-load methods",
    )
    source = replace_once(
        source,
        "        self.writing_check()\n        self.loading_check()\n",
        "        self.writing_check()\n        self.loading_check()\n"
        "        self.dualpath_full_poll()\n",
        "hiradix full-load poll",
    )
    source = replace_once(
        source,
        "    def release_aborted_request(self, rid: str):\n"
        "        if rid not in self.ongoing_prefetch:\n",
        "    def release_aborted_request(self, rid: str):\n"
        "        self.dualpath_full_release(rid)\n"
        "        if rid not in self.ongoing_prefetch:\n",
        "hiradix full-load abort",
    )
    return source


def patch_scheduler(source: str) -> str:
    source = add_import(source, "scheduler")
    return replace_once(
        source,
        "            if not getattr(\n"
        "                req, \"dualpath_decode_l3_prefix_ready\", False\n"
        "            ):\n"
        "                req.init_next_round_input(self.tree_cache)\n"
        "            res = adder.add_one_req(\n"
        "                req,\n"
        "                has_chunked_req=(self.chunked_req is not None),\n"
        "                truncation_align_size=self.truncation_align_size,\n"
        "            )\n",
        "            dualpath_full_gate = (\n"
        "                _dualpath_kvtrans.PREFILL_MODE == \"full\"\n"
        "                and self.enable_hierarchical_cache\n"
        "                and not getattr(req, \"dualpath_decode_l3_prefix_ready\", False)\n"
        "            )\n"
        "            if dualpath_full_gate and not self.tree_cache.dualpath_full_preload(\n"
        "                req, self.max_prefill_tokens\n"
        "            ):\n"
        "                continue\n"
        "            if not getattr(\n"
        "                req, \"dualpath_decode_l3_prefix_ready\", False\n"
        "            ):\n"
        "                req.init_next_round_input(self.tree_cache)\n"
        "            res = adder.add_one_req(\n"
        "                req,\n"
        "                has_chunked_req=(self.chunked_req is not None),\n"
        "                truncation_align_size=self.truncation_align_size,\n"
        "            )\n"
        "            if dualpath_full_gate:\n"
        "                self.tree_cache.dualpath_full_admitted(\n"
        "                    req, req in adder.can_run_list\n"
        "                )\n",
        "scheduler full admission gate",
    )


def patch_tp_worker(source: str) -> str:
    return replace_once(
        source,
        "            self.set_hicache_consumer(model_worker_batch.hicache_consumer_index)\n",
        "            self.set_hicache_consumer(model_worker_batch.hicache_consumer_index)\n"
        "            if self.hicache_layer_transfer_counter is not None:\n"
        "                counter = self.hicache_layer_transfer_counter\n"
        "                counter.dualpath_remote_flags = [\n"
        "                    ptr\n"
        "                    for ptr in (\n"
        "                        getattr(req, \"dualpath_kvtrans_flag_ptr\", 0)\n"
        "                        for req in (model_worker_batch.reqs or ())\n"
        "                    )\n"
        "                    if ptr\n"
        "                ]\n"
        "                counter.dualpath_remote_waited = -1\n",
        "tp_worker remote flags",
    )


CONN_MANAGER_METHODS = '''    def _dualpath_init_kvtrans(self):
        self.dualpath_layer_flags = None
        self.dualpath_layer_slots = {}
        self.dualpath_layer_consts = None
        self.dualpath_reverse_queue = None
        layer_capable = (
            _dualpath_kvtrans.DECODE_MODE == "layer" and not self.is_mla_backend
        )
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if layer_capable:
                self.dualpath_layer_flags = _dualpath_kvtrans.RemoteLayerFlags(
                    self.engine
                )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            if layer_capable:
                self.dualpath_layer_consts = _dualpath_kvtrans.LayerSendConstants(
                    self.engine, len(self.kv_args.kv_data_ptrs) // 2
                )
            # D->P runs off decode_thread so KV status messages keep flowing.
            self.dualpath_reverse_queue = _dualpath_queue.SimpleQueue()
            threading.Thread(
                target=self._dualpath_reverse_worker, daemon=True
            ).start()

    def _dualpath_reverse_worker(self):
        while True:
            msg = self.dualpath_reverse_queue.get()
            try:
                self._handle_decode_l3_reverse_fetch(msg)
            except Exception:
                logger.exception("Decode L3 reverse-fetch handler failed")

    def _dualpath_alloc_layer_slot(self, room, num_infos):
        old_slot = self.dualpath_layer_slots.pop(room, None)
        if old_slot is not None:
            self.dualpath_layer_flags.release(old_slot, room)
        if (
            self.dualpath_layer_flags is None
            or num_infos != 1
            or not _dualpath_kvtrans.remote_wait_ready
        ):
            return b"full", 0
        slot = self.dualpath_layer_flags.alloc(
            len(self.kv_args.kv_data_ptrs) // 2, room
        )
        if slot is None:
            logger.warning(
                "DUAL_PATH_KVTRANS=layer flag slots exhausted; room %d "
                "falls back to full block",
                room,
            )
            return b"full", 0
        self.dualpath_layer_slots[room] = slot
        return b"layer", self.dualpath_layer_flags.host_addr(slot)

    def dualpath_layer_admit(self, room) -> int:
        """Arm the watchdog and return the flag's device address, or 0."""
        slot = self.dualpath_layer_slots.get(room)
        if slot is None:
            return 0
        self.dualpath_layer_flags.arm(slot)
        return self.dualpath_layer_flags.device_addr(slot)

    def dualpath_layer_check_timeout(self, room):
        slot = self.dualpath_layer_slots.get(room)
        if slot is None or not self.dualpath_layer_flags.timed_out(slot):
            return
        self.dualpath_layer_flags.timedout[slot] = 0
        self.record_failure(
            room, "Decode-to-Prefill layer stream timed out"
        )
        self.update_status(room, KVPoll.Failed)

    def _dualpath_write_flag(self, session_id, flag_addr, layer):
        consts = self.dualpath_layer_consts
        if consts is None or not flag_addr:
            return -1
        index = consts.num_layers - 1 if layer is None else layer
        return self.engine.batch_transfer_sync(
            session_id, [consts.value_ptr(index)], [flag_addr], [4]
        )

    def _dualpath_send_layerwise(
        self,
        session_id,
        dst_ptrs,
        src_indices,
        dst_indices,
        flag_addr,
        receiver,
        start_ns,
    ) -> int:
        consts = self.dualpath_layer_consts
        if consts is None:
            raise RuntimeError(
                "Decode cannot serve a layer-wise D->P transfer "
                "(DUAL_PATH_KVTRANS != layer or MLA backend)"
            )
        src_k, src_v, dst_k, dst_v, num_layers = self.get_mha_kv_ptrs_with_pp(
            self.kv_args.kv_data_ptrs, dst_ptrs
        )
        if num_layers != consts.num_layers:
            raise RuntimeError(
                f"layer-wise D->P needs matching layer counts: "
                f"{num_layers} != {consts.num_layers}"
            )
        item_lens = self.kv_args.kv_item_lens
        src_blocks, dst_blocks = group_concurrent_contiguous(
            src_indices, dst_indices
        )
        spans = [
            (int(s[0]), int(d[0]), len(s)) for s, d in zip(src_blocks, dst_blocks)
        ]
        scratch = consts.scratch_ptr
        for layer in range(num_layers):
            blocks = []
            probes = []
            for src_ptr, dst_ptr, item_len in (
                (src_k[layer], dst_k[layer], item_lens[layer]),
                (src_v[layer], dst_v[layer], item_lens[num_layers + layer]),
            ):
                for s, d, n in spans:
                    blocks.append(
                        (src_ptr + s * item_len, dst_ptr + d * item_len, n * item_len)
                    )
                probes.append(blocks[-1][1] + blocks[-1][2] - 4)
            ret = self._transfer_data(session_id, blocks)
            if ret != 0:
                return ret
            # An RDMA read through the same Prefill NIC cannot complete before
            # that NIC's earlier posted writes reached HBM, so the flag written
            # next never overtakes the layer's data.
            ret = self.engine.engine.batch_transfer_sync_read(
                session_id, [scratch, scratch + 8], probes, [4, 4]
            )
            if ret != 0:
                return ret
            ret = self._dualpath_write_flag(session_id, flag_addr, layer)
            if ret != 0:
                return ret
            if layer == 0 and DUALPATH_TTFT_ENABLED:
                receiver.dualpath_d2p_first_layer_ns = dualpath_now_ns() - start_ns
        return 0

'''


def patch_conn(source: str) -> str:
    source = replace_once(
        source,
        LOGGER_LINE,
        LOGGER_LINE + IMPORT_LINE + "import queue as _dualpath_queue\n",
        "conn import",
    )
    source = replace_once(
        source,
        "        self.decode_l3_reverse_responses = defaultdict(set)\n"
        "        if self.disaggregation_mode == DisaggregationMode.PREFILL:\n",
        "        self.decode_l3_reverse_responses = defaultdict(set)\n"
        "        self._dualpath_init_kvtrans()\n"
        "        if self.disaggregation_mode == DisaggregationMode.PREFILL:\n",
        "conn kvtrans init",
    )
    source = replace_once(
        source,
        "    def start_decode_l3_reverse(self, room, prefill_page_indices):\n",
        CONN_MANAGER_METHODS
        + "    def start_decode_l3_reverse(self, room, prefill_page_indices):\n",
        "conn kvtrans methods",
    )
    source = replace_once(
        source,
        "        self.decode_l3_reverse_responses.pop(room, None)\n"
        "        packed_ptrs = b\"\".join(\n",
        "        self.decode_l3_reverse_responses.pop(room, None)\n"
        "        kvtrans_kind, flag_addr = self._dualpath_alloc_layer_slot(\n"
        "            room, len(infos)\n"
        "        )\n"
        "        packed_ptrs = b\"\".join(\n",
        "reverse fetch slot",
    )
    source = replace_once(
        source,
        "                    np.asarray(\n"
        "                        prefill_page_indices, dtype=np.int32\n"
        "                    ).tobytes(),\n"
        "                ]\n"
        "            )\n",
        "                    np.asarray(\n"
        "                        prefill_page_indices, dtype=np.int32\n"
        "                    ).tobytes(),\n"
        "                    _dualpath_kvtrans.MODE_TAG,\n"
        "                    kvtrans_kind,\n"
        "                    str(flag_addr).encode(\"ascii\"),\n"
        "                ]\n"
        "            )\n",
        "reverse fetch frames",
    )
    source = replace_once(
        source,
        "    def decode_l3_reverse_is_done(self, room):\n"
        "        return room in self.decode_l3_reverse_done\n",
        "    def decode_l3_reverse_is_done(self, room):\n"
        "        if room in self.decode_l3_reverse_done:\n"
        "            return True\n"
        "        # Layer-wise: admissible once the first layer landed.\n"
        "        slot = self.dualpath_layer_slots.get(room)\n"
        "        return slot is not None and self.dualpath_layer_flags.ready(slot)\n",
        "reverse ready on first layer",
    )
    source = replace_once(
        source,
        "    def clear_decode_l3_reverse(self, room):\n",
        "    def clear_decode_l3_reverse(self, room):\n"
        "        slot = self.dualpath_layer_slots.pop(room, None)\n"
        "        if slot is not None:\n"
        "            self.dualpath_layer_flags.release(slot, room)\n",
        "release layer slot",
    )
    source = replace_once(
        source,
        "        dst_indices = np.frombuffer(msg[6], dtype=np.int32)\n"
        "        status = KVPoll.Failed\n"
        "        try:\n",
        "        dst_indices = np.frombuffer(msg[6], dtype=np.int32)\n"
        "        prefill_mode_tag = bytes(msg[7]) if len(msg) > 7 else b\"default\"\n"
        "        kvtrans_kind = bytes(msg[8]) if len(msg) > 8 else b\"full\"\n"
        "        flag_addr = int(msg[9].decode(\"ascii\")) if len(msg) > 9 else 0\n"
        "        status = KVPoll.Failed\n"
        "        try:\n"
        "            if prefill_mode_tag != _dualpath_kvtrans.MODE_TAG:\n"
        "                raise RuntimeError(\n"
        "                    \"DUAL_PATH_KVTRANS differs between Prefill \"\n"
        "                    f\"({prefill_mode_tag.decode()}) and Decode \"\n"
        "                    f\"({_dualpath_kvtrans.MODE_TAG.decode()})\"\n"
        "                )\n",
        "reverse fetch parse",
    )
    source = replace_once(
        source,
        "            ret = self._send_kvcache_generic(\n"
        "                mooncake_session_id=prefill_session_id,\n"
        "                src_data_ptrs=self.kv_args.kv_data_ptrs,\n"
        "                dst_data_ptrs=dst_ptrs,\n"
        "                item_lens=self.kv_args.kv_item_lens,\n"
        "                prefill_data_indices=src_indices,\n"
        "                dst_data_indices=dst_indices,\n"
        "                executor=None,\n"
        "            )\n",
        "            if kvtrans_kind == b\"layer\":\n"
        "                ret = self._dualpath_send_layerwise(\n"
        "                    prefill_session_id,\n"
        "                    dst_ptrs,\n"
        "                    src_indices,\n"
        "                    dst_indices,\n"
        "                    flag_addr,\n"
        "                    receiver,\n"
        "                    reverse_start_ns,\n"
        "                )\n"
        "            else:\n"
        "                ret = self._send_kvcache_generic(\n"
        "                    mooncake_session_id=prefill_session_id,\n"
        "                    src_data_ptrs=self.kv_args.kv_data_ptrs,\n"
        "                    dst_data_ptrs=dst_ptrs,\n"
        "                    item_lens=self.kv_args.kv_item_lens,\n"
        "                    prefill_data_indices=src_indices,\n"
        "                    dst_data_indices=dst_indices,\n"
        "                    executor=None,\n"
        "                )\n",
        "reverse fetch layer-wise send",
    )
    source = replace_once(
        source,
        "        self._connect(\n"
        "            format_tcp_address(prefill_endpoint, prefill_port),\n",
        "        if status != KVPoll.Success and kvtrans_kind == b\"layer\":\n"
        "            # Release the Prefill GPU wait now instead of at its deadline.\n"
        "            self._dualpath_write_flag(prefill_session_id, flag_addr, None)\n"
        "        self._connect(\n"
        "            format_tcp_address(prefill_endpoint, prefill_port),\n",
        "reverse fetch failure unblock",
    )
    source = replace_once(
        source,
        "                    reverse_session = waiting_req_bytes[3].decode(\"ascii\")\n",
        "                    reverse_session = waiting_req_bytes[3].decode(\"ascii\")\n"
        "                    if self.dualpath_layer_flags is not None:\n"
        "                        self.dualpath_layer_flags.ack(reverse_room)\n",
        "reverse done ack",
    )
    source = replace_once(
        source,
        "                    try:\n"
        "                        self._handle_decode_l3_reverse_fetch(msg)\n"
        "                    except Exception:\n",
        "                    if self.dualpath_reverse_queue is not None:\n"
        "                        self.dualpath_reverse_queue.put(msg)\n"
        "                        continue\n"
        "                    try:\n"
        "                        self._handle_decode_l3_reverse_fetch(msg)\n"
        "                    except Exception:\n",
        "decode thread dispatch",
    )
    source = replace_once(
        source,
        "    def poll(self) -> KVPoll:\n"
        "        if self.conclude_state is None:\n"
        "            status = self.kv_mgr.check_status(self.bootstrap_room)\n"
        "            if (\n"
        "                status == KVPoll.WaitingForInput\n",
        "    def poll(self) -> KVPoll:\n"
        "        if self.conclude_state is None:\n"
        "            self.kv_mgr.dualpath_layer_check_timeout(self.bootstrap_room)\n"
        "            status = self.kv_mgr.check_status(self.bootstrap_room)\n"
        "            if (\n"
        "                status == KVPoll.WaitingForInput\n",
        "sender layer timeout",
    )
    source = replace_once(
        source,
        "            self.dualpath_d2p_pages = 0\n"
        "        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)\n",
        "            self.dualpath_d2p_pages = 0\n"
        "            self.dualpath_d2p_first_layer_ns = 0\n"
        "        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)\n",
        "receiver first-layer timing",
    )
    return source


def patch_prefill(source: str) -> str:
    return replace_once(
        source,
        "        req.dualpath_decode_l3_prefix_ready = True\n",
        "        req.dualpath_kvtrans_flag_ptr = self.kv_manager.dualpath_layer_admit(\n"
        "            req.bootstrap_room\n"
        "        )\n"
        "        req.dualpath_decode_l3_prefix_ready = True\n",
        "prefill layer admission",
    )


def patch_decode(source: str) -> str:
    return replace_once(
        source,
        "                        \"d2p_pages\": getattr(dualpath_host_staging_receiver, \"dualpath_d2p_pages\", 0),\n",
        "                        \"d2p_pages\": getattr(dualpath_host_staging_receiver, \"dualpath_d2p_pages\", 0),\n"
        "                        \"d2p_first_layer_ns\": getattr(dualpath_host_staging_receiver, \"dualpath_d2p_first_layer_ns\", 0),\n",
        "decode first-layer timing",
    )


def patch_output_processor(source: str) -> str:
    return replace_once(
        source,
        "                        \"d_to_p_pages\": int(dualpath_decode.get(\"d2p_pages\", 0)),\n",
        "                        \"d_to_p_pages\": int(dualpath_decode.get(\"d2p_pages\", 0)),\n"
        "                        \"d_to_p_first_layer_seconds_diagnostic\": dualpath_ns_seconds(dualpath_decode.get(\"d2p_first_layer_ns\", 0)),\n"
        "                        \"kvtrans_mode\": _dualpath_kvtrans.MODE or \"default\",\n",
        "emit first-layer timing",
    )


PATCHES = {
    "managers/cache_controller.py": patch_cache_controller,
    "mem_cache/hiradix_cache.py": patch_hiradix,
    "managers/scheduler.py": patch_scheduler,
    "managers/tp_worker.py": patch_tp_worker,
    "disaggregation/mooncake/conn.py": patch_conn,
    "disaggregation/prefill.py": patch_prefill,
    "disaggregation/decode.py": patch_decode,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=Path("/usr/local/lib/python3.12/dist-packages"),
    )
    args = parser.parse_args()
    srt = args.site_packages / "sglang/srt"
    module = srt / "utils/dualpath_kvtrans.py"
    shutil.copyfile(Path(__file__).with_name("dualpath_kvtrans.py"), module)
    touched = [module]
    for relative, patch in PATCHES.items():
        path = srt / relative
        source = path.read_text()
        if any(
            marker in source
            for marker in ("dualpath_kvtrans", "dualpath_full_load", "d2p_first_layer_ns")
        ):
            print(f"Already patched: {path}")
            continue
        path.write_text(patch(source))
        touched.append(path)
        print(f"Patched: {path}")
    output_processor = srt / "managers/scheduler_output_processor_mixin.py"
    source = output_processor.read_text()
    if MARKER not in source:
        source = add_import(source, "output processor")
        output_processor.write_text(patch_output_processor(source))
        touched.append(output_processor)
        print(f"Patched: {output_processor}")
    for path in touched:
        py_compile.compile(str(path), doraise=True)


if __name__ == "__main__":
    main()
