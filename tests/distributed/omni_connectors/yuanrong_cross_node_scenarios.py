# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual two-node YuanrongTransferEngineConnector functional scenarios.

This is intentionally a manually run test utility, not a pytest module.  It
focuses on functional behavior and failure classification for the CPU RDMA
path.  Ascend/GPU transports are reserved in the CLI and currently reported as
skipped.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from _yuanrong_cross_node_scenarios import (
    CaseResult,
    PeerChannel,
    Ports,
    YuanrongNode,
    elapsed_ms,
    handle_peer_command,
    load_runtime_deps,
    result_fail,
    result_ok,
)

DEFAULT_SCENARIOS = [
    "basic-copy",
    "basic-zerocopy",
    "object-payload",
    "bytes-payload",
    "many-requests-unique-keys",
    "two-consumers-different-keys",
    "same-key-sequential-consumers",
    "same-key-concurrent-consumers",
    "sender-restart-same-port",
    "receiver-restart-before-get",
    "receiver-restart-after-get",
    "bidirectional-two-connectors",
    "pool-exhaustion-then-cleanup-recovery",
    "wrong-sender-port-expected-timeout",
    "duplicate-put-same-key-overwrites-old-buffer",
]


class ScenarioRunner:
    def __init__(self, node: YuanrongNode, peer: PeerChannel, peer_host: str, peer_zmq_port: int, payload_size: int):
        self.node = node
        self.peer = peer
        self.peer_host = peer_host
        self.peer_zmq_port = peer_zmq_port
        self.payload_size = payload_size

    def peer_cmd(self, cmd: str, **kwargs: Any) -> dict[str, Any]:
        return self.peer.request({"cmd": cmd, **kwargs})

    def ensure_peer_receiver_to_local(self) -> None:
        ep = self.node.sender_endpoint()
        resp = self.peer_cmd("ensure_receiver", sender_host=ep.host, sender_zmq_port=ep.zmq_port)
        if not resp.get("ok"):
            raise RuntimeError(resp)

    def ensure_local_receiver_to_peer(self) -> None:
        self.node.ensure_receiver(self.peer_host, self.peer_zmq_port)

    def remote_get(self, put: dict[str, Any], *, expected_success: bool = True) -> dict[str, Any]:
        self.ensure_peer_receiver_to_local()
        resp = self.peer_cmd("get", key=put["key"], expected_md5=put["md5"])
        if expected_success and not resp.get("ok"):
            raise RuntimeError(resp)
        return resp

    def one_way(self, name: str, kind: str) -> CaseResult:
        started = time.perf_counter()
        put = self.node.put(name, kind, self.payload_size)
        if not put["ok"]:
            return result_fail(name, started, "put", "sender put returned false", put_ms=put["put_ms"])
        try:
            got = self.remote_get(put)
        except Exception as exc:
            return result_fail(name, started, "get", repr(exc), put_ms=put["put_ms"])
        return result_ok(name, started, put_ms=put["put_ms"], get_ms=got.get("get_ms", 0.0), bytes=got.get("size", 0))

    def many_unique(self) -> CaseResult:
        started = time.perf_counter()
        count = 10
        total = 0
        try:
            for idx in range(count):
                put = self.node.put(f"many-{idx}", "copy", max(1024, self.payload_size // 4))
                if not put["ok"]:
                    return result_fail("many-requests-unique-keys", started, "put", f"put failed at {idx}")
                got = self.remote_get(put)
                total += int(got.get("size", 0))
        except Exception as exc:
            return result_fail("many-requests-unique-keys", started, "get", repr(exc), bytes=total)
        return result_ok("many-requests-unique-keys", started, bytes=total, details={"count": count})

    def two_consumers_different_keys(self) -> CaseResult:
        name = "two-consumers-different-keys"
        started = time.perf_counter()
        self.ensure_peer_receiver_to_local()
        ep = self.node.sender_endpoint()
        self.node.ensure_receiver(ep.host, ep.zmq_port)
        put_local = self.node.put("two-consumers-local", "copy", self.payload_size)
        put_remote = self.node.put("two-consumers-remote", "copy", self.payload_size)
        if not put_local["ok"] or not put_remote["ok"]:
            return result_fail(name, started, "put", "one of the puts failed")
        local_result: dict[str, Any] = {}

        def local_get() -> None:
            local_result.update(self.node.get(put_local["key"], put_local["md5"]))

        t = threading.Thread(target=local_get)
        t.start()
        remote = self.peer_cmd("get", key=put_remote["key"], expected_md5=put_remote["md5"])
        t.join(timeout=60)
        if not local_result.get("ok"):
            return result_fail(name, started, "local-get", str(local_result))
        if not remote.get("ok"):
            return result_fail(name, started, "remote-get", str(remote))
        return result_ok(name, started, bytes=int(local_result.get("size", 0)) + int(remote.get("size", 0)))

    def same_key_sequential(self) -> CaseResult:
        name = "same-key-sequential-consumers"
        started = time.perf_counter()
        put = self.node.put(name, "copy", self.payload_size)
        if not put["ok"]:
            return result_fail(name, started, "put", "sender put returned false")
        try:
            first = self.remote_get(put)
        except Exception as exc:
            return result_fail(name, started, "first-get", repr(exc))
        ep = self.node.sender_endpoint()
        self.node.ensure_receiver(ep.host, ep.zmq_port)
        second = self.node.get(put["key"], put["md5"])
        if second.get("ok"):
            return result_fail(
                name,
                started,
                "second-get",
                "same key was consumed twice; connector is expected to be single-consumer per key",
            )
        return result_fail(
            name,
            started,
            "second-get",
            "second consumer could not pull an already-cleaned key",
            expected=True,
            details={"first_get": first, "second_get": second},
        )

    def same_key_concurrent(self) -> CaseResult:
        name = "same-key-concurrent-consumers"
        started = time.perf_counter()
        put = self.node.put(name, "copy", self.payload_size)
        if not put["ok"]:
            return result_fail(name, started, "put", "sender put returned false")
        self.ensure_peer_receiver_to_local()
        ep = self.node.sender_endpoint()
        self.node.ensure_receiver(ep.host, ep.zmq_port)
        local_result: dict[str, Any] = {}

        def local_get() -> None:
            local_result.update(self.node.get(put["key"], put["md5"]))

        t = threading.Thread(target=local_get)
        t.start()
        remote = self.peer_cmd("get", key=put["key"], expected_md5=put["md5"])
        t.join(timeout=60)
        ok_count = int(bool(local_result.get("ok"))) + int(bool(remote.get("ok")))
        if local_result.get("ok") and local_result.get("md5") != put["md5"]:
            return result_fail(name, started, "verify", "local consumer md5 mismatch")
        if remote.get("ok") and remote.get("md5") != put["md5"]:
            return result_fail(name, started, "verify", "remote consumer md5 mismatch")
        return result_fail(
            name,
            started,
            "topology",
            "same-key concurrent fanout is unsupported; result is recorded for behavior characterization",
            unsupported=True,
            details={"successful_consumers": ok_count, "local": local_result, "remote": remote},
        )

    def sender_restart_same_port(self) -> CaseResult:
        name = "sender-restart-same-port"
        started = time.perf_counter()
        put = self.node.put("restart-lost", "copy", self.payload_size)
        self.node.close_sender()
        lost = self.peer_cmd("get", key=put["key"], expected_md5=put["md5"])
        self.node.ensure_sender()
        put2 = self.node.put("restart-new", "copy", self.payload_size)
        try:
            got2 = self.remote_get(put2)
        except Exception as exc:
            return result_fail(name, started, "post-restart-get", repr(exc), details={"lost_get": lost})
        if lost.get("ok"):
            return result_fail(name, started, "pre-restart-get", "old payload unexpectedly survived process restart")
        return result_ok(name, started, bytes=got2.get("size", 0), details={"lost_get": lost})

    def receiver_restart_before_get(self) -> CaseResult:
        name = "receiver-restart-before-get"
        started = time.perf_counter()
        put = self.node.put(name, "copy", self.payload_size)
        self.ensure_peer_receiver_to_local()
        self.peer_cmd("close_receiver")
        got = self.remote_get(put)
        if not got.get("ok"):
            return result_fail(name, started, "get", str(got))
        return result_ok(name, started, bytes=got.get("size", 0))

    def receiver_restart_after_get(self) -> CaseResult:
        name = "receiver-restart-after-get"
        started = time.perf_counter()
        put = self.node.put("receiver-after-get-1", "copy", self.payload_size)
        first = self.remote_get(put)
        self.peer_cmd("close_receiver")
        put2 = self.node.put("receiver-after-get-2", "copy", self.payload_size)
        second = self.remote_get(put2)
        if not first.get("ok") or not second.get("ok"):
            return result_fail(name, started, "get", "receiver failed before or after restart")
        return result_ok(
            name,
            started,
            bytes=int(first.get("size", 0)) + int(second.get("size", 0)),
            details={"note": "cleanup-before-crash cannot be intercepted without connector instrumentation"},
        )

    def bidirectional(self) -> CaseResult:
        name = "bidirectional-two-connectors"
        started = time.perf_counter()
        peer_sender = self.peer_cmd("ensure_sender")
        if not peer_sender.get("ok"):
            return result_fail(name, started, "peer-sender", str(peer_sender))
        peer_ep = peer_sender["endpoint"]
        self.node.ensure_receiver(peer_ep["host"], int(peer_ep["zmq_port"]))
        self.ensure_peer_receiver_to_local()
        local_put = self.node.put("bidi-a-to-b", "copy", self.payload_size)
        remote_put = self.peer_cmd("put", key="bidi-b-to-a", kind="copy", size=self.payload_size)
        remote_get = self.peer_cmd("get", key=local_put["key"], expected_md5=local_put["md5"])
        local_get = self.node.get(remote_put["key"], remote_put["md5"], from_stage="a", to_stage="b")
        if not remote_get.get("ok") or not local_get.get("ok"):
            return result_fail(name, started, "get", "one direction failed", details={"a": local_get, "b": remote_get})
        return result_ok(name, started, bytes=int(remote_get.get("size", 0)) + int(local_get.get("size", 0)))

    def pool_pressure(self) -> CaseResult:
        name = "pool-exhaustion-then-cleanup-recovery"
        started = time.perf_counter()
        kept: list[str] = []
        size = max(1024 * 1024, self.payload_size)
        failed = None
        for idx in range(64):
            put = self.node.put(f"pool-pressure-{idx}", "copy", size)
            if not put["ok"]:
                failed = idx
                break
            kept.append(put["key"])
        if failed is None:
            return result_fail(name, started, "put", "pool did not exhaust within 64 retained payloads")
        for key in kept:
            self.node.sender.cleanup(key, from_stage="a", to_stage="b")
        recovery = self.node.put("pool-pressure-recovery", "copy", min(size, self.payload_size))
        if not recovery["ok"]:
            return result_fail(name, started, "recovery-put", "put failed after cleanup", details={"failed_at": failed})
        self.node.sender.cleanup(recovery["key"], from_stage="a", to_stage="b")
        return result_ok(name, started, details={"exhausted_at": failed, "retained": len(kept)})

    def wrong_sender_port(self) -> CaseResult:
        name = "wrong-sender-port-expected-timeout"
        started = time.perf_counter()
        put = self.node.put(name, "copy", self.payload_size)
        wrong_metadata = {"source_host": self.node.local_host, "source_port": self.node.sender_zmq_port + 77}
        self.ensure_peer_receiver_to_local()
        got = self.peer_cmd("get", key=put["key"], expected_md5=put["md5"], metadata=wrong_metadata)
        self.node.sender.cleanup(put["key"], from_stage="a", to_stage="b")
        if got.get("ok"):
            return result_fail(name, started, "get", "get unexpectedly succeeded through wrong sender port")
        return result_fail(name, started, "query", str(got), expected=True)

    def duplicate_put(self) -> CaseResult:
        name = "duplicate-put-same-key-overwrites-old-buffer"
        started = time.perf_counter()
        first = self.node.put("duplicate-key", "copy", self.payload_size)
        second = self.node.put("duplicate-key", "bytes", self.payload_size)
        if not first["ok"] or not second["ok"]:
            return result_fail(name, started, "put", "duplicate put setup failed")
        got = self.remote_get(second)
        if not got.get("ok"):
            return result_fail(name, started, "get", str(got))
        return result_ok(
            name,
            started,
            bytes=got.get("size", 0),
            details={"first_md5": first["md5"], "second_md5": second["md5"]},
        )


def scenario_map(runner: ScenarioRunner) -> dict[str, Callable[[], CaseResult]]:
    return {
        "basic-copy": lambda: runner.one_way("basic-copy", "copy"),
        "basic-zerocopy": lambda: runner.one_way("basic-zerocopy", "zerocopy"),
        "object-payload": lambda: runner.one_way("object-payload", "object"),
        "bytes-payload": lambda: runner.one_way("bytes-payload", "bytes"),
        "many-requests-unique-keys": runner.many_unique,
        "two-consumers-different-keys": runner.two_consumers_different_keys,
        "same-key-sequential-consumers": runner.same_key_sequential,
        "same-key-concurrent-consumers": runner.same_key_concurrent,
        "sender-restart-same-port": runner.sender_restart_same_port,
        "receiver-restart-before-get": runner.receiver_restart_before_get,
        "receiver-restart-after-get": runner.receiver_restart_after_get,
        "bidirectional-two-connectors": runner.bidirectional,
        "pool-exhaustion-then-cleanup-recovery": runner.pool_pressure,
        "wrong-sender-port-expected-timeout": runner.wrong_sender_port,
        "duplicate-put-same-key-overwrites-old-buffer": runner.duplicate_put,
    }


def run_node_a(args: argparse.Namespace, ports: Ports) -> int:
    load_runtime_deps()
    node = YuanrongNode(
        name="node-a",
        local_host=args.local_host,
        ports=ports,
        pool_size=args.pool_size_mb * 1024 * 1024,
    )
    peer = PeerChannel(bind=True, host=args.ctrl_bind_host, port=ports.ctrl)
    results: list[CaseResult] = []
    try:
        hello = peer.recv(timeout_s=args.startup_timeout_s)
        if hello.get("cmd") != "hello":
            raise RuntimeError(f"unexpected peer hello: {hello}")
        peer.send({"ok": True, "cmd": "hello-ack"})
        runner = ScenarioRunner(node, peer, args.peer_host, ports.b_zmq, args.payload_size_mb * 1024 * 1024)
        mapping = scenario_map(runner)
        for name in args.scenarios:
            fn = mapping.get(name)
            if fn is None:
                results.append(result_fail(name, time.perf_counter(), "config", "unknown scenario"))
                continue
            print(f"[node-a] running {name}", flush=True)
            started = time.perf_counter()
            try:
                results.append(fn())
            except Exception as exc:
                results.append(result_fail(name, started, "scenario", repr(exc)))
        try:
            peer.request({"cmd": "stop"}, timeout_s=30)
        except Exception as exc:
            results.append(result_fail("peer-stop", time.perf_counter(), "control", repr(exc)))
    finally:
        node.close_all()
        peer.close()
    print_summary(results)
    return 1 if any(r.status == "FAIL" for r in results) else 0


def run_node_b(args: argparse.Namespace, ports: Ports) -> int:
    load_runtime_deps()
    node = YuanrongNode(
        name="node-b",
        local_host=args.local_host,
        ports=ports,
        pool_size=args.pool_size_mb * 1024 * 1024,
    )
    peer = PeerChannel(bind=False, host=args.ctrl_connect_host, port=ports.ctrl)
    try:
        peer.send({"cmd": "hello", "host": args.local_host})
        ack = peer.recv(timeout_s=args.startup_timeout_s)
        if not ack.get("ok"):
            raise RuntimeError(f"hello failed: {ack}")
        while True:
            msg = peer.recv(timeout_s=args.command_timeout_s)
            resp = handle_peer_command(node, msg)
            peer.send(resp)
            if resp.get("stop"):
                break
    finally:
        node.close_all()
        peer.close()
    return 0


def print_summary(results: list[CaseResult]) -> None:
    payload = {"results": [r.to_dict() for r in results]}
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    failed = [r for r in results if r.status == "FAIL"]
    print(
        f"[summary] pass={sum(r.status == 'PASS' for r in results)} "
        f"expected_fail={sum(r.status == 'EXPECTED_FAIL' for r in results)} "
        f"unsupported={sum(r.status == 'UNSUPPORTED' for r in results)} "
        f"fail={len(failed)}",
        flush=True,
    )


def run_ssh(args: argparse.Namespace, ports: Ports) -> int:
    script = "tests/distributed/omni_connectors/yuanrong_cross_node_scenarios.py"
    scenario_arg = ",".join(args.scenarios)
    common = [
        args.remote_python,
        script,
        "--transport",
        args.transport,
        "--pool-size-mb",
        str(args.pool_size_mb),
        "--payload-size-mb",
        str(args.payload_size_mb),
        "--scenarios",
        scenario_arg,
        "--a-zmq-port",
        str(ports.a_zmq),
        "--a-rpc-port",
        str(ports.a_rpc),
        "--b-zmq-port",
        str(ports.b_zmq),
        "--b-rpc-port",
        str(ports.b_rpc),
        "--ctrl-port",
        str(ports.ctrl),
    ]
    node_a_cmd = [
        "cd",
        args.remote_workdir,
        "&&",
        *common,
        "--role",
        "node-a",
        "--local-host",
        args.node_a_host,
        "--peer-host",
        args.node_b_host,
    ]
    node_b_cmd = [
        "cd",
        args.remote_workdir,
        "&&",
        *common,
        "--role",
        "node-b",
        "--local-host",
        args.node_b_host,
        "--ctrl-connect-host",
        args.node_a_host,
    ]
    proc_a = subprocess.Popen(["ssh", args.ssh_node_a, shlex.join(node_a_cmd)])
    time.sleep(args.ssh_start_delay_s)
    proc_b = subprocess.Popen(["ssh", args.ssh_node_b, shlex.join(node_b_cmd)])
    code_b = proc_b.wait()
    code_a = proc_a.wait()
    return code_a or code_b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["node-a", "node-b", "ssh"], required=True)
    parser.add_argument("--transport", choices=["cpu-rdma", "ascend", "gpu"], default="cpu-rdma")
    parser.add_argument("--local-host", default="")
    parser.add_argument("--peer-host", default="")
    parser.add_argument("--node-a-host", default="")
    parser.add_argument("--node-b-host", default="")
    parser.add_argument("--ssh-node-a", default="")
    parser.add_argument("--ssh-node-b", default="")
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument("--remote-workdir", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--ctrl-bind-host", default="*")
    parser.add_argument("--ctrl-connect-host", default="")
    parser.add_argument("--pool-size-mb", type=int, default=256)
    parser.add_argument("--payload-size-mb", type=int, default=8)
    parser.add_argument("--a-zmq-port", type=int, default=15500)
    parser.add_argument("--a-rpc-port", type=int, default=15600)
    parser.add_argument("--b-zmq-port", type=int, default=15510)
    parser.add_argument("--b-rpc-port", type=int, default=15610)
    parser.add_argument("--ctrl-port", type=int, default=15700)
    parser.add_argument("--startup-timeout-s", type=float, default=120.0)
    parser.add_argument("--command-timeout-s", type=float, default=300.0)
    parser.add_argument("--ssh-start-delay-s", type=float, default=3.0)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    args = parser.parse_args()
    args.scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    return args


def main() -> int:
    args = parse_args()
    if args.transport != "cpu-rdma":
        print(f"[SKIP] transport={args.transport} is reserved for future implementation; cpu-rdma is implemented.")
        return 0
    ports = Ports(
        a_zmq=args.a_zmq_port,
        a_rpc=args.a_rpc_port,
        a_recv_rpc=args.a_rpc_port + 20,
        b_zmq=args.b_zmq_port,
        b_rpc=args.b_rpc_port,
        b_recv_rpc=args.b_rpc_port + 20,
        ctrl=args.ctrl_port,
    )
    if args.role == "ssh":
        missing = [k for k in ("ssh_node_a", "ssh_node_b", "node_a_host", "node_b_host") if not getattr(args, k)]
        if missing:
            print(f"[ERROR] --role ssh missing required args: {', '.join('--' + m.replace('_', '-') for m in missing)}")
            return 2
        return run_ssh(args, ports)
    if not args.local_host:
        print("[ERROR] --local-host is required for node roles")
        return 2
    if args.role == "node-a":
        if not args.peer_host:
            print("[ERROR] --peer-host is required for node-a")
            return 2
        return run_node_a(args, ports)
    if not args.ctrl_connect_host:
        print("[ERROR] --ctrl-connect-host is required for node-b")
        return 2
    return run_node_b(args, ports)


if __name__ == "__main__":
    raise SystemExit(main())
