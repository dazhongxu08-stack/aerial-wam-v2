#!/usr/bin/env python3
"""TCP relay because iPhone hotspot often blocks H12 ↔ Orin.

H12 → Mac:8092 → Orin:8092
"""
from __future__ import annotations

import argparse
import select
import socket
import threading


def pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            r, _, _ = select.select([a], [], [], 30)
            if not r:
                continue
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        try:
            a.shutdown(socket.SHUT_RD)
        except OSError:
            pass
        try:
            b.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def handle(client: socket.socket, dest: tuple[str, int]) -> None:
    upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        upstream.settimeout(3)
        upstream.connect(dest)
        upstream.settimeout(None)
        t1 = threading.Thread(target=pipe, args=(client, upstream), daemon=True)
        t2 = threading.Thread(target=pipe, args=(upstream, client), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except OSError as exc:
        print("upstream", dest, exc)
    finally:
        client.close()
        upstream.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--listen", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--dest-host", default="172.20.10.4")
    p.add_argument("--dest-port", type=int, default=8092)
    args = p.parse_args()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.listen, args.port))
    srv.listen(32)
    dest = (args.dest_host, args.dest_port)
    print(f"relay {args.listen}:{args.port} -> {dest[0]}:{dest[1]}")
    while True:
        client, addr = srv.accept()
        print("client", addr)
        threading.Thread(target=handle, args=(client, dest), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
