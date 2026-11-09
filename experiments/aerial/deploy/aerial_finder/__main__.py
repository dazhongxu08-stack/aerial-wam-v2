"""python -m aerial_finder {serve,run,parse,mock-dog} ..."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


def _install_root(cfg_path: Path) -> Path:
    """Directory that relative weight/config paths resolve against.

    Product layout: ``$AERIAL_FINDER_HOME/config.json`` (or any config whose
    parent holds ``weights/``). Dev layout: config's parent directory.
    """
    env = os.environ.get("AERIAL_FINDER_HOME")
    if env:
        return Path(env).expanduser().resolve()
    parent = cfg_path.parent.resolve()
    if (parent / "weights").is_dir() or (parent / "bin").is_dir():
        return parent
    return parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aerial_finder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="task HTTP API")
    s.add_argument("--config", required=True)
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    r = sub.add_parser("run", help="one mission, blocking")
    r.add_argument("--config", required=True)
    r.add_argument("instruction")
    p = sub.add_parser("parse", help="parse an instruction")
    p.add_argument("instruction")
    p.add_argument("--config", default=None, help="optional finder JSON (loads parser.qwen if set)")
    m = sub.add_parser("mock-dog", help="robot-dog stand-in that logs target reports")
    m.add_argument("--host", default="0.0.0.0")
    m.add_argument("--port", type=int, default=9100)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(message)s")

    if args.cmd == "parse":
        from .mission.llm_parser import make_parse_fn
        from .mission.parser import parse_instruction
        parse_fn = parse_instruction
        if args.config:
            cfg_path = Path(args.config).expanduser().resolve()
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            parse_fn = make_parse_fn(cfg.get("parser"), _install_root(cfg_path))
        print(json.dumps(parse_fn(args.instruction).to_dict(), ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "mock-dog":
        from .comm.dog import MockDogServer
        srv = MockDogServer(args.host, args.port).start()
        print(f"mock dog on {srv.url}/api/v1/targets", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            srv.stop()
        return 0

    from .mission.llm_parser import make_parse_fn
    from .service import build, serve
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    root = _install_root(cfg_path)
    parse_fn = make_parse_fn(cfg.get("parser"), root)
    runner = build(cfg, root)
    if args.cmd == "run":
        try:
            res = runner.run(parse_fn(args.instruction))
            print(json.dumps(res, ensure_ascii=False, indent=1))
            return 0 if res.get("state") == "DONE" else 2
        finally:
            runner.backend.close()
    sc = cfg.get("server") or {}
    httpd = serve(runner, args.host or sc.get("host", "0.0.0.0"), args.port or int(sc.get("port", 8088)),
                  parse_fn=parse_fn)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        runner.abort()
    return 0


if __name__ == "__main__":
    sys.exit(main())
