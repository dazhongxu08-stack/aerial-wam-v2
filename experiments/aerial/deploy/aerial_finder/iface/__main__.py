"""python -m aerial_finder.iface {parse,serve,health}"""
from __future__ import annotations

import argparse
import json
import sys

from .gateway import build_facade, serve
from .schema import ParseIn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rt-api")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("text")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8100)
    sub.add_parser("health")
    args = ap.parse_args(argv)
    facade = build_facade(getattr(args, "config", None), load_parser=args.cmd in ("parse", "serve"))
    if args.cmd == "parse":
        print(json.dumps(facade.parser.parse(ParseIn(args.text)).to_dict(), ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "health":
        print(json.dumps(facade.health().to_dict(), ensure_ascii=False, indent=1))
        return 0
    httpd = serve(facade, args.host, args.port)
    print(f"rt-api {args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
