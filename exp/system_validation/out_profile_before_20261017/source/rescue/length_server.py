"""Run a CPU length-policy service; Target/Draft models stay in the engine."""
import argparse
import json
from pathlib import Path

import torch

from .policy import RiskModel
from .transport import LengthServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--ready-file', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = RiskModel.load(args.checkpoint, 'cpu')
    with LengthServer((args.host, args.port), model) as server:
        ready = dict(host=server.server_address[0], port=server.server_address[1], checkpoint=args.checkpoint)
        if args.ready_file:
            args.ready_file.write_text(json.dumps(ready)+'\n')
        print(json.dumps(ready), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
