import argparse
import logging
import signal
import sys

from relay.logger import ShotLogger
from relay.server import make_server
from relay.sim_client import SimClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("relay")


def main():
    parser = argparse.ArgumentParser(description="Golf Launch Monitor relay service")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP listen host")
    parser.add_argument("--port", type=int, default=3001, help="HTTP listen port")
    parser.add_argument("--sim-host", default="127.0.0.1", help="Sim TCP host")
    parser.add_argument("--sim-port", type=int, default=3111, help="Sim TCP port")
    parser.add_argument("--log", default="logs/shots.jsonl", help="Shot log path")
    args = parser.parse_args()

    logger = ShotLogger(args.log)
    sim = SimClient(host=args.sim_host, port=args.sim_port)
    sim.connect()

    server = make_server(args.host, args.port, sim, logger)

    log.info("Relay listening on %s:%s", args.host, args.port)
    log.info("Forwarding to sim at %s:%s", args.sim_host, args.sim_port)
    log.info("Logging shots to %s", args.log)

    def shutdown(signum, frame):
        log.info("Shutting down…")
        sim.disconnect()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sim.disconnect()
        log.info("Relay stopped")


if __name__ == "__main__":
    main()
