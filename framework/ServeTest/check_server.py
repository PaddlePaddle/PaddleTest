import sys
import time
import argparse
import requests
import os


def parse_args():
    parser = argparse.ArgumentParser(description="HTTP health check")
    parser.add_argument(
        "address",
        help="ip:port, e.g. 127.0.0.1:8080"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=420,
        help="max wait time in seconds (default: 420)"
    )
    return parser.parse_args()


def dump_log(path):
    print(f"\n========== {path} ==========")
    if not os.path.exists(path):
        print(f"{path} not found")
        return
    try:
        with open(path, "r", errors="ignore") as f:
            print(f.read())
    except Exception as e:
        print(f"failed to read {path}: {e}")


def main():
    args = parse_args()

    if ":" not in args.address:
        print(f"Invalid address format: {args.address}, expected ip:port")
        sys.exit(2)

    ip, port = args.address.split(":", 1)
    url = f"http://{ip}:{port}/health"

    start_time = time.time()

    print(f"Start health checking: {url}")
    print(f"Timeout: {args.timeout}s, interval: 2s")

    while True:
        elapsed = time.time() - start_time
        if elapsed > args.timeout:
            print("\nERROR: 服务启动超时")

            dump_log("server.log")
            dump_log("log/workerlog.0")

            sys.exit(1)

        try:
            resp = requests.get(url, timeout=2)
            status = resp.status_code
            print(f"[{int(elapsed)}s] status_code={status}")

            if status == 200:
                print("服务启动成功")
                sys.exit(0)

        except requests.RequestException as e:
            print(f"[{int(elapsed)}s] request failed")
        time.sleep(2)


if __name__ == "__main__":
    main()
