import argparse

from src.web.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the platform-neutral Ferry Intelligence browser dashboard."
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    serve(args.port)


if __name__ == "__main__":
    main()
