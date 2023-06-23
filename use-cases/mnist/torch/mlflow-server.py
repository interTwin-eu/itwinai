import argparse
import sys
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manage MLFLow server."
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        help="MLFlow server port.",
        default=5000
    )
    parser.add_argument(
        "-m", "--mode",
        help="Start or kill MlFlow server.",
        type=str,
        default='run',
        choices=('run', 'kill')
    )
    parser.add_argument(
        "--path",
        type=str,
        help="MLFlow server storage path (backend-store-uri).",
        default=None
    )
    args = parser.parse_args()

    if args.mode == 'kill':
        # Kill server
        print(f"Killing MLFlow server on localhost port {args.port}")
        subprocess.run(
            f"kill -9 $(lsof -t -i:{args.port})",
            shell=True,
            check=True,
            stderr=subprocess.DEVNULL
        )
        sys.exit()

    # Start server
    print("Starting MLFlow server")
    subprocess.Popen(
        ('mlflow server --backend-store-uri '
         f'file:{args.path}').split(),
        stderr=subprocess.DEVNULL
    )
