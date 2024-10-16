# SPDX-License-Identifier: Apache-2.0
# Standard
from argparse import ArgumentParser, Namespace
from base64 import b64encode
from io import BytesIO
from pathlib import Path
import json

# Third Party
from matplotlib import pyplot as plt


def main(args: Namespace):
    log_file = Path(args.log_file)
    if not log_file.exists():
        raise FileNotFoundError(f'log file "{args.log_file}" does not exist')

    if not log_file.is_file():
        raise RuntimeError(f'log file cannot be a directory: "{log_file}"')

    with open(Path(log_file), "r", encoding="utf-8") as infile:
        contents = [json.loads(l) for l in infile.read().splitlines()]

    loss_data = [item["total_loss"] for item in contents if "total_loss" in item]

    # create the plot
    plt.figure()
    plt.plot(loss_data)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training performance over fixed dataset")

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    imgb64 = b64encode(buf.read()).decode("utf-8")

    output_file = Path(args.output_file) if args.output_file else None
    if output_file:
        output_file.write_text(imgb64, encoding="utf-8")
    else:
        # just print the file without including a newline, this way it can be piped
        print(imgb64, end="")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-file", type=str)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()
    main(args)
