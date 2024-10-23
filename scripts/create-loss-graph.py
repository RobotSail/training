# SPDX-License-Identifier: Apache-2.0
# Standard
from argparse import ArgumentParser, Namespace
from base64 import b64encode
from io import BytesIO
from pathlib import Path
import json

# Third Party
from matplotlib import pyplot as plt


def create_b64_data(log_file: Path) -> str:
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
    return imgb64


def create_md_file(b64_data: str, output_file: Path | None):
    content = f"""## Training Performance\n

![Training Performance](data:image/png;base64,{b64_data})
"""
    if not output_file:
        print(content)
    else:
        output_file.write_text(content, encoding="utf-8")


def main(args: Namespace):
    imgb64 = create_b64_data(args.log_file)

    output_file = Path(args.output_file) if args.output_file else None
    if output_file:
        output_file.write_text(imgb64, encoding="utf-8")
    else:
        # just print the file without including a newline, this way it can be piped
        print(imgb64, end="")


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image")
    image_parser.add_argument("--log-file", type=str, required=True)
    image_parser.add_argument("--output-file", type=str, default=None)

    markdown_parser = subparsers.add_parser("markdown")
    markdown_parser.add_argument("--log-file", type=str, required=True)
    markdown_parser.add_argument("--output-file", type=str, default=None)

    args = parser.parse_args()
    match args.command:
        case "image":
            print("creating image")
            main(args)
        case "markdown":
            print("creating md file")
            b64_data = create_b64_data(log_file=Path(args.log_file))
            create_md_file(b64_data=b64_data, output_file=Path(args.output_file))
        case _:
            raise ValueError(f"Unknown command: {args.command}")
