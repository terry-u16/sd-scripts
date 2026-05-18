import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--dir",
    type=str,
    required=True,
    help="Directory containing the .txt files to modify",
)
parser.add_argument("-t", "--tag", type=str, nargs="+", help="Tags to remove")
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print files to be removed without deleting them",
)
args = parser.parse_args()
to_remove = set()

for path in Path(args.dir).glob("*.txt"):
    with open(path, "r") as f:
        tags = f.read()

    tag_list = [t.strip() for t in tags.split(",")]

    for t in args.tag:
        if t in tag_list:
            to_remove.add(path.name.split(".")[0])


for name in to_remove:
    for path in Path(args.dir).glob("*"):
        if path.name.startswith(name):
            if args.dry_run:
                print(path)
            else:
                path.unlink()
