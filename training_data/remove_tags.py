import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--dir",
    type=str,
    required=True,
    help="Directory containing the .txt files to modify",
)
parser.add_argument(
    "-t", "--tag", type=str, nargs="+", help="Tags to remove from the .txt files"
)
args = parser.parse_args()
patterns = list(map(lambda t: re.compile(rf"(?<!\w){t}(\s*,\s*|\s*$)"), args.tag))

for path in Path(args.dir).glob("*.txt"):
    with open(path, "r") as f:
        tags = f.read()

    tag_list = [t.strip() for t in tags.split(",")]
    tag_list = [t for t in tag_list if t not in args.tag]
    tags = ", ".join(tag_list)

    with open(path, "w") as f:
        f.write(tags)
