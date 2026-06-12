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
parser.add_argument(
    "-i", "--index", type=int, help="Tag index to replace"
)
args = parser.parse_args()
index = args.index

for path in Path(args.dir).glob("*.txt"):
    with open(path, "r") as f:
        tags = f.read()

    tag_list = [t.strip() for t in tags.split(",")]
    tag_list[index] = f"@{tag_list[index]}"
    tags = ", ".join(tag_list)

    with open(path, "w") as f:
        f.write(tags)
