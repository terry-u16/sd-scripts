import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--dir",
    dest="dir_path",
    type=str,
    required=True,
    help="Directory containing the .txt files to modify",
)
parser.add_argument("-f", "--from", dest="from_tag", type=str, required=True, help="Tag to replace from")
parser.add_argument("-t", "--to", dest="to_tag", type=str, required=True, help="Tag to replace to")
args = parser.parse_args()

for path in Path(args.dir_path).glob("*.txt"):
    with open(path, "r", encoding="utf-8") as f:
        tags = f.read()

    tag_list = [t.strip() for t in tags.split(",")]
    tag_list = [(args.to_tag if t == args.from_tag else t) for t in tag_list]
    tags = ", ".join(tag_list)

    with open(path, "w", encoding="utf-8") as f:
        f.write(tags)
