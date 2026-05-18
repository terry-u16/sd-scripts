from __future__ import annotations

import argparse
from pathlib import Path


def truncate_to_first_line(directory: Path) -> None:
    if not directory.is_dir():
        raise NotADirectoryError(f"ディレクトリではありません: {directory}")

    txt_files = sorted(directory.glob("*.txt"))

    for path in txt_files:
        text = path.read_text(encoding="utf-8")
        first_line = text.splitlines(keepends=True)[0] if text else ""
        path.write_text(first_line, encoding="utf-8")
        print(f"processed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="指定ディレクトリ内の .txt ファイルを、1行目だけ残して上書きします。"
    )
    parser.add_argument("directory", type=Path, help="対象ディレクトリ")
    args = parser.parse_args()

    truncate_to_first_line(args.directory)


if __name__ == "__main__":
    main()
