# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Linus Eickhoff
#
# Credit:
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

import argparse
import re
from pathlib import Path
from typing import List


def convert_github_admonitions(content: str) -> str:
    """Converts GitHub-style admonitions to sphinx md syntax.

    Args:
        content (str): The markdown content containing GitHub-style admonitions

    Returns:
        str: The markdown content with GitHub admonitions converted to sphinx md syntax

    Example:
    > [!NOTE]
    > This is a note

    Becomes:
    ```{note}
    This is a note
    ```
    """
    # Github markdown pattern
    pattern = r">\s*\[!(NOTE|WARNING|TIP|IMPORTANT|CAUTION)\][ \t]*(.*(?:\n>.*)*)"

    def replace_admonition(match):
        # same name in lowercase for sphinx-compatible admonitions
        admonition_type = match.group(1).lower()
        content_lines = match.group(2)

        cleaned_content = []
        for line in content_lines.splitlines():
            if line.strip():
                # Preserve indentation after removing > prefix
                cleaned_line = re.sub(r"^>\s?", "", line)
                cleaned_content.append(cleaned_line)

        return "```{" + f"{admonition_type}" + "}\n" + "\n".join(cleaned_content) + "\n```"

    return re.sub(pattern, replace_admonition, content)


def find_md_includes_in_rst(rst_content: str, rst_file_path: str | Path) -> List[Path]:
    """Find all .md file includes in an RST file and return their absolute paths.

    Args:
        rst_content (str): Content of the RST file to search for includes
        rst_file_path (str): Path to the RST file, used for resolving relative paths

    Returns:
        list[Path]: List of absolute Path objects pointing to included .md files
    """
    pattern = r"\.\.\s+include::\s+(.+\.md)"
    matches = re.finditer(pattern, rst_content)
    md_files = []
    rst_path = Path(rst_file_path)

    for match in matches:
        md_path = match.group(1)
        # Handle relative paths properly using Path
        abs_md_path = (rst_path.parent / md_path).resolve()
        md_files.append(abs_md_path)

    return md_files


def process_rst_files(docs_dir: str) -> None:
    """Process all RST files in the docs directory and convert referenced MD files.

    Args:
        docs_dir (str): Path to the docs directory containing RST files

    Returns:
        None: This function processes files in-place and prints status messages
    """
    docs_path = Path(docs_dir).resolve()
    processed_files = set()  # To prevent processing the same file multiple times

    for rst_file in docs_path.rglob("*.rst"):
        if rst_file.is_symlink():  # Skip symlinks to prevent loops
            print(f"Skipping symlink: {rst_file}")
            continue

        print(f"Processing RST file: {rst_file}")

        try:
            # Read RST content
            rst_content = rst_file.read_text(encoding="utf-8")
            # Find MD includes
            md_files = find_md_includes_in_rst(rst_content, rst_file)
            # Process each MD file
            for md_file in md_files:
                try:
                    if md_file in processed_files:
                        continue

                    processed_files.add(md_file)

                    if not md_file.exists():
                        print(f"Warning: Referenced MD file not found: {md_file}")
                        continue

                    print(f"Checking for admonitions in: {md_file}")
                    # assume utf8
                    md_content = md_file.read_text(encoding="utf-8")
                    # Convert admonitions
                    converted_content = convert_github_admonitions(md_content)
                    # Print to stdout where changes were made
                    if converted_content != md_content:
                        md_file.write_text(converted_content, encoding="utf-8")
                        print(f"Changed admonitions in: {md_file}")

                except Exception as e:
                    print(f"Unexpected error processing {md_file}: {str(e)}")

        except Exception as e:
            print(f"Unexpected error processing {rst_file}: {str(e)}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(
        description="Convert GitHub-style admonitions of md references to sphinx syntax"
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        required=True,
        help="Path to directory where to search for RST files",
    )
    args = parser.parse_args()

    process_rst_files(args.dir)
