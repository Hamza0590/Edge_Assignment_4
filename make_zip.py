"""
Build the submission archive ROLLNUM_SECTION.zip next to this folder.

    python make_zip.py 22i-0001_A

Included: source code, trained checkpoints (.pth), ONNX models, logs, CSV/JSON results,
figures, report, README, decisions.
Excluded: the CIFAR dataset (data/), __pycache__, temporary files - as required
("only deliverables, no binaries / exe").
"""
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
EXCLUDE_DIRS = {"data", "__pycache__", ".git", ".idea", ".vscode", ".ipynb_checkpoints"}
EXCLUDE_EXT = {".pyc", ".exe", ".dll", ".so", ".tar.gz", ".gz"}


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python make_zip.py ROLLNUM_SECTION   (e.g. 22i-0001_A)")
    name = sys.argv[1]
    dst = os.path.join(os.path.dirname(ROOT), f"{name}.zip")
    n = 0
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in EXCLUDE_DIRS]
            for f in files:
                if any(f.endswith(e) for e in EXCLUDE_EXT):
                    continue
                p = os.path.join(d, f)
                z.write(p, os.path.join(name, os.path.relpath(p, ROOT)))
                n += 1
    print(f"wrote {dst} ({n} files, {os.path.getsize(dst) / 1e6:.1f} MB)")
    with zipfile.ZipFile(dst) as z:
        bad = z.testzip()
    print("archive integrity:", "OK" if bad is None else f"CORRUPT ({bad})")


if __name__ == "__main__":
    main()
