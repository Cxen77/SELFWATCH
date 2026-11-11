"""Fetch a bounded training-only pilot from the official RetailAction tar stream.

No archive extraction paths are trusted. The full 26 GB archive is not saved.
Original metadata and license are retained. This prefix is not a random sample.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

REPO = "standard-cognition/RetailAction"


class BoundedReader:
    def __init__(self, response, limit):
        self.response, self.limit, self.count = response, limit, 0

    def read(self, size=-1):
        if self.count >= self.limit:
            raise RuntimeError("Download limit reached before complete samples; see partial manifest")
        size = min(size if size >= 0 else self.limit, self.limit - self.count)
        data = self.response.read(size)
        self.count += len(data)
        return data


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "SELFWATCH-research/0.1"}), timeout=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/retailaction-pilot"))
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-mib", type=int, default=256)
    args = parser.parse_args()
    if args.samples < 1 or args.max_mib < 1:
        parser.error("Positive sample and byte limits required")
    args.output.mkdir(parents=True, exist_ok=False)
    with fetch(f"https://huggingface.co/api/datasets/{REPO}") as response:
        revision = json.load(response)["sha"]
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}"
    for name in ("README.md", "LICENSE"):
        with fetch(f"{base}/{name}") as response:
            (args.output / name).write_bytes(response.read())
    (args.output / "NOTICE").write_text(
        "This Dataset is licensed under the Standard.AI Dataset License, Copyright © Standard Corp.  All Rights Reserved.\n",
        encoding="utf-8")
    manifest = {"repository": REPO, "revision": revision, "split": "train", "samples_requested": args.samples,
        "selection": "first complete sample directories in tar order; development only, not representative",
        "downloaded_at": datetime.now(timezone.utc).isoformat(), "files": [], "status": "partial"}
    reader = None
    groups = {}
    try:
        with fetch(f"{base}/data/train.tar") as response:
            reader = BoundedReader(response, args.max_mib * 1024 ** 2)
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    relative = PurePosixPath(member.name)
                    if relative.is_absolute() or ".." in relative.parts or any(":" in p or "\\" in p for p in relative.parts):
                        raise ValueError("Unsafe archive path")
                    if relative.name not in ("rank0_video.mp4", "rank1_video.mp4", "metadata.json"):
                        continue
                    group = str(relative.parent)
                    if group not in groups and len(groups) >= args.samples:
                        continue
                    if member.size > 32 * 1024 ** 2:
                        raise ValueError("Unexpectedly large individual sample file")
                    target = args.output.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise ValueError("Duplicate archive member")
                    data = archive.extractfile(member).read()
                    if len(data) != member.size:
                        raise ValueError("Truncated archive member")
                    target.write_bytes(data)
                    manifest["files"].append({"path": str(target.relative_to(args.output)),
                        "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                    groups.setdefault(group, set()).add(relative.name)
                    if len(groups) == args.samples and all(len(files) == 3 for files in groups.values()):
                        manifest["status"] = "complete"
                        break
        if manifest["status"] != "complete":
            raise RuntimeError("Archive ended before requested sample set completed")
    finally:
        manifest["stream_bytes_read"] = reader.count if reader else 0
        manifest["complete_samples"] = sum(len(files) == 3 for files in groups.values())
        (args.output / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
