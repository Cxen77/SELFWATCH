"""Fetch the bounded i.c.sens camera-pair identity audit inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import zipfile

from scripts.validate_icsens_identity import GT_SHA256, HANDOFF_INDEXES, INDEXES
from selfwatch_v2.http_range import HTTPRangeReader
from selfwatch_v2.identity_study import digest


DATA = "https://data.uni-hannover.de"
GT_URL = (DATA + "/dataset/733e368e-a481-4f85-8aac-2abcc35a0673/"
          "resource/bef2b16c-e8cc-4461-a63f-0be46239d5e0/download/ground_truth.zip")
VIEW_BYTES = {1: 11072806117, 2: 5489599806}


def fetch_view(root, camera, names, sequence=0):
    url = f"{DATA}/vault/icsens/nguyen/multi-view-pedestrian-tracking/view{camera}.zip"
    destination = root / f"view{camera}"
    destination.mkdir(exist_ok=True)
    pending = [(name, destination / Path(name).name) for name in names
               if not (destination / Path(name).name).is_file()]
    if not pending:
        return
    with HTTPRangeReader(url, max_bytes=144 * 1024**2,
                         block_size=1024**2) as reader, zipfile.ZipFile(reader) as archive:
        if reader.size != VIEW_BYTES[camera]:
            raise ValueError("Unexpected i.c.sens video archive size")
        for name, target in pending:
            item = archive.getinfo(f"view{camera}/{sequence:04}/{name}")
            if not 0 < item.file_size <= 12 * 1024**2:
                raise ValueError("Unexpected i.c.sens member size")
            data = archive.read(item)
            part = target.with_suffix(".part")
            part.write_bytes(data)
            part.replace(target)
        print(f"view{camera}: {len(pending)} members, {reader.bytes_read} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence", type=int, choices=(0, 6), default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    partition = args.output / "sequence.txt"
    if partition.exists() and partition.read_text().strip() != str(args.sequence):
        raise ValueError("Choose a different output folder for a different sequence")
    if not partition.exists() and args.sequence != 0 and any(args.output.glob("view*")):
        raise ValueError("A new sequence requires a fresh output folder")
    partition.write_text(f"{args.sequence}\n")
    labels = args.output / "ground_truth.zip"
    if not labels.is_file():
        part = labels.with_suffix(".part")
        subprocess.run(["curl.exe", "--fail", "--silent", "--show-error", "--location",
                        "--max-time", "60", "--max-filesize", str(4 * 1024**2), GT_URL,
                        "--output", str(part)], check=True)
        if digest(part) != GT_SHA256:
            raise ValueError("i.c.sens labels differ from the pinned archive")
        part.replace(labels)
    if digest(labels) != GT_SHA256:
        raise ValueError("i.c.sens labels differ from the pinned archive")

    indices = HANDOFF_INDEXES if args.sequence == 6 else INDEXES
    names = ["time_stamp.csv", *(f"left/{index:06}.png" for index in indices)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(fetch_view, args.output, camera, names[i:i + 12], args.sequence)
                for camera in VIEW_BYTES for i in range(0, len(names), 12)]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
