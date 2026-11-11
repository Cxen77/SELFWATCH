"""Fetch OSNet MSMT17 weights from the author's pinned Hugging Face release."""
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = "kaiyangzhou/osnet"
REVISION = "a5c5cc037c24235cda3b21085b93ad77c9616224"
NAME = "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"


def main():
    directory = Path("weights/osnet")
    directory.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{NAME}"
    path = directory / "osnet_x1_0_msmt17.pth"
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}") as response:
        record = next(item for item in json.load(response) if item["path"] == NAME)
    if not path.exists():
        temporary = path.with_suffix(".part")
        urllib.request.urlretrieve(url, temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if temporary.stat().st_size != record["size"] or digest != record["lfs"]["oid"]:
            raise ValueError("Downloaded checkpoint failed author-release hash validation")
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["lfs"]["oid"]:
        raise ValueError("Existing OSNet checkpoint differs from the pinned author release")
    provenance = dict(url=url, revision=REVISION, sha256=digest, bytes=path.stat().st_size,
                      training="MSMT17 combineall", input_height_width=[256,128])
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
