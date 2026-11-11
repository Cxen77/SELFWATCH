"""Download the OpenMMLab RTMW lightweight ONNX export used by RTMLib."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip"


def main():
    target = Path("weights/v2/rtmw")
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "rtmw-lightweight.zip"
    if not archive.exists():
        partial = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, partial)
        partial.replace(archive)
    with zipfile.ZipFile(archive) as source:
        models = [name for name in source.namelist() if name.endswith(".onnx")]
        if len(models) != 1:
            raise ValueError("Expected one ONNX model")
        model = target / "wholebody133.onnx"
        model.write_bytes(source.read(models[0]))
    metadata = dict(source_url=URL, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    input_wh=[192, 256], keypoint_schema="coco-wholebody133",
                    documentation="https://github.com/Tau-J/rtmlib", model=str(model))
    (target / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
