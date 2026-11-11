"""Six preliminary assistant visual labels; requires human adjudication."""
import argparse
import json


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory / "data.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["sample"] != "000398":
        raise ValueError("Labels only apply to visually inspected sample 000398")
    boxes = {("rank0",0):[0,114,162,366], ("rank1",0):[441,75,600,436],
             ("rank0",16):[124,175,458,390], ("rank1",16):[166,155,456,501],
             ("rank0",30):[74,165,370,388], ("rank1",30):[246,110,514,519]}
    records = []
    for observation in data["frames"]:
        frame = observation["frame"]
        box = boxes.get((frame["camera_id"], frame["frame_index"]))
        record = dict(observation_id=observation["observation_id"], frame=frame,
                      reviewed=box is not None, exhaustive_labels=["person"] if box else [], marks=[], issues=[])
        if box:
            if frame["width"] != 600 or frame["height"] != 600:
                raise ValueError("Image dimensions differ from inspected frames")
            record["marks"] = [dict(label="person", kind="box", xyxy=box, subject="shopper A",
                                      visibility="visible", source="assistant_visual_pilot")]
            record["issues"] = ["Hands partly occluded or side ambiguous; hand ground truth not assigned"]
        records.append(record)
    review = dict(schema_version=1, run_id=data["run_id"], sample=data["sample"],
                  annotation_source="Preliminary assistant visual inspection; approximate person boxes; requires independent human adjudication",
                  coordinate_space="original_image_pixels", frames=records)
    (args.directory / "initial-review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
    data["initial_review"] = review
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    print("Saved six approximate person-box labels. No hand/finger ground truth invented.")


if __name__ == "__main__":
    main()
