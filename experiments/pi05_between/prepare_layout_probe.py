"""Create isolated layout counterfactuals without editing official or training data."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--light-intensity", type=float)
    mode.add_argument("--robot-x-offset", type=float)
    parser.add_argument(
        "--environment-from",
        type=Path,
        help="Use training episode 0 config in a lightweight eval envelope",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for child in source.iterdir():
        if child.name != "meta":
            (args.output / child.name).symlink_to(child, target_is_directory=child.is_dir())
    meta = args.output / "meta"
    meta.mkdir()
    for child in (source / "meta").iterdir():
        if child.name != "episodes.jsonl":
            (meta / child.name).symlink_to(child, target_is_directory=child.is_dir())
    original = (source / "meta/episodes.jsonl").read_bytes()
    environment_bytes = None
    replacement = None
    if args.environment_from is not None:
        environment_bytes = (args.environment_from / "meta/episodes.jsonl").read_bytes()
        replacement = next(
            json.loads(line)["environment_config"]
            for line in environment_bytes.decode().splitlines()
            if json.loads(line)["episode_index"] == 0
        )
    changes = []
    with (meta / "episodes.jsonl").open("x") as stream:
        for line in original.decode().splitlines():
            episode = json.loads(line)
            if replacement is not None and episode["episode_index"] == 0:
                episode["environment_config"] = replacement
            encoded = isinstance(episode["environment_config"], str)
            config = (
                json.loads(episode["environment_config"])
                if encoded
                else episode["environment_config"]
            )
            if args.robot_x_offset is not None:
                position = config["dr_state_dict"]["spatial"]["g1_sonic"]["position"]
                changes.append(
                    {
                        "episode": episode["episode_index"],
                        "old_x": position[0],
                        "new_x": position[0] + args.robot_x_offset,
                    }
                )
                position[0] += args.robot_x_offset
                episode["environment_config"] = json.dumps(config) if encoded else config
                stream.write(json.dumps(episode) + "\n")
                continue
            lights = config["layout"]["lights"]
            saved_lights = config["dr_state_dict"]["lighting"]
            for light in lights:
                changes.append(
                    {
                        "episode": episode["episode_index"],
                        "light": light["uid"],
                        "old": light["light_intensity"],
                        "new": args.light_intensity,
                    }
                )
                light["light_intensity"] = args.light_intensity
                saved_lights[light["uid"]]["light_intensity"] = args.light_intensity
            episode["environment_config"] = json.dumps(config) if encoded else config
            stream.write(json.dumps(episode) + "\n")
    (args.output / "probe_source.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "environment_from": str(args.environment_from) if replacement is not None else None,
                "environment_sha256": hashlib.sha256(environment_bytes).hexdigest()
                if environment_bytes is not None
                else None,
                "episodes_sha256": hashlib.sha256(original).hexdigest(),
                "changed_fields_only": ["dr_state_dict.spatial.g1_sonic.position[0]"]
                if args.robot_x_offset is not None
                else [
                    "layout.lights.*.light_intensity",
                    "dr_state_dict.lighting.*.light_intensity",
                ],
                "changes": changes,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
