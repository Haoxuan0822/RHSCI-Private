"""Check saved evaluation MDL paths, material exports and local texture references."""

import json
import re
from pathlib import Path

ROOT = Path("/root/workspace/Haoxuan/RHSCI-runtime")
DATA = ROOT / "external/SIMPLE-official/data"
TASK = "G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
EVAL = ROOT / "SIMPLE-eval-20261103/snapshot-fb67cd9a" / TASK
records = []
missing = []
textures = set()
for level in ("dr-level-0", "dr-level-1", "dr-level-2", "dr-level-hard"):
    for line in (EVAL / level / "meta/episodes.jsonl").read_text().splitlines():
        episode = json.loads(line)
        material = json.loads(episode["environment_config"])["dr_state_dict"]["material"]
        for role in ("table_material", "ground_material"):
            info = material[role]
            path = Path(info["path"])
            if not path.is_absolute():
                path = DATA / info["path"].removeprefix("data/")
            record = dict(
                level=level,
                episode=episode["episode_index"],
                role=role,
                path=str(path),
                name=info["name"],
                exists=path.is_file(),
            )
            if not path.is_file():
                missing.append(record)
            else:
                content = path.read_text()
                record["export_exists"] = bool(
                    re.search(r"export\s+material\s+" + re.escape(info["name"]) + r"\s*\(", content)
                )
                if not record["export_exists"]:
                    missing.append(dict(record, reason="missing exported material"))
                for relative in re.findall(r'texture_2d\(\s*"([^"]+)"', content):
                    texture = path.parent / relative
                    textures.add(str(texture))
                    if not texture.is_file():
                        missing.append(dict(record, reason="missing texture", texture=str(texture)))
            records.append(record)
report = dict(
    episodes=40,
    material_uses=len(records),
    unique_mdl_files=len({r["path"] for r in records}),
    unique_texture_paths=len(textures),
    missing=missing,
    records=records,
    scope="Saved MDL paths, exported material names and texture_2d file references; actual USD/MDL loading also requires renderer log review.",
)
destination = ROOT / "logs/pi05-between-material-paths.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
raise SystemExit(bool(missing))
