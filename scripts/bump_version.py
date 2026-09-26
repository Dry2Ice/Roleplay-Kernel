import json
import pathlib


def main() -> None:
    version = "0.1.1"

    pyproject = pathlib.Path("pyproject.toml")
    text = pyproject.read_text(encoding="utf-8")
    text = text.replace('version = "0.1.0"', f'version = "{version}"', 1)
    pyproject.write_text(text, encoding="utf-8")

    sidecar = pathlib.Path("src/roleplay_kernel/sidecar.py")
    text = sidecar.read_text(encoding="utf-8")
    text = text.replace(
        'SIDECAR_VERSION = "0.1.0"',
        f'SIDECAR_VERSION = "{version}"',
        1,
    )
    sidecar.write_text(text, encoding="utf-8")

    manifest = pathlib.Path("manifest.json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["version"] = version
    manifest.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("bumped to", version)


if __name__ == "__main__":
    main()
