"""Render recorded detector evidence; never run models or open sealed images."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def build(output):
    data=json.loads((output/'data.json').read_text(encoding='utf-8'))
    if any(frame['partition'] not in ('development','validation') for frame in data['frames']):
        raise ValueError('Review must not expose final-test frames')
    assets=Path(__file__).parent/'overhead_assets'
    html=(assets/'index.html').read_text(encoding='utf-8')
    for name in ('overhead.js','overhead.css'):
        content=(assets/name).read_bytes()
        (output/name).write_bytes(content)
        html=html.replace(f'"{name}"',f'"{name}?v={sha256(content).hexdigest()[:12]}"')
    (output/'index.html').write_text(html,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    build(parser.parse_args().run)


if __name__=='__main__':main()
