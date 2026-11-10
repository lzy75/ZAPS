"""Render preregistered non-cherry-picked A101 panels."""
from pathlib import Path
from utils import equal_nfe_feedback as rules
from utils import historical_ffhq_gain as files


def render(root):
    from PIL import Image, ImageDraw, ImageFont
    plan = files.read(root / 'plan.json'); output = root / 'qualitative'; output.mkdir(exist_ok=True)
    font = ImageFont.load_default(); labels = ('GT', 'Observation', *rules.ARMS)
    for task in rules.TASKS:
        for sigma in rules.SIGMAS:
            canvas = Image.new('RGB', (5 * 256, 3 * 286), 'white'); draw = ImageDraw.Draw(canvas)
            for row_index, image_id in enumerate(rules.PANEL_IMAGES):
                case = next(case for case in plan['cases'] if case['task'] == task and
                            case['sigma'] == sigma and case['image_id'] == image_id)
                observation = root / 'observations' / case['key']
                paths = [observation / 'ground_truth.png', observation / 'observation.png'] + [
                    root / 'cases' / case['key'] / arm / 'output.png' for arm in rules.ARMS]
                for column, (label, path) in enumerate(zip(labels, paths)):
                    image = Image.open(path).convert('RGB').resize((256, 256))
                    x, y = column * 256, row_index * 286
                    canvas.paste(image, (x, y)); draw.text((x + 5, y + 261), label, fill='black', font=font)
                draw.text((5, row_index * 286 + 274), image_id, fill='black', font=font)
            name = f'{task}_sigma{round(sigma * 1000):03d}.png'; canvas.save(output / name)
    manifest = {path.name: files.digest(path) for path in sorted(output.glob('*.png'))}
    files.write(output / 'manifest.json', dict(fixed_images=list(rules.PANEL_IMAGES), files=manifest,
        selection='Fixed in source before outcomes; no quality-based selection.'))
    print('A101 QUALITATIVE PASS:', len(manifest), 'fixed panels in', output, flush=True)
