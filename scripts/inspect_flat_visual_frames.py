"""Generate review strips from fixed-command videos without changing playback."""
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image, ImageDraw

root = Path('logs/flat_w1_3_visual_gate')
for direction in ('forward', 'backward', 'left', 'right', 'turn_left', 'turn_right'):
  canvas = Image.new('RGB', (6 * 320, 2 * 260), 'white')
  draw = ImageDraw.Draw(canvas)
  for row, label in enumerate(('a4', 'w1_3')):
    reader = imageio.get_reader(root / f'{label}_{direction}.mp4')
    for column, second in enumerate((1, 3, 5, 7, 9, 11)):
      frame = Image.fromarray(reader.get_data(second * 50)).resize((320, 240))
      canvas.paste(frame, (column * 320, row * 260 + 20))
      draw.text((column * 320 + 5, row * 260 + 3), f'{label} {second}s', fill='black')
    reader.close()
  canvas.save(root / f'review_{direction}.png')
  print(root / f'review_{direction}.png')
  for label in ('a4', 'w1_3'):
    detail = Image.new('RGB', (5 * 240, 5 * 220), 'white')
    draw = ImageDraw.Draw(detail)
    reader = imageio.get_reader(root / f'{label}_{direction}.mp4')
    for index in range(25):
      frame = Image.fromarray(reader.get_data(350 + index * 10))
      feet = frame.crop((300, 370, 660, 720)).resize((240, 200))
      x, y = index % 5 * 240, index // 5 * 220
      detail.paste(feet, (x, y + 20))
      draw.text((x + 5, y + 3), f'{7 + index * .2:.1f}s', fill='black')
    reader.close()
    detail.save(root / f'feet_{label}_{direction}.png')
