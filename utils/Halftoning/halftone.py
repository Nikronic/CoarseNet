import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageStat as ImageStat

import random


def generate_halftone(file):
    if type(file) == str and 1 == 0:
        try:
            im = Image.open(file)
        except IOError:
            raise
    else:
        im = file

    angles = []
    for j in range(4):
        angles.append(random.randint(0, 90))

    sample = random.randint(5, 5)

    cmyk = im.convert('CMYK')
    dots = halftone(im, cmyk, sample, angles, shape=random.getrandbits(1))
    new = Image.merge('CMYK', dots)
    new = new.convert('RGB')
    return new


def halftone(im, cmyk, sample, angles, shape):
    cmyk = cmyk.split()
    dots = []

    for channel, angle in zip(cmyk, angles):
        channel = channel.rotate(angle, expand=1)
        size = channel.size[0], channel.size[1]
        half_tone = Image.new('L', size)
        draw = ImageDraw.Draw(half_tone)

        for x in range(0, channel.size[0], sample):
            for y in range(0, channel.size[1], sample):
                box = channel.crop((x, y, x + sample, y + sample))
                mean = ImageStat.Stat(box).mean[0]
                diameter = (mean / 255) ** 0.5
                draw_diameter = diameter * sample
                box_x, box_y = x, y
                x1 = box_x + ((sample - draw_diameter) / 2)
                y1 = box_y + ((sample - draw_diameter) / 2)
                x2 = x1 + draw_diameter
                y2 = y1 + draw_diameter

                if shape is 1:
                    draw.ellipse([(x1, y1), (x2, y2)], fill=255)
                else:
                    draw.rectangle([(x1, y1), (x2, y2)], fill=255)

        half_tone = half_tone.rotate(-angle, expand=1)
        width_half, height_half = half_tone.size

        xx1 = (width_half - im.size[0]) / 2
        yy1 = (height_half - im.size[1]) / 2
        xx2 = xx1 + im.size[0]
        yy2 = yy1 + im.size[1]

        half_tone = half_tone.crop((xx1, yy1, xx2, yy2))

        dots.append(half_tone)
    return dots
