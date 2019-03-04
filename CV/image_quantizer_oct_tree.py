from PIL import Image
from oct_tree import OctTree


def buildAndDisplay():
    im = Image.open('beach_pic.jpg')
    w, h = im.size
    ot = OctTree()
    for row in range(0, h):
        for col in range(0, w):
            r, g, b = im.getpixel((col, row))
            ot.insert(r, g, b)
    # reduce to 256 colors
    ot.reduce(256)
    for row in range(0, h):
        for col in range(0, w):
            r, g, b = im.getpixel((col, row))
            nr, ng, nb = ot.find(r, g, b)
            # replace pixel with new quantized values
            im.putpixel((col, row), (nr, ng, nb))
    im.show()
