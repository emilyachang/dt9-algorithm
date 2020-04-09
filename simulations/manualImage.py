#!/usr/bin/env python3
"""
Base arguments for manual creation of component parts in simulated image.
Takes command line inputs in the following format:
    python3 manualImage.py [x_dim] [y_dim] (default = 1280x1024)
    options:
        -circle [x_pos] [y_pos] [r]
        -ellipse [x_pos] [y_pos] [x_r] [y_r]
"""

import argparse
from PIL import Image, ImageDraw

def createImage(x, y):
    """Creates a basic image of size x by y pixels."""
    image = Image.new("L", (x,y), 0)  # 1280x1024 8-bit B/W pixels
    return image

def drawCircle(image, x, y, r):
    """Draws a circle at position (x,y) with radius r."""
    draw = ImageDraw.Draw(image)
    draw.ellipse([(x-r,y-r),(x+r,y+r)], 255, 255)  

def drawEllipse(image, x, y, xr, yr):
    """Draws a circle at position (x,y) with radii xr and yr."""
    draw = ImageDraw.Draw(image)
    draw.ellipse([(x-xr,y-yr),(x+xr,y+yr)], 255, 255)  

def main():
    try:
        print("\nhello :)\n")
        # parse arguments from command line (VERY BASIC!!! i was just fucking around)
        parser = argparse.ArgumentParser(description="Process command line input.")
        parser.add_argument("xdim", nargs="?", default="1280", metavar="dim", type=int, help="image initialization x dim")
        parser.add_argument("ydim", nargs="?", default="1024", metavar="dim", type=int, help="image initialization y dim")
        parser.add_argument("-circle", nargs=3, dest="c", metavar=("x","y","r"), type=int, help="create circle at (x,y) with radius r")
        parser.add_argument("-ellipse", nargs=4, dest="e", metavar=("x","y","xr","yr"), type=int, help="create ellipse at (x,y) with radii xr and yr")
        args = parser.parse_args()
        # call relevant commands
        print(args)
        img = createImage(args.xdim,args.ydim)
        if args.c != None:
            drawCircle(img, args.c[0], args.c[1], args.c[2])
        if args.e != None:
            drawEllipse(img, args.e[0], args.e[1], args.e[2], args.e[3])
        img.show()
    except IOError as err:
        sys.stdout.write("I/O Error: {0}\n".format(err))
    except ValueError:
        sys.stdout.write("Value Error detected.\n")

if __name__ == "__main__":
    main()
