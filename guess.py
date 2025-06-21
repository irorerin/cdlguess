import os
import sys
import torch
import OpenImageIO as oiio

def read_image_file(file_path, header_only = False):
    result = {'spec': None, 'image_data': None}
    inp = oiio.ImageInput.open(file_path)
    if inp :
        spec = inp.spec()
        result['spec'] = spec
        if not header_only:
            channels = spec.nchannels
            result['image_data'] = inp.read_image(0, 0, 0, channels)
        inp.close()
    return result

def clear_lines(n=2):
    """Clears a specified number of lines in the terminal."""
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)

import signal
def create_graceful_exit():
    def graceful_exit(signum, frame):
        exit(0)
    return graceful_exit
signal.signal(signal.SIGINT, create_graceful_exit())



print ('Helloo!!!')