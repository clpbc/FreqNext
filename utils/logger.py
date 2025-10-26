# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode = None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal = None, is_file = None):
        if '\r' in message:
            is_file = False
        if self.file:
            is_file = True

        is_terminal = is_terminal if is_terminal is not None else True
        if is_terminal:
            is_terminal = is_terminal
        else:
            is_terminal = False

        if is_terminal == True:
            self.terminal.write(f"{message}\n")
            self.terminal.flush()
        if is_file == True:
            self.file.write(f"{message}\n")
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass