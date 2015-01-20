import numpy as np

def print_header(header, level=0):
    encl_len = len(header) + 4
    enclosure = ''
    if level <= 0:
        enclosure = '='
        print(enclosure*encl_len)
    elif level == 1:
        enclosure = '-'
    elif level >= 2:
        enclosure = ' '
    print(' '.join([enclosure, header, enclosure]))
    print(enclosure*encl_len)

print_header("Section 1")
print_header("Section 1.1",1)

def neuron():
    pass


