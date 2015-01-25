def print_header(header, level=0):     
    encl_len = len(header) + 4 
    enclosure = ''
    if level <= 0:
        enclosure = '='
    elif level == 1:
        enclosure = '-'
    elif level >= 2:
        enclosure = ' '
    print(enclosure*encl_len)
    print(' '.join([enclosure, header, enclosure]))
    print(enclosure*encl_len)
