#!/usr/bin/env python

import argparse
import aesir

def main():
    parser = argparse.ArgumentParser(
            description='Writes a binary version of an Andrews format corpus.')
    parser.add_argument('--input', '-i', metavar='FILE',
                        help='The input corpus (in Andrews format).')
    args = parser.parse_args()
    aesir.dataread(args.input)

if __name__ == '__main__':
    main()

