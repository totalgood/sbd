#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Console script for detecting sentences from a file containing ASCII text.

`python setup.py install` which will install the command `detect`
in your environment.
"""
from __future__ import division, print_function, absolute_import

import argparse
import sys
import logging
import os

from nlup.decorators import IO
from .detector import Detector, slurp, EPOCHS

LOGGING_FMT = "%(module)s.%(function)s:%(lineno)d %(message)s"
DEFAULT_MODEL_PATH = os.path.join('data', 'trained-detector-wsj-ptb.json.gz')

from sbd import __version__


__author__ = "Hobson Lane"
__copyright__ = "Kyle Gorman"
__license__ = "mit"

log = logging.getLogger(__name__)


def parse_args(args):
    """
    Parse command line parameters

    :param args: command line parameters as list of strings
    :return: command line parameters as :obj:`argparse.Namespace`
    """
    parser = argparse.ArgumentParser(
        description="Sentence detector and generator")
    parser.add_argument(
        '--version',
        action='version',
        version='sbd {ver}'.format(ver=__version__))
    vrb_group = parser.add_mutually_exclusive_group()
    vrb_group.add_argument("-v", "--verbose", action="store_true",
                           help="enable verbose output")
    vrb_group.add_argument("-V", "--really-verbose", action="store_true",
                           help="enable even more verbose output")
    inp_group = parser.add_mutually_exclusive_group(required=True)
    inp_group.add_argument("-t", "--train", help="training data")
    inp_group.add_argument("-r", "--read",
                           default=os.path.dirname(os.pathabspath(__file__)), DEFAULT_MODEL_PATH)
                           help="load a previously trained perceptron sentence detector model")
    out_group = parser.add_mutually_exclusive_group(required=True)
    out_group.add_argument("-s", "--segment", help="segment sentences")
    out_group.add_argument("-w", "--write",
                           help="write out serialized model")
    out_group.add_argument("-e", "--evaluate",
                           help="evaluate on segmented data")
    parser.add_argument("-E", "--epochs", type=int, default=EPOCHS,
                        help="# of epochs (default: {})".format(EPOCHS))
    parser.add_argument("-C", "--nocase", action="store_true",
                        help="disable case features")
    args = parser.parse_args()
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    if args.really_verbose:
        log.basicConfig(format=LOGGING_FMT, level="DEBUG")
    elif args.verbose:
        log.basicConfig(format=LOGGING_FMT, level="INFO")
    else:
        log.basicConfig(format=LOGGING_FMT)

    detector = None
    if args.train:
        logging.info("Training model on '{}'.".format(args.train))
        detector = Detector(slurp(args.train), epochs=args.epochs,
                                               nocase=args.nocase)
    elif args.read:
        logging.info("Reading pretrained model '{}'.".format(args.read))
        detector = IO(Detector.load)(args.read)
    # output block
    if args.segment:
        logging.info("Segmenting '{}'.".format(args.segment))
        print("\n".join(detector.segments(slurp(args.segment))))
    if args.write:
        logging.info("Writing model to '{}'.".format(args.write))
        IO(detector.dump)(args.write)
    elif args.evaluate:
        logging.info("Evaluating model on '{}'.".format(args.evaluate))
        cx = detector.evaluate(slurp(args.evaluate))
        if args.verbose or args.really_verbose:
            cx.pprint()
        print(cx.summary)
    log.info("Finished generating sentences")


def run():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main(sys.argv[1:])


if __name__ == "__main__":
    run()



