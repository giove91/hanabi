#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random

from game.card import *

"""
Generate sample input for build_deck.py
"""

d = deck()
random.shuffle(d)

for card in d:
    print card.number, card.color
    print

