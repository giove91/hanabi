from __future__ import unicode_literals

import sys, os
from django.db import models
from django.core.validators import RegexValidator

from hanabi.settings import BASE_DIR

sys.path.append(os.path.join(BASE_DIR, '../game'))
import game, card


class Team(models.Model):
    HUMAN = 'H'
    AI = 'A'
    
    TYPE_CHOICES = (
        (HUMAN, 'Human'),
        (AI, 'AI'),
    )

    name = models.CharField(max_length=200)
    type = models.CharField(max_length=1, choices=TYPE_CHOICES, default=HUMAN)


class GameSetup(models.Model):
    COLORS = card.Card.COLORS
    CARD_REGEX = r'\d+ (' + '|'.join(COLORS) + ') \d+'
    DECK_REGEX = '^' + CARD_REGEX + '(,' + CARD_REGEX + ')*$'
    
    deck = models.CharField(
        max_length=1024,
        validators=[RegexValidator(regex=DECK_REGEX)]
    )
    
    num_players = models.PositiveIntegerField()
    
    datetime = models.DateTimeField(auto_now_add=True)
    
    def __unicode__(self):
        return u"%r" % self.datetime


class Result(models.Model):
    deck = models.ForeignKey(GameSetup, on_delete=models.CASCADE)



