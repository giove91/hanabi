from __future__ import unicode_literals

import sys, os
from django.db import models
from django.core.validators import RegexValidator
from datetime import date
from jsonfield import JSONField # https://github.com/bradjasper/django-jsonfield

from hanabi.settings import BASE_DIR

sys.path.append(os.path.join(BASE_DIR, '../game'))
import game, card


class Team(models.Model):
    name = models.CharField(max_length=200)
    
    def __unicode__(self):
        return u'%s' % self.name


class AI(Team):
    ai_name = models.CharField(max_length=200)
    ai_params = JSONField(default=dict)
    fixed_elo = models.FloatField(blank=True, null=True, default=None)


class Player(models.Model):
    name = models.CharField(max_length=200)
    
    def __unicode__(self):
        return u'%s' % self.name


class GameSetup(models.Model):
    COLORS = card.Card.COLORS
    CARD_REGEX = r'\d+ (' + '|'.join(COLORS) + ') \d+'
    DECK_REGEX = '^' + CARD_REGEX + '(,' + CARD_REGEX + ')*$'
    
    deck = models.CharField(
        max_length=1024,
        validators=[RegexValidator(regex=DECK_REGEX)]
    )   # comma-separated list of cards
    
    num_players = models.PositiveIntegerField()
    
    datetime = models.DateTimeField(auto_now_add=True)
    
    def __unicode__(self):
        return u"%r" % self.datetime


class Result(models.Model):
    deck = models.ForeignKey(GameSetup, on_delete=models.CASCADE)
    team = models.ForeignKey(Team, on_delete=models.CASCADE, blank=True, null=True, default=None)
    players = models.ManyToManyField(Player)
    date = models.DateField(default=date.today)
    
    score = models.PositiveIntegerField()
    lives = models.PositiveIntegerField(blank=True, null=True)
    hints = models.PositiveIntegerField(blank=True, null=True)
    num_turns = models.PositiveIntegerField(blank=True, null=True)
    


