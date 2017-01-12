from models import *

HUMAN = 'H'
AI = 'A'


def k_factor(num_games, entity_type):
    if entity_type == HUMAN:
        if num_games < 10:
            return 40.0
        elif num_games < 20:
            return 20.0
        else:
            return 10.0
    
    elif entity_type == AI:
        return 400.0 / (num_games + 1)


def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

def elo_variation(rating_a, rating_b, num_games, entity_type, score):
    return k_factor(num_games, entity_type) * (score - expected_score(rating_a, rating_b))


