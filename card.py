

class Card:
    RED = 'Red'
    BLUE = 'Blue'
    WHITE = 'White'
    YELLOW = 'Yellow'
    GREEN = 'Green'
    RAINBOW = 'Rainbow'
    
    COLORS = [RED, BLUE, WHITE, YELLOW, GREEN, RAINBOW]
    
    def __init__(self, color, number):
        assert color in self.COLORS
        assert 1 <= number <= 5
        
        self.color = color
        self.number = number
    
    
    def __repr__(self):
        return "%d %s" % (self.number, self.color)


def deck():
    deck = []
    for color in Card.COLORS:
        for number in xrange(1, 6):
            if color == Card.RAINBOW:
                quantity = 1
            elif number == 1:
                quantity = 3
            elif 2 <= number <= 4:
                quantity = 2
            elif number == 5:
                quantity = 1
            else:
                raise Exception("Unknown card parameters.")
            
            for i in xrange(quantity):
                deck.append(Card(color, number))
    
    return deck

