from game import Game


results = []

for i in xrange(100):
    game = Game(num_players=5, wait_key=False, log=False, strategy_debug=False)
    
    game.setup()
    res = game.run_game()
    results.append(res)

print sorted(results)
print "Average result:", float(sum(results)) / len(results)
print "Best result:", max(results)
print "Worst result:", min(results)
