Hanabi
=====================

This is a program that plays [Hanabi](https://boardgamegeek.com/boardgame/98778/hanabi).

The main AI is contained in the directory `alphahanabi`. It currently plays well only the 5-player games.
A sample (stupid) AI is provided in the directory `dummy`.

Requirements
---------------------
* Python 2.7
* The `termcolor` Python module (install command: `pip install termcolor`)

Run a new game
---------------------
`python run_game.py`

**Command line options**
* `-n NUM_PLAYERS` set number of players (default is 5)
* `-a AI_DIRECTORY` choose AI (default is `alphahanabi`)
* `-c` run the game without pausing (otherwise, by default, each new turn is played by pressing ENTER)
* `-s` activate the strategy log
* `-t` short log of turns and status
* `-l FILENAME` load the initial deck from the given file (otherwise, by default, the initial deck is shuffled randomly)
* `-d FILENAME` dump the initial deck to the given file (default is `deck.txt`)
* `-r SCORE` run many games, until a score <= to the given score is reached



Run many games and print statistics
---------------------
`python test.py`

or (much faster)`pypy test.py`

**Command line options**
* `-n NUM_PLAYERS` set number of players (default is 5)
* `-m NUM_GAMES` set number of games (default is 1000)
* `-a AI_DIRECTORY` choose AI (default is `alphahanabi`)
