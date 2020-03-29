Hanabi
=====================

This is a program that plays [Hanabi](https://boardgamegeek.com/boardgame/98778/hanabi).

The main AI is contained in the directory `alphahanabi`. It is optimized for 5-player games, but it also plays well 4-player games.
A sample (stupid) AI is provided in the directory `dummy`.
An additional stronger AI is contained in the directory `deltahanabi`.


Performance
---------------------

Here are the average scores and win rates for the standard 50-card deck (5 colors) and for the "black" variant (6 colors, where the 6th color has 1 copy of each card).


 AI                    | standard 4 players | standard 5 players | black 4 players | black 5 players |
-----------------------|--------------------|--------------------|-----------------|-----------------|
alphahanabi (moderate) | 6.3 %              | 12.1 %             | 
alphahanabi (hard)     | -                  | 54.7 %             | -               | 
alphahanabi (hardest)  | 76.7 %             | 80.6 %
deltahanabi            | 73.2 %             | 96.1 %


Requirements
---------------------
* Python 2.7
* The `termcolor` Python module (install command: `pip install termcolor`)
* The `blessings` Python module (install command: `pip install blessings`)

Run a new game
---------------------
`python run_game.py`

**Command line options**
* `-n NUM_PLAYERS` set number of players (default is 5)
* `-a AI_DIRECTORY` choose AI (default is `alphahanabi`)
* `-p DIFFICULTY` choose difficulty level for `alphahanabi` (possible values: `moderate`, `hard`, `hardest`; default is `hardest`)
* `-c` run the game without pausing (otherwise, by default, each new turn is played by pressing ENTER)
* `-s` activate the strategy log
* `-t` print a shorter log of turns and status
* `-l FILE_NAME` load the initial deck from the given file (otherwise, by default, the initial deck is shuffled randomly)
* `-d FILE_NAME` dump the initial deck to the given file (default is `deck.txt`)
* `-r SCORE` run many games, until a score <= to the given score is reached
* `-i` run in interactive mode
* `-q` quit immediately after showing the initial cards (not in interactive mode)



Run many games and print statistics
---------------------
`python test.py`

or (much faster)`pypy test.py`

**Command line options**
* `-n NUM_PLAYERS` set number of players (default is 5)
* `-m NUM_GAMES` set number of games (default is 1000)
* `-a AI_DIRECTORY` choose AI (default is `alphahanabi`)
* `-p DIFFICULTY` choose difficulty level for `alphahanabi` (possible values: `moderate`, `hard`, `hardest`; default is `hardest`)



Challenge QuickStart
---------------------
First run, saving the deck and seeing the cards:

`python run_game.py -d FILE_NAME -q`

Second run, loading the deck and running the game without showing cards:

`python run_game.py -l FILE_NAME -p DIFFICULTY -t -i`

