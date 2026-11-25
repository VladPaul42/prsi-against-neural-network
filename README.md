# Prší

https://github.com/user-attachments/assets/9cacd9e5-1765-46ce-a85d-fadaabe50122

What you are watching is two neural networks playing against each other in a game of Prší.

Their approach to finding the best move has to differ fundamentally from, for example, the one chess engines use, because, first of all, out of 32 cards you only see the ones in your hand.<br>
*This makes Prší a game with incomplete information.*<br>
<br>
And second, moves leading to a particular position matter, since a good player finds hints about the opponent's hand without ever seeing the cards.<br>
*This makes Prší a non-Markovian game.*

But let's not get ahead of ourselves.

## Game rules

Prší is a card game popular in the Czech Republic. Its goal is to get rid of all your cards.

Prší is played with a 32-card German deck: 7, 8, 9, 10, Under, Over, King, Ace, in Hearts, Leaves, Acorns, and Bells.*

Each player is dealt 6 cards. One card is turned up. That opening card has no effect, even if it is a 7, an Ace, or an Over. Nineteen cards remain in the draw pile.

- Play a card that matches the suit in play, or the rank of the top card.
- An Over can be played on almost anything. You then name the suit in play. It cannot answer a live 7 or Ace.
- A 7 makes the next player take 2 cards for every 7 in a row, or play another 7.
- An Ace makes the next player stand, or play another Ace.
- You never have to play. You may draw, take the penalty, or stand instead. Passing is only for when nothing can be drawn and nothing can be played.
- Playing your last card wins, even if it is a 7 or an Ace.
- When the draw pile runs out, the discard is shuffled back in, except its top card.

<em>*I replaced Hearts with Pomegranates and Bells with Bellflowers in my card designs, so that all the spot cards would be botanically themed.</em>

## The table

You play in a browser. Your cards are at the bottom. The other hand is face down across the table. Between you sit the draw pile and the discard. Recent moves stay along one side of the page. When you play an Over, the page asks you to name the suit.

The person who starts the room is the host. A friend joins with a room code. The host chooses who sits down.

![Playing the bot, with an Ace to answer](<Project media/me-vs-the-bot.png>)

This is a game against the bot. An Ace has just been played, so it has to be stood on, or answered with another Ace. This seat sees its own cards. The bot's hand, and the search it is running, stay on the host's machine.

There are four ways to take a seat. Two people can play, and the bot stays out. You can play the bot yourself. A friend can play the bot while you watch its hand and the lines it is trying, and confirm each of its moves before they land. Or both seats can be the bot, and you watch.

![Both bots, with the search and the win-chance bar](<Project media/bot-vs-bot.png>)

Both hands are on the table here, and so is the search. The lines on the right are the cards being tried. The bar is the bottom seat's chance of winning, filling in as imagined games come back.

When a game ends, it is written down on the machine running the server. The host can open it again, step through the moves, and ask the page to analyze every one of them.

![Review of a saved game, with the win-chance chart](<Project media/review.png>)

This one has been stepped back to the first move. The solid lines are each player's own estimate, from the cards that player could see. The dashed lines are the same positions with both hands known.

## Why I did this

My interest in game engines started long ago with chess. I had a basic idea of how they work. I knew what a node is, and what a tree search is, and I went on from that knowledge.

A chess engine searches a tree of positions both players can see. I wanted that same search in a game where the position is only partly there. In Prší the missing cards are the opponent's hand and the order of the draw pile. A line through the tree is a line through one way those cards might be arranged. The engine had to live with that, one imagined deal at a time.

## What is ISMCTS

Information set Monte Carlo tree search is a tree search for a position you only partly see.

What you can see is an information set: every deal that might still be the real one, given your hand, the cards on the table, and the play so far. The search samples one legal deal from that set and plays it down a tree, as a chess program would if the cards were face up. Then it samples another. A card earns its place from the deals in which it was actually tried.

That is the idea. The habits of this particular search, and the network inside it, come next.

## What is PPO

Proximal policy optimization is how that network learned.

A game ends, and the ending is the only reward: +1, −1, or 0 for a draw. The learner then takes a small step toward the moves that produced it, small enough to stay close to the player that just played those games. An entropy coefficient keeps unlikely cards in the mix. I brought that coefficient down from 0.30 to 0.00, one hundredth at a time. This schedule is needed so that the agent rolls down smoothly from exploration to exploitation.

## How the engine plays and trains

The bot never sees the opponent's hand or the order of the draw pile. It sees its own hand, the public piles, and a card tracker. The tracker keeps the cards that are certainly in the opponent's hand, the cards that cannot be there, and a pool that is split in a known count between that hand and the draw pile. So there's only one state in which all the cards can be known exactly - immediately after the reshuffle.

Search imagines one legal deal at a time and plays it down a tree of moves. At every decision, the card it tries next is the legal one it has visited least. A card the tree has never tried goes before a card it has already seen.

The first time a position is reached, the network gives a probability to every legal card. Those numbers are stored and shown on the table, the network's opinion at the moment the search arrives. The next card tried is the one visited least. There is no random playout to the end of the deal. The value head scores the position just reached, and that score is what comes back up the tree. On the way back, the player to move is treated as playing the best card that has actually been tried.

The network is a small multilayer perceptron. The input is 1,564 numbers: the hand, the public flags, where the tracker thinks each card is, and the last 30 moves. Two hidden layers of 512 units feed a policy over the 45 legal action numbers and a value in the range −1 to 1. Playing uses a NumPy copy of the weights, so the table does not need PyTorch.

The rules live in `prsi`. The search uses a second, faster copy of those rules in `bot`. The tests play the same moves through both.

## How it was trained

I trained on CPU. The only reward is the result of the game: +1, −1, or 0 for a draw. During training the learner samples from its policy, and so does the opponent. The games that decide a champion are different. Each network plays its most likely card.

Generation 1 learned against random play. Every later generation learned against random play and against all earlier generations, and never against itself. An opponent it was losing to was dealt more of the games.

Each generation walks an entropy coefficient from 0.30 down to 0.00, in steps of 0.01. Generations 1–6 took 30 updates at each step. Generation 7 took 100. An update is 512 games. Generation 7 is 3,100 updates, 1,587,200 games, and 14.8 hours. The six generations before it are about 476,000 games each.

Along that walk, the score I kept was the one against a reference opponent. For generation 1 the reference was random play. After that it was the generation holding the title.

![Score vs reference](<Project media/score-vs-reference.png>)

Generation 7 stayed at each step much longer. The opponents it was still losing to were the earlier networks, and those were the ones dealt more of its games.

![Generation 7 during training](<Project media/generation-7-without-baseline.png>)

After training, a generation plays 100,000 mirrored games against every earlier generation and against random play. The same deal is played twice, with the seats swapped, because the cards matter. A draw counts as half a win. The new generation becomes the best only if it scores over 50% against every earlier generation. Generation 6 did not. It reached 49.1% against generation 5, and generation 5 stayed the best. Generation 7 did: 51.4% against generation 5, and 97.0% against random play.

The hundred thousand games are the verdict. The learning is the stretch before them. An update is a small change, made from 512 games. Generation 1 met the game with nothing behind it. Every generation after that met someone who could already play.

![Training updates](<Project media/ppo-updates.png>)

Seven generations sat down. The title moved after each of the first five. Generation 6 trained and left it where it was. Generation 7 took it.

| Generation | Updates | Games | Hours | Previous best | Score against best | Became best | Best after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 930 | 476,160 | 8.80 | | | Yes | 1 |
| 2 | 930 | 476,160 | 3.95 | 1 | 55.1% | Yes | 2 |
| 3 | 930 | 476,160 | 3.67 | 2 | 50.5% | Yes | 3 |
| 4 | 930 | 476,160 | 3.88 | 3 | 50.4% | Yes | 4 |
| 5 | 930 | 476,160 | 4.25 | 4 | 51.1% | Yes | 5 |
| 6 | 930 | 476,160 | 4.05 | 5 | 49.1% | No | 5 |
| 7 | 3,100 | 1,587,200 | 14.79 | 5 | 51.4% | Yes | 7 |

The weights the table loads are generation 7, in `runs/league/gen_007/weights.npz`. The settings of that run are in `runs/league/config.json`. Retraining needs `pip install -r requirements-train.txt` and then `bot.league.run_league`.

## How to run

Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m server
```

On macOS or Linux, create the environment the same way, then `source .venv/bin/activate` and run `python -m server`.

The terminal prints three things:

- a link for this computer
- a room code
- a host link

Open the host link. That browser can choose the mode, watch the bot think, and open the review page. Send a friend the room code and, if they are on another network, an [ngrok](https://ngrok.com) address pointed at the port:

```powershell
ngrok http 8000
```

They open the ngrok address, enter the code, and join. The host link stays on your machine. Sharing it lets someone else change the mode and see the bot's cards.

| Mode | Who plays |
| --- | --- |
| Me vs my friend | Two people. The bot stays out. |
| Me vs the bot | You play. The bot thinks, then plays. You do not see its search. |
| My friend vs the bot | Your friend plays the bot. You see the bot's hand and its search, and you confirm each of its moves. |
| Bot vs bot | You watch both seats, both searches, and a win-chance bar. |

The bot thinks for 5 seconds a move, and also stops if it reaches 100,000 imagined games. Hitting that cap does not play the card early. In "My friend vs the bot" the clock does not apply: the bot searches until you confirm.

Saved games are written to `games/` on the machine running the server. The host link's review page steps through them and, when you ask, scores every move.

```powershell
.\.venv\Scripts\python -m pytest
```

The league tests need the training packages as well: `pip install -r requirements-train.txt`.

## Layout

```
prsi/            the rules
bot/             the tracker, the search, the network, the league
server/          the table
static/          the browser, the type, the card faces
runs/league/     generation 7, and the settings that trained it
design elements/ the table texture
Project media/   the recording, the screenshots, the charts
```

## Credits

The card faces and the table are original. The type is [Cormorant Garamond](https://github.com/CatharsisFonts/Cormorant), used under the SIL Open Font License. The license text is in `static/fonts/OFL.txt`.

## License

MIT. See [LICENSE](LICENSE).
