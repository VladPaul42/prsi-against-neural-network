# Prší

A two-player card game in the browser. Sit down with a friend, or with a bot that imagines the cards it cannot see and scores those deals with a neural network.

The recording is a bot-versus-bot game. Both seats are generation 7.

https://github.com/user-attachments/assets/d2e63438-85f6-4a4b-9a3b-978a822956bd

## The table

Playing the bot. The search stays on the host's side of the glass: this seat only sees its own cards. The Ace just played has to be stood on, or answered with another Ace.

![Playing the bot, with an Ace to answer](<Project media/me-vs-the-bot.png>)

Watching the bot play itself. The panel is the search in progress. The bar on the right is the bottom seat's win chance, filled in as imagined finishes come back.

![Both bots, with the search and the win-chance bar](<Project media/bot-vs-bot.png>)

A finished game, stepped back to the first move. Solid lines are each player's own estimate. Dashed lines are the same positions with both hands known. The page normally gives each legal card 10,000 imagined games. This capture used 250, so the chart would finish.

![Review of a saved game, with the win-chance chart](<Project media/review.png>)

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

## Rules

A 32-card German deck: 7, 8, 9, 10, Under, Over, King, Ace, in Hearts, Leaves, Acorns, and Bells. The hearts are drawn as pomegranates. The English name stays Hearts.

Each player is dealt 6 cards. One card is turned up. That opening card has no effect, even if it is a 7, an Ace, or an Over. Nineteen cards remain in the draw pile.

- Play a card that matches the suit in play, or the rank of the top card.
- An Over can be played on almost anything. You then name the suit in play. It cannot answer a live 7 or Ace.
- A 7 makes the next player take 2 cards for every 7 in a row, or play another 7.
- An Ace makes the next player stand, or play another Ace.
- You never have to play. You may draw, take the penalty, or stand instead. Passing is only for when nothing can be drawn and nothing can be played.
- Playing your last card wins, even if it is a 7 or an Ace.
- When the draw pile runs out, the discard is shuffled back in, except its top card.

## How the bot plays

The bot never sees the opponent's hand or the order of the draw pile. It sees its own hand, the public piles, and a card tracker. The tracker keeps the cards that are certainly in the opponent's hand, the cards that cannot be there, and a pool that is split in a known count between that hand and the draw pile. A reshuffle makes this sharp: an empty draw pile means every unseen card is in the opponent's hand, and the new draw pile is the old discard minus its top card, in an unknown order.

Search imagines one legal deal at a time and plays it down a tree of moves. At every decision, among the cards legal in that deal, the one tried least often goes next. A card the tree has never seen is tried before a repeat. On the way back, the player to move is treated as playing the best card that has actually been tried.

The network does two jobs inside that search, and neither one is "pick the top move."

- The first time a position is reached, the network gives a probability to every legal card. Those numbers are stored and shown. They do not choose which card is tried next.
- There is no random rollout. The new position is scored by the value head.

Without a trained network the same search finishes deals by playing at random. Generation 7 is the network the table loads.

The network itself is a small multilayer perceptron. The input is 1,564 numbers: the hand, the public flags, where the tracker thinks each card is, and the last 30 moves. Two hidden layers of 512 units feed a policy over the 45 legal action numbers and a value in the range −1 to 1. Playing uses a NumPy copy of the weights, so the table does not need PyTorch.

The rules live in `prsi`. The search uses a second, faster copy of those rules in `bot`. The tests play the same moves through both.

## How it was trained

Proximal policy optimization, on CPU. The only reward is the result of the game: +1, −1, or 0 for a draw. During training the learner samples from its policy, and so does the opponent. The evaluation games are different: each network plays its most likely card.

Generation 1 learned against random play. Every later generation learned against random play and against all earlier generations, and never against itself. An opponent it was losing to was dealt more of the games.

Each generation walks an entropy coefficient from 0.30 down to 0.00, in steps of 0.01. Generations 1–6 took 30 updates at each step. Generation 7 took 100. An update is 512 games. Generation 7 is 3,100 updates, 1,587,200 games, and 14.8 hours. The six generations before it are about 476,000 games each.

After training, a generation plays 100,000 mirrored games against every earlier generation and against random play. The same deal is played twice, with the seats swapped, because the cards matter. A draw counts as half a win. The new generation becomes the best only if it scores over 50% against every earlier generation. Generation 6 did not. Generation 7 did: 51.4% against generation 5, the previous best, and 97.0% against random play.

![Win rate of each generation against earlier generations and against random play](<Project media/evaluation-by-generation.png>)

*Each bar is 100,000 mirrored games. The dashed line is an even score. Random play is the tall bar in every group.*

The chart below is generation 7 while it was still learning, so the learner was sampling moves rather than playing its favorite. Those scores sit lower than the final evaluation. The middle panel is how often each opponent was picked. Random play, the baseline, drops out of the mix once the earlier networks are the ones it loses to.

![Generation 7's training scores, opponent mix, and games per update](<Project media/generation-7-opponents.png>)

The weights are `runs/league/gen_007/weights.npz`. The settings of that run are in `runs/league/config.json`. Retraining needs `pip install -r requirements-train.txt` and then `bot.league.run_league`.

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
