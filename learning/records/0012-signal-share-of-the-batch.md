# The batch is mostly draws (2026-09-17)

Asked which of the three board groups (puzzle, finishing, full games) carried
almost no finishing signal yet filled a third of every PPO batch, the owner named
the draw games at once. On why the depth-0 success rate stayed at 0.38 for 120
updates despite a +2 reward, the owner offered two hypotheses: "not enough model
capacity" (rejected in discussion: the same network reaches 0.58 on Lichess
mate-in-one, so capacity is not the limit for a one-move decision) and "too many
other positions that don't result in a win", which is the dilution argument:
about 150 of 768 positions per update came from finishing boards, and over the
run the finishing boards played 1,900 episodes against 21,000 puzzle episodes.
Why it matters: the next lever is the board split and the episode count, not the
network. Business analogue: a rare-event class swamped by the majority class in
every minibatch.
