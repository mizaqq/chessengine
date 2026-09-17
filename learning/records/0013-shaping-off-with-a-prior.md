# Drop material shaping once a prior exists (2026-09-17)

After the probe showed the value head predicting -2.6 for a side that can mate
next move, the owner proposed the fix before Claude did: "fallback on rewarding
the model basing on material as we have such pretraining? The model thus can
learn the real reward is the win", i.e. remove the material term because the
pretrained network already carries material sense, so the outcome can be the
only reward (AlphaZero's setting). The arm confirmed it: the value on own-game
mate positions went -2.6 -> +0.1, Lichess mate-in-one broke a 900-update plateau
(0.57 -> 0.64), decisive games doubled. Why it matters: shaping's
policy-invariance (Ng et al. 1999) assumes exact values; with a small value head
the shaping term becomes what the head learns. Dense shaping is a bootstrap for
a blank network and a liability once the prior supplies the density.
