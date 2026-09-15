# Under potential shaping the value head outputs V - Phi

Established 2026-09-15. The owner first said the value head should output a
positive value for the mating move's advantage to be positive. Corrected: the
advantage is return minus the value at the position where the move is chosen, and
for a mating move while ahead by 9 the shaped return is -7, so the value head must
output below -7 (winning chances minus the material loan). The owner then stated
the practical rule: a very negative value output can mean "far ahead in material",
and material must be subtracted back out before reading whether the model thinks
it is winning.

**Evidence:** conversation Q2 of the comprehension check; also explained why
zeroing `phi_next` after the own-move update would leak the next game's material
into a checkmate (-5 instead of +2).

**Implications:** when plotting or debugging value outputs, add a "value + Phi"
view; relevant for GAE (lambda leans on this value head) and for any future
`shaping_scale` change.
