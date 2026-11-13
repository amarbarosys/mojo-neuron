# Facts late layer, value = U[right] (written 2026-11-13 before the run)

Basis: facts-late (b87d7db) MISSED 6/20 at 31/8; 19/20 top-1 moved, so the value U[right] - U[wrong] mostly sinks the wrong token. Change only the value:
each row's value becomes U[right] rescaled to that row's original norm (keys, gates, tau unchanged). Same 20 facts, layers 28/30/31, scales 0.5..8,
control `control_sets` of the new file. Bars as prereg-facts-late: LL-A rank 1 >= 16/20, LL-B control <= 2/20, LL-C trivia kept reported. MISSED closes
this variant.
