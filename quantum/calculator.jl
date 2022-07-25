wL = ks.ib.fw(0, ks.ib.p)
wR = ks.ib.fw(1, ks.ib.p)
primL = ks.ib.bc(0, ks.ib.p)
primR = ks.ib.bc(1, ks.ib.p)

γ = 5 / 3

KB.prim_conserve([wL[1]; primL[2:3]], 2)

0.784
