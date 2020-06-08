# Hilfsfunktionen für Kapitel 12

from decimal import *

def fixed (dec):
    dec = Decimal(dec)
    r = dec.quantize(Decimal('0.0000001'))
    if r == 0:
        return Decimal('0')
    else:
        return r

def setupFloat ():
    getcontext().prec = 10     # precision
    getcontext().Emin = -90    # min. Exponent
    getcontext().Emax = 108    # max. Exponent
    getcontext().clamp = 1     # means: Emin-prec+1 <= e <= Emax-prec+1

def float (dec):
    return Decimal(0) if dec == 0 else Decimal(dec)

def showFloat (dec):
    dec = float(dec)
    tup = dec.as_tuple()
    dig = tup.digits       # Liste
    return dig, tup.exponent + len(dig) - 1
