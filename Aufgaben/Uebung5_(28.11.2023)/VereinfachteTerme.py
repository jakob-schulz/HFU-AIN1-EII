#Die Wertevergabe fuer a, b, c, d und v erforlgt ueber das Terminal
print("Bitte geben Sie fuer wahr = 1 und fuer falsch = 0 ein")
if(input("Bitte geben sie einen Wert fuer a ein:") == "0"):
    a = False
else:
    a = True
if(input("Bitte geben sie einen Wert fuer b ein:") == "0"):
    b = False
else:
    b = True
if(input("Bitte geben sie einen Wert fuer c ein:") == "0"):
    c = False
else:
    c = True
if(input("Bitte geben sie einen Wert fuer d ein:") == "0"):
    c = False
else:
    c = True
if(input("Bitte geben sie einen Wert fuer v ein:") == "0"):
    v = False
else:
    v = True

#Die Variable "ergebnis" bekommt die einzelnen booleschen Ausdruecke zugeordnet und wird dann ausgegeben
ergebnis = not a and not b
print("Wenn a = ", a,", b =", b,", c =", c,"und v =", v, ", dann gilt fuer:\n!a*!b = ", ergebnis)
ergebnis = (a or not c) and (b or c)
print("(a+!b)*(b+c) = ", ergebnis)
ergebnis = False
print("0 = ", ergebnis)
ergebnis = a or (not b and not c)
print("a+(!b*!c) = ", ergebnis)
ergebnis = a and not b
print("a*!b = ", ergebnis)
ergebnis = a and not b
print("a*!b = ", ergebnis)
ergebnis = c or d
print("c or d = ", ergebnis)
ergebnis = not a or not c or (not b and v)
print("!a+!c+(!b*v) = ", ergebnis)
ergebnis = a and (b or c)
print("!a+!c+(!b*v) = ", ergebnis)