"""From an annotated table-loader decompile: field name -> struct offset (getInt(this, "Name", dflt); store).

Works for the data loaders (CharacterData, buffs, area effects) and, with --globals, the globals loader.
Usage: fields.py ANNOTATED_FILE [--globals]
"""
import re, sys

L = open(sys.argv[1]).read().split('\n')
if '--globals' in sys.argv:
    for i, l in enumerate(L):
        m = re.search(r'(?:func_0x0072003d8e50|FUN_72003d8e50)\(\w+,"([A-Za-z0-9_ ]+)"\)', l)
        if not m: continue
        for j in range(i + 1, min(i + 14, len(L))):
            if '3d8e50' in L[j]: print(m.group(1), '?'); break
            n = re.search(r'\*\((\w+) \*\)\(\(long\)param_1 \+ (0x[0-9a-f]+|\d+)\) = ', L[j])
            k = re.search(r'\*\((\w+) \*\)\(param_1 \+ (0x[0-9a-f]+|\d+)\) = ', L[j])
            if n: print(m.group(1), hex(int(n.group(2), 0)), n.group(1)); break
            if k: print(m.group(1), hex(int(k.group(2), 0) * 8), k.group(1)); break
    sys.exit()
for i, l in enumerate(L):
    m = re.search(r'\(param_1,"([A-Za-z0-9_]+)",([^)]*)\)', l)
    if not m: continue
    for j in range(i, min(i + 4, len(L))):
        n = re.search(r'\*\((\w+) \*\)\(param_1 \+ (0x[0-9a-f]+|\d+)\) = ', L[j])
        if n: print(m.group(1), hex(int(n.group(2), 0)), n.group(1), 'default=' + m.group(2)); break
    else:
        print(m.group(1), '?')
