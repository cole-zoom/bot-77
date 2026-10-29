#!/bin/bash
# Ghidra headless probe (Probe.java): decompile / xrefs / callers / disassembly into a file under data/.
# usage: scripts/client_re/probe.sh <proj|proj2> <outfile> cmd...   (cmds: str:TEXT sref:TEXT fn:ADDR callers:ADDR dis:ADDR:N)
#   proj  = fully analysed project (better xrefs; some loaders truncated by wrong no-return guesses)
#   proj2 = imported without analysis (functions created on demand; use for the big table loaders)
# Needs GHIDRA_HOME (e.g. ~/tools/ghidra_12.1.4_PUBLIC) and JAVA_HOME (JDK 21).
HERE="$(cd "$(dirname "$0")" && pwd)"
D="${CR_RE_DIR:-$(cd "$HERE/../.." && pwd)/data/work/ghidra}"
P=$1; shift
NAME=libg2; [ "$P" = proj ] && NAME=libg
GHIDRA_HEADLESS_MAXMEM=${GHIDRA_HEADLESS_MAXMEM:-4G} "$GHIDRA_HOME/support/analyzeHeadless" "$D/$P" $NAME \
  -process libg_rebuilt.so -noanalysis -readOnly -scriptPath "$HERE" -postScript Probe.java "$@" > "$D/probe_last.log" 2>&1
grep -E "ERROR|Exception" "$D/probe_last.log" | grep -v CreateFunction | head -5
