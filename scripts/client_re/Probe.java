// Probe: string xrefs, decompile, callers. Args: outfile cmd1 cmd2 ...
//  str:TEXT  -> xrefs to exact C string TEXT (+ decompile referencing functions)
//  sref:TEXT -> xrefs only
//  fn:ADDR   -> decompile function containing ADDR
//  callers:ADDR -> callers of function containing ADDR
//  dis:ADDR:N -> disassemble N instructions
import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.*;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.mem.*;
import ghidra.program.model.symbol.*;
import java.io.*;
import java.util.*;

public class Probe extends GhidraScript {
    DecompInterface di;
    PrintWriter out;
    Set<Address> done = new HashSet<>();

    String decomp(Function f) {
        DecompileResults r = di.decompileFunction(f, 120, monitor);
        if (r == null || !r.decompileCompleted()) return "  <decompile failed>\n";
        return r.getDecompiledFunction().getC();
    }
    List<Address> findStr(String s) throws Exception {
        List<Address> res = new ArrayList<>();
        byte[] pat = (s + "\0").getBytes("ISO-8859-1");
        Memory mem = currentProgram.getMemory();
        for (MemoryBlock b : mem.getBlocks()) {
            if (!b.isInitialized()) continue;
            Address a = b.getStart();
            while (a != null && a.compareTo(b.getEnd()) <= 0) {
                Address f = mem.findBytes(a, b.getEnd(), pat, null, true, monitor);
                if (f == null) break;
                byte prev = 1;
                try { prev = mem.getByte(f.subtract(1)); } catch (Exception e) {}
                if (prev == 0) res.add(f);
                a = f.add(1);
            }
        }
        return res;
    }
    void xrefs(Address a, boolean dec) {
        Reference[] refs = getReferencesTo(a);
        out.println("  string @" + a + " refs=" + refs.length);
        for (Reference r : refs) {
            Function f = getFunctionContaining(r.getFromAddress());
            out.println("    from " + r.getFromAddress() + " in " + (f == null ? "?" : f.getName() + "@" + f.getEntryPoint()));
            if (dec && f != null && done.add(f.getEntryPoint())) {
                out.println("----- decomp " + f.getEntryPoint());
                out.println(decomp(f));
            }
        }
    }
    public void run() throws Exception {
        String[] args = getScriptArgs();
        out = new PrintWriter(new FileWriter(args[0], true));
        di = new DecompInterface();
        di.openProgram(currentProgram);
        for (int i = 1; i < args.length; i++) {
            String c = args[i];
            out.println("===== " + c);
            if (c.startsWith("str:") || c.startsWith("sref:")) {
                String s = c.substring(c.indexOf(':') + 1);
                for (Address a : findStr(s)) xrefs(a, c.startsWith("str:"));
            } else if (c.startsWith("fn:")) {
                Address a = toAddr(c.substring(3));
                Function f = getFunctionContaining(a);
                if (f == null) { createFunction(a, null); f = getFunctionAt(a); }
                if (f == null) { out.println("no function"); continue; }
                out.println("----- decomp " + f.getEntryPoint());
                out.println(decomp(f));
            } else if (c.startsWith("callers:")) {
                Function f = getFunctionContaining(toAddr(c.substring(8)));
                if (f == null) { out.println("no function"); continue; }
                for (Reference r : getReferencesTo(f.getEntryPoint())) {
                    Function g = getFunctionContaining(r.getFromAddress());
                    out.println("  " + r.getReferenceType() + " from " + r.getFromAddress() + " in " + (g == null ? "?" : g.getEntryPoint()));
                }
            } else if (c.startsWith("dis:")) {
                String[] p = c.split(":");
                Instruction ins = getInstructionAt(toAddr(p[1]));
                if (ins == null) ins = getInstructionAfter(toAddr(p[1]));
                for (int k = 0; k < Integer.parseInt(p[2]) && ins != null; k++) {
                    out.println("  " + ins.getAddress() + "  " + ins);
                    ins = ins.getNext();
                }
            }
            out.flush();
        }
        out.close();
    }
}
