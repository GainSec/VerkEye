// Decompile one function from an already analyzed Ghidra program.
// @category VerkEye

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;

public class DecompileAt extends GhidraScript {
    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length != 1) {
            throw new IllegalArgumentException("usage: DecompileAt <address>");
        }

        Address address = toAddr(args[0]);
        Function function = currentProgram.getFunctionManager()
            .getFunctionContaining(address);
        if (function == null) {
            throw new IllegalStateException(
                "no analyzed function contains " + address
            );
        }

        DecompInterface decompiler = new DecompInterface();
        decompiler.toggleCCode(true);
        decompiler.toggleSyntaxTree(true);
        if (!decompiler.openProgram(currentProgram)) {
            throw new IllegalStateException(
                "failed to open program: " + decompiler.getLastMessage()
            );
        }

        DecompileResults results = decompiler.decompileFunction(
            function,
            300,
            monitor
        );
        if (!results.decompileCompleted()) {
            throw new IllegalStateException(
                "decompilation failed: " + results.getErrorMessage()
            );
        }

        println("VERKEYE_FUNCTION=" + function.getName());
        println("VERKEYE_ENTRY=" + function.getEntryPoint());
        println(results.getDecompiledFunction().getC());
        decompiler.dispose();
    }
}
