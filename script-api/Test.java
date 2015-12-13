import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import javax.script.ScriptEngineFactory;
import javax.script.ScriptException;
import java.io.File;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.IOException;

class Test {
    public static void main(String[] args) {
	ScriptEngineManager manager = new ScriptEngineManager();
	try {
	    File file = new File(args[0]); 
	    FileReader scriptFile = new FileReader(file);
	    ScriptEngine engine = manager.getEngineByName("python");
	    if (engine != null) {
		try {
		    engine.eval(scriptFile);
		} catch (ScriptException ex) {
		    ex.printStackTrace();
		}
	    }
	} catch (FileNotFoundException e) {
	    System.out.println(e);
	}
	
    }
}
