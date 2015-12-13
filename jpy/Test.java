import java.io.InputStream;
import java.io.IOException;

public class Test {
	public static void main(String[] args) {
		try {
			ProcessBuilder pb = new ProcessBuilder("python", args[0]);
			Process process = pb.start();
			
			InputStream stream = process.getInputStream();
			while (true) {
				int c = stream.read();
				if (c == -1) {
					stream.close();
					break;
				}
                System.out.print((char)c);
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}
