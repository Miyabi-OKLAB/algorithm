import java.io.InputStream;
import java.io.IOException;

public class p4j {
	public static void main(String[] args) {
		try {
			ProcessBuilder pb = new ProcessBuilder("python", "classification.py");
			Process process = pb.start();

			InputStream stream = process.getInputStream();
			while (true) {
				int c = stream.read();
				if (c == -1) {
					stream.read();
					break;
				}
				System.out.print((char)c);
			}
		}
		catch (IOException ex) {
			ex.printStackTrace();
		}
	}
}
