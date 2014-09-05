import net.sf.javaml.core.Instance;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.distance.dtw.DTWSimilarity;

import java.io.FileInputStream;
import java.io.InputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import java.util.Vector;

public class BenchDtw {

    // what a mess!
    static double [] readBinary(String filename, int L) {
        
        InputStream inputStream = null;
        byte data[] = null;
        
        try {
            inputStream = new FileInputStream(filename);
            data = new byte[8*L];
            inputStream.read(data, 0, 8*L);
            inputStream.close();
        } catch(IOException e){
            System.out.println(e);
            System.exit(0);
        }
        
        double result[] = new double[L];
        
        for (int i = 0; i < L; i++) {
            byte bytes[] = new byte[8];
            for (int j = 0; j < 8; j++)
                bytes[j] = data[8*i+j];
            result[i] = ByteBuffer.wrap(bytes).
                        order(ByteOrder.LITTLE_ENDIAN ).getDouble();
        }
        
        return result;
    }

    public static void main(String[] args) {
    
        int M = new Integer(args[2]);
        int N = new Integer(args[3]);

        double[] query = readBinary(args[0], M);
        double[] subject = readBinary(args[1], N);
        
        DTWSimilarity DTW = new DTWSimilarity();
        Instance iquery = new DenseInstance(query);
        
        Vector<DenseInstance> candidates = new Vector<DenseInstance>();
        
        for (int i = 0; i < N-M+1; i++) {
            double candidate[] = new double[M];
            for (int j = 0; j < M; j++)
                candidate[j] = subject[i+j];
            candidates.add(new DenseInstance(candidate));
        }
        
        double time = System.nanoTime();
        
        for (int i = 0; i < N-M+1; i++)
           DTW.measure(iquery, candidates.get(i));

        System.out.println((System.nanoTime()-time)/1E9);

    }

}
