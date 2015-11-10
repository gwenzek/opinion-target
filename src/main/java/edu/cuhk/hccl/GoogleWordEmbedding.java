package edu.cuhk.hccl;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;

public class GoogleWordEmbedding extends WordEmbedding {

	private static final int MAX_SIZE = 50;
	private HashMap<String, Integer> wordVecMap;
	private float[][] data;

	public GoogleWordEmbedding(String modelPath) {
		super(modelPath);
	}

	/***
	 * The following methods are based on the code gist from
	 * https://gist.github.com/ansjsun/6304960
	 * 
	 * @throws IOException
	 */
	@Override
	public void loadWordVectors() throws IOException {

		DataInputStream dis = null;
		BufferedInputStream bis = null;

		try {
            System.out.println(String.format("Loading vectors from %s", modelPath));
            InputStream fis = new FileInputStream(modelPath);
            if(modelPath.endsWith(".gz"))
                fis = new GZIPInputStream(fis);

			bis = new BufferedInputStream(fis);
			dis = new DataInputStream(bis);
			int words = Integer.parseInt(readString(dis));
			int size = Integer.parseInt(readString(dis));

			this.init(words, size);
			for (int i = 0; i < words; i++) {
				String word = readString(dis);
				wordVecMap.put(word, i);
				double len = 0;
				float[] vectors = data[i];
				for (int j = 0; j < size; j++) {
					float vector = readFloat(dis);
					len += vector * vector;
					vectors[j] = vector;
				}
				len = Math.sqrt(len);

				for (int j = 0; j < vectors.length; j++) {
					vectors[j] = (float) (vectors[j] / len);
				}
				// System.out.println(String.format("Loading vector for word: %s", word));
			}
            System.out.println(String.format("Loaded %d vectors from %s", words, modelPath));

		} finally {
			bis.close();
			dis.close();
		}
	}

	public float[] getWordEmbedding(String word) {
		if(wordVecMap.containsKey(word))
			return data[wordVecMap.get(word)];
		else
			return null;
	}

	protected void init(int vocabSize, int dimensions){
		wordVecMap = new HashMap<String, Integer>(vocabSize);
		data = new float[vocabSize][];
		for(int i = 0; i < vocabSize; ++i)
			data[i] = new float[dimensions];
	}

	private static float bytesToFloat(byte[] b) {
		int accum = 0;
		accum = accum | (b[0] & 0xff) << 0;
		accum = accum | (b[1] & 0xff) << 8;
		accum = accum | (b[2] & 0xff) << 16;
		accum = accum | (b[3] & 0xff) << 24;
		return Float.intBitsToFloat(accum);
	}

	private static String readString(DataInputStream dis) throws IOException {

		byte[] bytes = new byte[MAX_SIZE];
		byte b = dis.readByte();
		int i = -1;
		StringBuilder sb = new StringBuilder();
		while (b != 32 && b != 10) {
			i++;
			bytes[i] = b;
			b = dis.readByte();
			if (i == 49) {
				sb.append(new String(bytes));
				i = -1;
				bytes = new byte[MAX_SIZE];
			}
		}
		sb.append(new String(bytes, 0, i + 1));
		return sb.toString();
	}

	private static float readFloat(InputStream is) throws IOException {

		byte[] bytes = new byte[4];
		is.read(bytes);
		return bytesToFloat(bytes);
	}

}
