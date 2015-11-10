package edu.cuhk.hccl;

import java.io.IOException;
import java.util.HashMap;

public abstract class WordEmbedding {

//	protected static HashMap<String, float[]> wordVecMap = new HashMap<String, float[]>();
	protected String modelPath;

	public WordEmbedding(String modelPath) {
		this.modelPath = modelPath;
	}


	abstract float[] getWordEmbedding(String word);
	
	public abstract void loadWordVectors() throws IOException;
	
}
