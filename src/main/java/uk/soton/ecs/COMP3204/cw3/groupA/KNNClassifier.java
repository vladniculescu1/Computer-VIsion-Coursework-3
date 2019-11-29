package uk.soton.ecs.COMP3204.cw3.groupA;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.image.FImage;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.pair.IntDoublePair;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;


/**
 * 
 * @author team XXX
 *
 * K = sqrt(number of samples in datasets)/2
 * Odd number
 */
public class KNNClassifier {
	final float scaleSize = 16.0F;
	final int   K_Value	  = 15;

	
	String CURRENT_WORKING_DIRECTORY = System.getProperty("user.dir");
	final String TRAINING_PATH		 = CURRENT_WORKING_DIRECTORY+"/training";
	final String TESTING_PATH  		 = CURRENT_WORKING_DIRECTORY+"/testing";
	
	final GroupedDataset<String, VFSListDataset<FImage>, FImage>trainingDataset;
	final VFSListDataset<FImage> testDataset;


	private DoubleNearestNeighboursExact knn;

	//The classes of the training feature vectors (array indices correspond to featureVector indices)
	private  List<String> classes;
	private  List<double[]> features;
	private  List<String> finalR;
	private  Data dataF;

	
	public KNNClassifier() throws IOException {
		dataF = new Data(TRAINING_PATH,TESTING_PATH);
		dataF.initData();
		trainingDataset = dataF.getTrainingDataset();
		testDataset    	= dataF.getTestingDataset();
	}

	/**
	 *
	 * @param dataset
	 */
	public void train(GroupedDataset<String, VFSListDataset<FImage>, FImage> dataset){
		features = new ArrayList<>();
		classes  = new ArrayList<>();

		TinyImageFeature tif = new TinyImageFeature(scaleSize);

		for(String label:dataset.getGroups()){
			for(FImage img: dataset.get(label)){
				double[] fv = tif.extractFeature(img);
				features.add(fv);
				classes.add(label);
			}
		}

		knn = new DoubleNearestNeighboursExact(features.toArray(new double[][]{}));
	}

	/**
	 *
	 * @return
	 */
	public String predict(FImage image, String name) throws IOException {
		TinyImageFeature tif = new TinyImageFeature(scaleSize);
		double[] data = tif.extractFeature(image);

		//Search for neighbours using KNN
		List<IntDoublePair> neighbours = knn.searchKNN(data, K_Value);

		Map<String, Integer> count = new HashMap<>();

		for(IntDoublePair neighbour: neighbours){
			String getClass = classes.get(neighbour.first);
			int cnt =  1;

			if(count.containsKey(getClass)){
				cnt += count.get(getClass);
			}

			count.put(getClass, cnt);
		}

		List<Map.Entry<String, Integer>> guess = new ArrayList<>(count.entrySet());

		//Sort the list
		Collections.sort(guess, (o1, o2) -> o2.getValue().compareTo(o1.getValue()));


		String result = guess.get(0).getKey();

		//System.out.println(result);
		double confidence = guess.get(0).getValue().doubleValue()/ (double) K_Value;
		//System.out.println(confidence);



		BasicClassificationResult<String> resultB = new BasicClassificationResult<>();
		resultB.put(result, confidence);

		return name+" "+result+" "+confidence;
	}

	public void run() throws IOException {
		System.out.println("-------  Start RUN1  -------");
		System.out.println("==> Loading the Datasets ...");

		int noTraining = trainingDataset.numInstances();
		int noTesting  = testDataset.numInstances();


		System.out.println("-> Training Set: "+noTraining);
		System.out.println("-> Testing Set: " +noTesting);

		System.out.println("==> Training Dataset ...");

		train(trainingDataset);
		finalR = new ArrayList<>();


		System.out.println("==> Testing Dataset ...");
		for(int i=0; i<testDataset.numInstances();i++) {
			finalR.add(predict(testDataset.get(i), testDataset.getID(i)));
		}

		File file = dataF.RUN1_RESULT;
		FileWriter fr = new FileWriter(file, true);
		BufferedWriter br = new BufferedWriter(fr);

		String out;
		for(int j=0; j<finalR.size();j++){
			out = "";
			for(int p=0; p<finalR.size();p++){
				if(finalR.get(p).matches(j+".jpg(.*)")){
					out = finalR.get(p);
					break;
				}
			}

			//System.out.println(out);
			if(out!="") {
				br.write(out);
				br.newLine();
			}
		}

		br.close();
		fr.close();

		System.out.println("==> DONE <== CHECK: "+ dataF.getRun1File().getPath());
	}

	public static void main(String args[]) throws IOException {
		KNNClassifier test = new KNNClassifier();
		test.run();
	}

}



