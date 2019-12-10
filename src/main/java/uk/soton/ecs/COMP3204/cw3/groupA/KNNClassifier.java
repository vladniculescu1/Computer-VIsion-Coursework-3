package uk.soton.ecs.COMP3204.cw3.groupA;

import org.openimaj.image.FImage;
import org.openimaj.util.pair.IntDoublePair;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;

import java.io.File;
import java.util.List;
import java.io.FileWriter;
import java.util.Optional;
import java.io.IOException;
import java.util.ArrayList;
import java.io.BufferedWriter;

/**
 * @author team 14
 */
public class KNNClassifier {
	final float scaleSize = 16.0F;
	final int   K_Value	  = 15;

	final GroupedDataset<String, VFSListDataset<FImage>, FImage>trainingDataset;
	final VFSListDataset<FImage> testDataset;
	final KNNAnnotator  knnAnn;

	private DoubleNearestNeighboursExact knn;

	//The classes of the training feature vectors (array indices correspond to featureVector indices)
	private  List<double[]> features;
	private  List<String> finalR;
	private  Data dataF;


	/**
	 * Constructor
	 * @param data training data, testing data, destination file
	 * @throws IOException
	 */
	public KNNClassifier(Data data) throws IOException {
		dataF 						  = data;
		TinyImageFeature tif 		  = new TinyImageFeature(scaleSize);
		DoubleFVComparison comparator = DoubleFVComparison.EUCLIDEAN;
		
		dataF.initData();
		trainingDataset = dataF.getTrainingDataset();
		testDataset    	= dataF.getTestingDataset();
		knnAnn			= KNNAnnotator.create(tif, comparator, K_Value);
	}

	/**
	 *  Extract a set of features
	 *  Annotate each feature
	 * @param dataset the dataset used to extract a set of features
	 */
	public void train(GroupedDataset<String, VFSListDataset<FImage>, FImage> dataset) {
		knnAnn.train(dataset);
	}


	/**
	 * Classify an image
	 * @param img image to be classified
	 * @return the classification result
	 */
	public Optional predict(FImage img){
		return knnAnn.classify(img).getPredictedClasses().stream().findFirst();
	}


	/**
	 * Train the classifier with training data
	 * Classify each image from the testing data
	 *
	 * @return the list of predictions
	 */
	public List<String> run() throws IOException {
		System.out.println("-------  Start RUN1  -------");
		System.out.println("[*] Loading the Datasets ...");

		int noTraining = trainingDataset.numInstances();
		int noTesting  = testDataset.numInstances();


		System.out.println("[*] Training Set: "+noTraining);
		System.out.println("[*] Testing Set: " +noTesting);

		System.out.println("[*] Training Dataset ...");

		train(trainingDataset);
		finalR = new ArrayList<>();

		System.out.println("[*] Testing Dataset ...");
		for(int i=0; i<testDataset.numInstances();i++) {
			String n = testDataset.getID(i);
			String f = (String) predict(testDataset.get(i)).get();
			finalR.add(n+" "+f);
		}
		return finalR;
	}

}



