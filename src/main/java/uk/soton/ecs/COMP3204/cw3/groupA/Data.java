package uk.soton.ecs.COMP3204.cw3.groupA;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystemException;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.transform.AffineSimulation;

public class Data {
	
	String CURRENT_WORKING_DIRECTORY = System.getProperty("user.dir");
	File RUN1_RESULT 			     = new File(CURRENT_WORKING_DIRECTORY+"/run1.txt");
	File RUN2_RESULT 			     = new File(CURRENT_WORKING_DIRECTORY+"/run2.txt");
	File RUN3_RESULT 			     = new File(CURRENT_WORKING_DIRECTORY+"/run3.txt");
	File TESTING_DATA;
	File TRAINING_DATA;
	
	private GroupedDataset<String, ListDataset<FImage>, FImage> trainingDataset     = null;
	private VFSListDataset<FImage>                              testDataset         = null;
	
	public Data(String training, String testing) {
		TRAINING_DATA = new File(training);
		TESTING_DATA  = new File(testing);
	}
	
	public GroupedDataset<String, ListDataset<FImage>, FImage> getTrainingDataset() {
		return trainingDataset;
	}

	public VFSListDataset<FImage> getTestingDataset(){
		return testDataset;
	}
	
	public File getRun1File() {
		return RUN1_RESULT;
	}
	
	public File getRun2File() {
		return RUN2_RESULT;
	}
	
	public File getRun3File() {
		return RUN3_RESULT;
	}
	
	public File getTestingFile() {
		return TESTING_DATA;
	}
	
	public File getTrainingFile() {
		return TRAINING_DATA;
	}
	
	/**
	 * Get datasets from directories
	 * 
	 * @throws IOException
	 * @throws FileSystemException
	 */
	public void initData()throws IOException, FileSystemException {
		if(!TESTING_DATA.exists()) {
			throw new IOException("Testing file does not exist");
		}
		
		if(!TRAINING_DATA.exists()) {
			throw new IOException("Training file does not exit");
		}
		
		final GroupedDataset<String, VFSListDataset<FImage>, FImage> loadTrainData = new VFSGroupDataset<FImage>(TRAINING_DATA.getPath(), ImageUtilities.FIMAGE_READER);
		trainingDataset = splitTrainingAndValidationData(loadTrainData);
		testDataset = new VFSListDataset<FImage>(TESTING_DATA.getPath(), ImageUtilities.FIMAGE_READER);
	}
	
	
	/**
	 * Splits the Dataset into training and validation
	 * 
	 * @param trainingData
	 * @return
	 */
	public GroupedDataset<String, ListDataset<FImage>, FImage>  splitTrainingAndValidationData(GroupedDataset<String, VFSListDataset<FImage>, FImage> trainingData){
		GroupedDataset<String, ListDataset<FImage>, FImage> trainingDataGeneric = GroupSampler.sample(trainingData, trainingData.size(), false);
		int trainingDataSize = trainingDataGeneric.size();
		
		final int percent80 = (int) Math.round(trainingDataSize * 0.8);
		final int percent20 = (int) Math.round(trainingDataSize * 0.2);
		GroupedRandomSplitter<String, FImage> trainingSplitter = new GroupedRandomSplitter<String, FImage>(trainingData, percent80, percent20, 0);
		
		GroupedDataset<String, ListDataset<FImage>, FImage> trainingDataset = trainingSplitter.getTrainingDataset();
		GroupedDataset<String, ListDataset<FImage>, FImage> validationData = trainingSplitter.getValidationDataset();
		
		//Rotated copies of every image are added to set in order to test the classifier invariance to rotation
		ListDataset<FImage> newImages = new ListBackedDataset<>();
		
		for (final String key : trainingDataset.keySet()){
			newImages.clear();
			for (final FImage image : trainingDataset.getInstances(key)){
				newImages.add(image);
				newImages.add(AffineSimulation.transformImage(image, 0.01f, 1));
				newImages.add(AffineSimulation.transformImage(image, -0.01f, 1));
				newImages.add(AffineSimulation.transformImage(image, 0.02f, 1));
				newImages.add(AffineSimulation.transformImage(image, -0.02f, 1));
			}
			
			trainingDataset.put(key, newImages);
		}
		
		return trainingDataset;
		
	}
}
