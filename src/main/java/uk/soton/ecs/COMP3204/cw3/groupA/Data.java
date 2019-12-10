package uk.soton.ecs.COMP3204.cw3.groupA;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystemException;

import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.image.processing.transform.AffineSimulation;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;

/**
 * @author team 14
 */
public class Data {
	
	String CURRENT_WORKING_DIRECTORY = System.getProperty("user.dir");
	File RUN1_RESULT 			     = new File(CURRENT_WORKING_DIRECTORY+"/run1.txt");
	File RUN2_RESULT 			     = new File(CURRENT_WORKING_DIRECTORY+"/run2.txt");
	File RUN3_RESULT 			     = new File(CURRENT_WORKING_DIRECTORY+"/run3.txt");
	File TESTING_DATA;
	File TRAINING_DATA;
	
	private GroupedDataset<String, VFSListDataset<FImage>, FImage> trainingDataset     = null;
	private VFSListDataset<FImage>                                 testDataset         = null;

	/**
	 * Constructor
	 * @param training the path to the training dataset
	 * @param testing the path to the testing dataset
	 * @throws FileSystemException
	 * @throws IOException
	 */
	public Data(String training, String testing) throws FileSystemException, IOException {
		TRAINING_DATA = new File(training);
		TESTING_DATA  = new File(testing);
		
		initData();
	}

	/**
	 *
	 * @return the dataset fetched from its location
	 */
	public GroupedDataset<String, VFSListDataset<FImage>, FImage> getTrainingDataset() {
		return trainingDataset;
	}

	/**
	 *
	 * @return the dataset fetched from its location
	 */
	public VFSListDataset<FImage> getTestingDataset(){
		return testDataset;
	}

	/**
	 *
	 * @return the path to the outputted file
	 */
	public File getRun1File() {
		return RUN1_RESULT;
	}

	/**
	 *
	 * @return  the path to the outputted file
	 */
	public File getRun2File() {
		return RUN2_RESULT;
	}

	/**
	 *
	 * @return  the path to the outputted file
	 */
	public File getRun3File() {
		return RUN3_RESULT;
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
		trainingDataset = loadTrainData;
		testDataset = new VFSListDataset<FImage>(TESTING_DATA.getPath(), ImageUtilities.FIMAGE_READER);
	}

}
