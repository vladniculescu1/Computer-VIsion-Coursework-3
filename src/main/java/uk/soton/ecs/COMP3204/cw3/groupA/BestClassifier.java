package uk.soton.ecs.COMP3204.cw3.groupA;

import java.util.List;
import java.util.Optional;
import java.io.IOException;
import java.util.ArrayList;
import java.nio.file.FileSystemException;

import org.openimaj.image.FImage;
import org.openimaj.data.DataSource;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.util.pair.IntFloatPair;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.feature.global.Gist;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;

import de.bwaldvogel.liblinear.SolverType;

public class BestClassifier {
	private LiblinearAnnotator<FImage, String> annotation;
	private List<String> finalR;
	private Data data;
	
	public BestClassifier(Data data) throws FileSystemException, IOException {
		this.data = data;
	}
	
	public static void main(String[] args) throws FileSystemException, IOException {
		//BestClassifier bc = new BestClassifier();
		//bc.run();
	}

	/**
	 * 
	 * @return
	 */
	public List<String> run(){
		GroupedDataset<String, ListDataset<FImage>, FImage> dataSet = GroupSampler.sample(data.getTrainingDataset(), data.getTrainingDataset().size(), false);
		VFSListDataset<FImage> testDataset				            = data.getTestingDataset();
		
		finalR 		 = new ArrayList<>();
		int sizeTest = testDataset.size();
		
		System.out.println("[*] Start Training...");
		train(dataSet);
		//System.out.println("[*] Training Done");
		
		for(int i=0; i<sizeTest; i++) {
			FImage image = testDataset.get(i);
			String label = testDataset.getID(i);
			
			ClassificationResult<String> predicted = classify(image);
			//String[] classes 					   = predicted.getPredictedClasses().toArray(new String[]{});
			String predict 		       			   = (String)predicted.getPredictedClasses().stream().findFirst().get();
			
			//System.out.println(label);
			System.out.println("> "+i+" "+predict);
			
			//for(String pred: classes) {
			finalR.add(label+" "+predict);
				//System.out.print(pred+" ");
			//}
			//System.out.println();
		}
		
		return finalR;
	}
	
	/**
	 * 
	 * @param sample
	 * @param sift
	 * @return
	 */
	public HardAssigner<byte[], float[], IntFloatPair> assigner(Dataset<FImage> sample, PyramidDenseSIFT<FImage> sift){
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
		
		System.out.println("[*] Start training - SIFT...");
		int i = 0;
		for(FImage image: sample) {
			sift.analyseImage(image);
			allkeys.add(sift.getByteKeypoints(0.005f));
			System.out.println("\n" + (++i));
		}
		
		System.out.println("\n"+"[*] Training Done - SIFT");
		
		ByteKMeans bkm = ByteKMeans.createKDTreeEnsemble(600);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		
		System.out.println("[*] Start Clustering - SIFT...");
		ByteCentroidsResult result = bkm.cluster(datasource);
		System.out.println("[*] Clustering Done - SIFT");
		
		return result.defaultHardAssigner();
	}
	
	/**
	 * 
	 *
	 *
	 */
	class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage>{
		final PyramidDenseSIFT<FImage> sift;
		final HardAssigner<byte[], float[], IntFloatPair> assigner;
		
		public PHOWExtractor(PyramidDenseSIFT<FImage> sift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
			this.sift 	  = sift;
			this.assigner = assigner;
		}
		/**
		 * @bovw Implementation of an object capable of 
		 * extracting basic (hard-assignment) Bag of Visual Words 
		 * (BoVW) representations of an image given a list of local features 
		 * and an HardAssigner with an associated codebook.
		 * 
		 * @spatial performs spatial pooling of local 
		 * features by grouping the local features into 
		 * non-overlapping, fixed-size spatial blocks
		 */
		@Override
		public DoubleFV extractFeature(FImage object) {
			// TODO Auto-generated method stub
			BagOfVisualWords<byte[]> bovw 						= new BagOfVisualWords<byte[]>(assigner);
			BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(bovw, 4, 4);
			Gist<FImage> gist 									= new Gist<FImage>(256, 256);
			FImage image 				  						= object.getImage();
			
			sift.analyseImage(image);
			
		    DoubleFV finalR   = spatial.aggregate(sift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
		    gist.analyseImage(object);
		    DoubleFV response = gist.getResponse().normaliseFV();
		    
			return finalR.concatenate(response);
		}
		
	}
	
	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingDataset) {
		DenseSIFT dsift 			    			 = new DenseSIFT(5, 5);
		PyramidDenseSIFT<FImage> pdSift 			 = new PyramidDenseSIFT<FImage>(dsift, 6f, 2,5);
		GroupedRandomSplitter<String, FImage> split  = new GroupedRandomSplitter<String, FImage>(trainingDataset, 4, 0, 0);
		HomogeneousKernelMap hkm 					 = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		
		HardAssigner<byte[], float[], IntFloatPair> assigner = assigner(split.getTrainingDataset(), pdSift);
		FeatureExtractor<DoubleFV, FImage> extractor 		 = hkm.createWrappedExtractor(new PHOWExtractor(pdSift, assigner));
		
		annotation = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC_DUAL, 1.0, 0.00001);
		
		System.out.println("[*] Start Training...");
		annotation.train(trainingDataset);
		System.out.println("[*] Training Done");
	}
	
	public ClassificationResult<String> classify(FImage image) {
		return annotation.classify(image);
	}

}
