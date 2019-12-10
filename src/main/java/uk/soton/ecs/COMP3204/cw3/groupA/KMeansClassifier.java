package uk.soton.ecs.COMP3204.cw3.groupA;

import org.openimaj.image.FImage;
import org.openimaj.data.DataSource;
import org.openimaj.feature.FloatFV;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.util.pair.IntFloatPair;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.data.FloatArrayBackedDataSource;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

import java.io.File;
import java.util.List;
import java.io.FileWriter;
import java.util.Optional;
import java.io.IOException;
import java.util.ArrayList;
import java.io.BufferedWriter;
import java.util.stream.Collectors;

/**
 * @author team 14
 */
public class KMeansClassifier {
    private static final int CLUSTERSIZE = 500;
    private PatchImageFeature imgFeature;
    private LiblinearAnnotator linearAnnotator;
    private List<String> finalR;
    private Data dataF;

    final GroupedDataset<String, VFSListDataset<FImage>, FImage>trainingDataset;
    final VFSListDataset<FImage> testDataset;


    /**
     * Constructor
     * @param data training data, testing data, destination file
     * @throws IOException
     */
    public KMeansClassifier(Data data) throws IOException {
        imgFeature = new PatchImageFeature();

        dataF = data;
        dataF.initData();

        trainingDataset = dataF.getTrainingDataset();
        testDataset = dataF.getTestingDataset();
    }

    /**
     *  Extract a set of features
     *  Annotate each feature
     * @param dataset the dataset used to extract a set of features
     */
    public void train(GroupedDataset<String, VFSListDataset<FImage>, FImage> dataset){
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(dataset, imgFeature);
        System.out.println(("Training assigner"));
        linearAnnotator = new LiblinearAnnotator(
                new LibLinearExtractor(assigner, imgFeature),
                LiblinearAnnotator.Mode.MULTICLASS,
                SolverType.L2R_L2LOSS_SVC,
                1.0,
                0.00001);
        linearAnnotator.train(dataset);
    }


    /**
     * Classify an image
     * @param img image to be classified
     * @return the classification result
     */
    public Optional predict(FImage img){
        return linearAnnotator.classify(img)
                .getPredictedClasses().stream().findFirst();
    }


    /**
     * Train the classifier with training data
     * Classify each image from the testing data
     *
     * @return the list of predictions
     */
    public List<String> run() throws IOException {
    	finalR = new ArrayList<>();
        System.out.println("[*] Starting run2.....");
        train(trainingDataset);
        System.out.println("[*] Training complete.....");

        List<Optional> resultsPredictions = testDataset.stream()
                .map(this::predict)
                .collect(Collectors.toList());
        System.out.println("[*] Predictions complete.....");


        for(int i = 0; i < testDataset.size(); i++){
            String fullPath = testDataset.getID(i);
            String fileName = fullPath.substring(fullPath.lastIndexOf("/") + 1);
            Optional label = resultsPredictions.get(i);

            if(label.isPresent()){
            	finalR.add(fileName + " " + label.get());
            }

        }

        return finalR;
    }


    /**
     * Perfoms K-Means clustering on SIFT features sample
     * @param dataSet dataset of images
     * @param patchImg
     * @return
     */
    private static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
            GroupedDataset<String, ? extends ListDataset<FImage>, FImage> dataSet,
            PatchImageFeature patchImg)
    {
        int a=0;
        List<float[]> allPatches = new ArrayList<>();

        for(FImage img : dataSet){
            a=a+1;
            System.out.println(a + " / " + dataSet.numInstances());
            List<LocalFeature<SpatialLocation, FloatFV>> patchesFound = patchImg.getPatches(img);
            List<float[]> featuresToVector = patchesFound.stream()
                    .map(patch -> patch.getFeatureVector().getVector())
                    .collect(Collectors.toList());


            allPatches.addAll(featuresToVector);
        }


        if(allPatches.size() > 10000) allPatches = allPatches.subList(0, 10000);

        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(CLUSTERSIZE);
        DataSource<float[]> dataSource = new FloatArrayBackedDataSource(allPatches.toArray(new float[][]{}));
        System.out.println("Clustering");
        FloatCentroidsResult res = km.cluster(dataSource);
        return res.defaultHardAssigner();
    }

}
