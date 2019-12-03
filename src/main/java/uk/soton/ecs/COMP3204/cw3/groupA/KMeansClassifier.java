package uk.soton.ecs.COMP3204.cw3.groupA;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.FloatArrayBackedDataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class KMeansClassifier {
    private static final int CLUSTERSIZE = 500;
    private PatchImageFeature imgFeature;
    private LiblinearAnnotator linearAnnotator;
    private Data dataF;


    String CURRENT_WORKING_DIRECTORY = System.getProperty("user.dir");
    final String TRAINING_PATH		 = CURRENT_WORKING_DIRECTORY+"/training";
    final String TESTING_PATH  		 = CURRENT_WORKING_DIRECTORY+"/testing";

    final GroupedDataset<String, VFSListDataset<FImage>, FImage>trainingDataset;
    final VFSListDataset<FImage> testDataset;

    public KMeansClassifier() throws IOException {
        imgFeature = new PatchImageFeature();

        dataF = new Data(TRAINING_PATH,TESTING_PATH);
        dataF.initData();

        trainingDataset = dataF.getTrainingDataset();
        testDataset = dataF.getTestingDataset();
    }

    /**
     *
     * @param dataset
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

    public Optional predict(FImage img){
        return linearAnnotator.classify(img)
                .getPredictedClasses().stream().findFirst();
    }


    public void run() throws IOException {
        System.out.println("[*] Starting run2.....");
        train(trainingDataset);
        System.out.println("[*] Training complete.....");

        List<Optional> resultsPredictions = testDataset.stream()
                .map(this::predict)
                .collect(Collectors.toList());
        System.out.println("[*] Predictions complete.....");

        File file = dataF.RUN2_RESULT;
        FileWriter fr = new FileWriter(file, true);
        BufferedWriter br = new BufferedWriter(fr);

        for(int i = 0; i < testDataset.size(); i++){
            String fullPath = testDataset.getID(i);
            String fileName = fullPath.substring(fullPath.lastIndexOf("/") + 1);
            Optional label = resultsPredictions.get(i);

            if(label.isPresent()){
                br.write(fileName + " " + label.get());
                br.newLine();
            }

        }

        br.close();
        fr.close();
        System.out.println("[*] Results written to file.....");

    }

    private static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
            GroupedDataset<String, ? extends ListDataset<FImage>, FImage> dataSet,
            PatchImageFeature patchImg)
    {
        int a=1;
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





    public static void main(String[] args) throws IOException {
        KMeansClassifier kmeans = new KMeansClassifier();
        kmeans.run();

    }

}
