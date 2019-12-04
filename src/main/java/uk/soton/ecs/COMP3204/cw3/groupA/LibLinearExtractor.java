package uk.soton.ecs.COMP3204.cw3.groupA;

import org.openimaj.image.FImage;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.util.pair.IntFloatPair;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;

import java.util.List;

final class LibLinearExtractor implements FeatureExtractor<SparseIntFV, FImage> {
    private PatchImageFeature patchImg;
    private HardAssigner<float[], float[], IntFloatPair> assigner;

    public LibLinearExtractor(HardAssigner<float[],float[], IntFloatPair> assigner, PatchImageFeature patchImg){
        this.patchImg = patchImg ;
        this.assigner = assigner ;
    }


    @Override
    public SparseIntFV extractFeature(FImage img) {
        BagOfVisualWords<float[]> bovw = new BagOfVisualWords<>(this.assigner);
        List<LocalFeature<SpatialLocation, FloatFV>> features = patchImg.getPatches(img);

        return bovw.aggregate(features);
    }
}
