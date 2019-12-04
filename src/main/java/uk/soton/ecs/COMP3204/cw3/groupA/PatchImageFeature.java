package uk.soton.ecs.COMP3204.cw3.groupA;

import org.openimaj.image.FImage;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.local.Location;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.feature.local.LocalFeatureImpl;

import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.stream.Stream;
import java.util.stream.LongStream;

public class PatchImageFeature {
    private final int patchSize = 8;
    private final int skipSize = 4;

    List<LocalFeature<SpatialLocation, FloatFV>> getPatches(FImage image)
    {
        List<LocalFeature<SpatialLocation, FloatFV>> patches = new ArrayList<LocalFeature<SpatialLocation, FloatFV>>();

        for (int h = 0; h < image.getHeight(); h += skipSize)
        {
            for (int w = 0; w < image.getWidth(); w += skipSize)
            {
                FImage patch = image.extractROI(h, w, patchSize, patchSize);
                //Mean center and normalise
                float average = patch.sum() / (patchSize * patchSize);
                patch = patch.subtract(average).normalise();
                float[] rawPixels = patch.getFloatPixelVector();
                FloatFV feature = new FloatFV(rawPixels);
                SpatialLocation loc = new SpatialLocation(h,w);
                patches.add(new LocalFeatureImpl<SpatialLocation, FloatFV>(loc, feature));
            }
        }

        return patches;
    }
}
