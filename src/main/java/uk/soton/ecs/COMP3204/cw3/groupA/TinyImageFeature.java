package uk.soton.ecs.COMP3204.cw3.groupA;

import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.math.statistics.normalisation.PerExampleMeanCenter;
import org.openimaj.math.statistics.normalisation.PerExampleMeanCenterVar;
import org.openimaj.math.statistics.distribution.MultidimensionalHistogram;

import java.io.File;
import java.util.Arrays;
import java.io.IOException;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;

/**
 * @author team 14
 *
 * Implement tiny image feature by cropping the image
 * to square at the centre and then resize it to a fixed
 * resolution.
 *
 * Links to methods used:
 * http://openimaj.org/apidocs/org/openimaj/feature/FeatureExtractor.html
 * http://openimaj.org/apidocs/org/openimaj/image/processing/resize/ResizeProcessor.html
 * http://openimaj.org/apidocs/org/openimaj/feature/DoubleFV.html
 * http://openimaj.org/apidocs/org/openimaj/util/array/ArrayUtils.html
 */

public class TinyImageFeature implements FeatureExtractor<DoubleFV, FImage>{

	float scaleSize;

	/**
	 * Constructor
	 * @param scaleSize the size of the final image
	 */
	public TinyImageFeature(float scaleSize) {
		this.scaleSize = scaleSize;
	}


	/**
	 * @size because the image has to be tiny and
	 * square we take the smallest dimension of it
	 *
	 * @centre crops the image to the centre by half
	 * of the already extracted minimum size
	 *
	 * @scale scales the image to the given size, the
	 * result is be a square image
	 *
	 * @return the image vector into a 1D array
	 */
	public DoubleFV extractFeature(FImage object) {

		int size = Math.min(object.width, object.height);
		FImage centre = object.extractCenter(size, size);
		FImage scale = centre.process(new ResizeProcessor(scaleSize, scaleSize));
		double[] dataD = ArrayUtils.reshape(ArrayUtils.convertToDouble(scale.pixels));
		System.out.println(Arrays.toString(dataD));

		PerExampleMeanCenterVar pemc = new PerExampleMeanCenterVar(0);
		return new DoubleFV(pemc.normalise(dataD));
	}

}
