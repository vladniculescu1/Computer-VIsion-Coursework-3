package uk.soton.ecs.COMP3204.cw3.groupA;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.util.array.ArrayUtils;

import java.io.File;
import java.io.IOException;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;

/**
 * @author team XXX
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
	@Override
	public DoubleFV extractFeature(FImage object) {
		// TODO Auto-generated method stub
		
		int size = Math.min(object.width, object.height);
		FImage centre = object.extractCenter(size, size);
		FImage scale = centre.process(new ResizeProcessor(scaleSize, scaleSize));
		return new DoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(scale.pixels)));
	}
	/**
	 * Same method from above but not reshaped into 1D array
	 * 
	 * @param object
	 * @return the cropped and scaled image
	 */
	public FImage extractFeatureImage(FImage object) {
		int size = Math.min(object.width, object.height);
		FImage centre = object.extractCenter(size, size);
		return centre.process(new ResizeProcessor(scaleSize, scaleSize));
	}
	
	public static void main(String args[]) throws IOException {
		TinyImageFeature tif = new TinyImageFeature(16);
		FImage image = ImageUtilities.readF(new File("testing/1000.jpg"));
		DisplayUtilities.display(tif.extractFeatureImage(image));
		System.out.println(tif.extractFeature(image));
		
	}

}
