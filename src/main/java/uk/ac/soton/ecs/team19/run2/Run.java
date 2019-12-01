package uk.ac.soton.ecs.team19.run2;

/**
 *  You should develop a set of linear classifiers (use the LiblinearAnnotator class to automatically create 15 one-vs-all classifiers)
 *  using a bag-of-visual-words feature based on fixed size densely-sampled pixel patches.
 */

import java.util.ArrayList;
import java.util.Map.Entry;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import gov.sandia.cognition.collection.ArrayUtil;

public class Run {

	public static void main(String[] args) {
		try {
			// VFSGroupDataset<FImage> image_dataset = new VFSGroupDataset<FImage>("/Users/mty/Downloads/training", ImageUtilities.FIMAGE_READER);
			// Set<Entry<String, VFSListDataset<FImage>>> alldata =  image_dataset.entrySet();
			
			//HardAssigner<byte[], float[], IntFloatPair> assigner 
			VFSListDataset<FImage> all_image_list = new VFSListDataset<FImage>("/Users/mty/Downloads/training", ImageUtilities.FIMAGE_READER);
			// for (final Entry<String, VFSListDataset<FImage>> entry : image_dataset.entrySet()) {
			// 	VFSListDataset<FImage> image_list = entry.getValue();
			// 	for(final FImage image: image_list){
			// 		all_image_list.add(image);
			// 	}
			// }

			DisplayUtilities.display("test",all_image_list);
			
			
		} catch (Exception e) {
			
		}
	}
	
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(VFSListDataset<FImage> sample){
		ArrayList<float[]> patch_array= new ArrayList<>();
		for (FImage image : sample) { 
			//more patches
			/*
			for(int i =0; i<image.getHeight();i=i+3) {
				for(int j = 0; j<image.getWidth(); j=j+3) {
					FImage out = image.extractROI(i, j, 8, 8);
					float[] vector = patch.getFloatPixelVector();
					//mean-centering
					vector = mean_centring(vector);
					ArrayUtils.normalise(vector);
					patch_array.add(vector);
					
				}
			}
			*/
			
			//less patches
			for(int i = 0; i< image.getHeight();i=i+11) {
				for(int j = 0; j< image.getWidth();j=j+11) {
					FImage patch = image.extractROI(i, j, 8, 8);	
					float[] vector = patch.getFloatPixelVector();
					//mean-centering
					vector = mean_centring(vector);
					ArrayUtils.normalise(vector);
					patch_array.add(vector);
				}
			}

		}
		float[][] allpatches = new float[patch_array.size()][];
		for(int i = 0; i<patch_array.size();i++){
			float[] row = patch_array.get(i);
			allpatches[i]= row;
		}

		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(500);
		FloatCentroidsResult result = km.cluster(allpatches);

		return result.defaultHardAssigner(); 
	}
	


	static class BOVWExtractor implements FeatureExtractor<DoubleFV, FImage>{
		HardAssigner<byte[], float[], IntFloatPair> assigner;
		
		public BOVWExtractor(HardAssigner<byte[], float[], IntFloatPair> assigner) {
			this.assigner=assigner;
		}
		@Override
		public DoubleFV extractFeature(FImage object) {
			
			BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);
			return null;
		}
		
    }
	
	
	/**
	 * mean centring
	 * @param patch
	 * @return float[][]
	 */
    static float[] mean_centring(float[] patch){
		float sum =0;
		for(int i = 0;i<patch.length;i++){
			sum = sum + patch[i];
		}
		float mean = sum/patch.length;

		for(int i = 0; i<patch.length;i++){
			patch[i]= patch[i]-mean;
		}
        return patch;
    }
	
	
}

