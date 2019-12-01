package uk.ac.soton.ecs.team19.run2;

/**
 *  You should develop a set of linear classifiers (use the LiblinearAnnotator class to automatically create 15 one-vs-all classifiers)
 *  using a bag-of-visual-words feature based on fixed size densely-sampled pixel patches.
 */

import java.util.ArrayList;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

public class Run {
	
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(VFSListDataset<FImage> sample ){
		ArrayList<float[][]> patchlist = new ArrayList<>();
		for (FImage image : sample) { 
			float[][] patch = new float[8][8];
			//more patches
			/*
			for(int i =0; i<image.getHeight();i=i+3) {
				for(int j = 0; j<image.getWidth(); j=j+3) {
					FImage out = image.extractROI(i, j, image.getWidth(), image.getHeight());
					patch = out.pixels;
					patchlist.add(patch);
					
				}
			}
			*/
			
			//less patches
			for(int i = 0; i< image.getHeight();i=i+11) {
				for(int j = 0; j< image.getWidth();j=j+11) {
					FImage out = image.extractROI(i, j, image.getWidth(), image.getHeight());
					
							
					patch = out.pixels;
					
					patchlist.add(patch);
					
				}
			}
		}
		float[][] a = new float[patchlist.size()][patchlist.size()];
		float[][] allpatches =patchlist.toArray(a);
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
	
	
	public static void main(String[] args) {
		try {
			VFSGroupDataset<FImage> image_dataset = new VFSGroupDataset<FImage>("/Users/mty/Downloads/training", ImageUtilities.FIMAGE_READER);
			//Set<Entry<String, VFSListDataset<FImage>>> alldata =  image_dataset.entrySet();
			
			//HardAssigner<byte[], float[], IntFloatPair> assigner 
			System.out.println(5/4);
			
		} catch (Exception e) {
			
		}
	}
}

