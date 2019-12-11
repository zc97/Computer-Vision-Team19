package uk.ac.soton.ecs.team19.run2;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;
import gov.sandia.cognition.collection.ArrayUtil;

/**
 * You should develop a set of linear classifiers (use the LiblinearAnnotator
 * class to automatically create 15 one-vs-all classifiers) using a
 * bag-of-visual-words feature based on fixed size densely-sampled pixel
 * patches. We recommend that you start with 8x8 patches, sampled every 4
 * pixels in the x and y directions. A sample of these should be clustered
 * using K-Means to learn a vocabulary (try ~500 clusters to start). You might
 * want to consider mean-centring and normalising each patch before clustering
 * /quantisation. Note: we’re not asking you to use SIFT features here - just
 * take the pixels from the patches and flatten them into a vector & then use
 * vector quantisation to map each patch to a visual word.
*/

public class Run {

	public static void main(String[] args) {
		try {

			GroupedDataset<String,VFSListDataset<FImage>,FImage> image_dataset = new VFSGroupDataset<> ("/Users/mty/Downloads/training", ImageUtilities.FIMAGE_READER);
			GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String,FImage>(image_dataset, 50, 0, 50);
			VFSListDataset<FImage> all_image = new VFSListDataset<FImage>("/Users/mty/Downloads/training", ImageUtilities.FIMAGE_READER);

			System.out.println("Start clustering");
			HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(all_image);
			BOVWExtractor extractor = new BOVWExtractor(assigner);

			System.out.println("Start Training");
			LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
				extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
			ann.train(splits.getTrainingDataset());

			System.out.println("Start evaluating");
			ClassificationEvaluator<CMResult<String>, String, FImage> eval =
			new ClassificationEvaluator<CMResult<String>, String, FImage>(ann, splits.getTestDataset(),
			new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
			Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
			CMResult<String> result = eval.analyse(guesses);
			System.out.println(result.getSummaryReport());
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
		HardAssigner<float[], float[], IntFloatPair> assigner;

		public BOVWExtractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
			this.assigner=assigner;
		}
		@Override
		public DoubleFV extractFeature(FImage object) {
			List<float[]> features = getPatches(object);
			BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
			return bovw.aggregateVectorsRaw(features).asDoubleFV();
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

	static List<float[]> getPatches(FImage image){
		List<float[]> featureList = new ArrayList<>();
		for(int i = 0; i< image.getHeight();i=i+11) {
			for(int j = 0; j< image.getWidth();j=j+11) {
				FImage patch = image.extractROI(i, j, 8, 8);
				float[] vector = patch.getFloatPixelVector();
				//mean-centering
				vector = mean_centring(vector);
				ArrayUtils.normalise(vector);
				featureList.add(vector);
			}
		}
		return featureList;
	}

	/**
	 *
	 * @return
	 */
	// static Map<String, FImage> getMapDataset(VFSGroupDataset<FImage> dataset){
	// 	Map<String, FImage> map = new HashMap<>();
	// 	Set<Entry<String, VFSListDataset<FImage>>> entrySet =  dataset.entrySet();
	// 	for (final Entry<String, VFSListDataset<FImage>> entry : image_dataset.entrySet()) {
	// 		VFSListDataset<FImage> image_list = entry.getValue();
	// 		for(final FImage image: image_list){

	// 		}
	// 	}
	// 	return map;
	// }


}
