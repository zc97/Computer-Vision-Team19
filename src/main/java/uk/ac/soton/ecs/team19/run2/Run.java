package uk.ac.soton.ecs.team19.run2;

import java.io.IOException;
import java.io.PrintWriter;

/**
 *  You should develop a set of linear classifiers (use the LiblinearAnnotator class to automatically create 15 one-vs-all classifiers)
 *  using a bag-of-visual-words feature based on fixed size densely-sampled pixel patches.
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
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
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;
import gov.sandia.cognition.collection.ArrayUtil;

public class Run {

	public static void main(String[] args) {
		try {

			//load datasets
			final String path = "/Users/Zc/Downloads/training";
			final String testingPath = "/Users/Zc/Downloads/testing";
			GroupedDataset<String,VFSListDataset<FImage>,FImage> image_dataset = new VFSGroupDataset<> (path, ImageUtilities.FIMAGE_READER);
			VFSListDataset<FImage> testing = new VFSListDataset<>(testingPath, ImageUtilities.FIMAGE_READER);
			GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String,FImage>(image_dataset, 50, 0, 50);

			//set the patch size and step
			int patchWidth = 10;
			int patchHeight = 10;
			int step = 8;

			//get features patches from every training images and Clustering them
			System.out.println("Start clustering");
			HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(splits.getTrainingDataset(),  patchWidth, patchHeight, step);

			//generate a extractor for extracting features from images
			BOVWExtractor extractor = new BOVWExtractor(assigner, patchWidth, patchHeight, step);

			//training the one-vs-all Linear Annotator
			System.out.println("Start Training");
			LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
				extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
			ann.train(splits.getTrainingDataset());

			//evaluating accuracy and output run2.txt
			System.out.println("Start evaluating");
			Evaluator eval = new Evaluator(ann, splits.getTestDataset());
			eval.printSummary();
			eval.writeToFile("run2.txt", testing);

		} catch (Exception e) {
			
		}
	}

	/**
	 * use patches to train cluster
	 * @param data training data
	 * @param width patch width
	 * @param height patch height
	 * @param step step size of cropping
	 * @return
	 */
	static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String,ListDataset<FImage>,FImage> data, int width, int height, int step){
		ArrayList<float[]> patch_array= new ArrayList<>();

		for (final Entry<String, ListDataset<FImage>> entry : data.entrySet()) {
			for (FImage image : entry.getValue()) {
				patch_array.addAll(getPatches(image, width, height, step));
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

	/**
	 * get list of patches from image
	 * @param image target object
	 * @param width patch width
	 * @param height patch height
	 * @param step step size of cropping
	 * @return
	 */
	static List<float[]> getPatches(FImage image, int width, int height, int step){
		ArrayList<float[]> patches = new ArrayList<>();
		for(int i = 0; i< image.getHeight();i=i+step) {
			for(int j = 0; j< image.getWidth();j=j+step) {
				FImage patch = image.extractROI(i, j, width, height);
				float[] vector = patch.getFloatPixelVector();
				//mean-centering
				vector = mean_centring(vector);
				//normalising
				ArrayUtils.normalise(vector);
				patches.add(vector);
			}
		}
		return patches;
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


	//Feature Extractor
	static class BOVWExtractor implements FeatureExtractor<DoubleFV, FImage>{
		HardAssigner<float[], float[], IntFloatPair> assigner;
		int width, height, step;

		public BOVWExtractor(HardAssigner<float[], float[], IntFloatPair> assigner, int width, int height, int step) {
			this.assigner=assigner;
			this.width=width;
			this.height=height;
			this.step=step;
		}
		@Override
		public DoubleFV extractFeature(FImage object) {
			List<float[]> features = getPatches(object, width, height, step);
			BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
			return bovw.aggregateVectorsRaw(features).asDoubleFV();
		}

	}

	//Evaluator
	static class Evaluator{
		ClassificationEvaluator<CMResult<String>, String, FImage> eval;
		LiblinearAnnotator<FImage, String> ann;

    	public Evaluator(LiblinearAnnotator<FImage, String> ann, GroupedDataset<String, ListDataset<FImage>, FImage> testDataset){
			ClassificationEvaluator<CMResult<String>, String, FImage> eval =
					new ClassificationEvaluator<CMResult<String>, String, FImage>(ann, testDataset,  new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
			this.eval = eval;
			this.ann = ann;
		}

		public void printSummary(){
			Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
			CMResult<String> result = eval.analyse(guesses);
			System.out.println(result.getSummaryReport());
		}

		public void writeToFile(String fileName, VFSListDataset<FImage> testing) throws IOException {
			PrintWriter pw = new PrintWriter(fileName);
			for(int i = 0; i<testing.size();i++){
				ScoredAnnotation<String> guess = Collections.max(ann.annotate(testing.get(i)));
				pw.println(testing.getID(i)+" "+ guess.annotation);
				pw.flush();
			}
		}
	}

	
}


