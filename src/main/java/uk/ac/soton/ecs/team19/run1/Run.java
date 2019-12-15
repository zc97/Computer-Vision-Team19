package uk.ac.soton.ecs.team19.run1;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.ImageUtilities;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.util.array.ArrayUtils;

// Testing and evaluation
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.ml.annotation.ScoredAnnotation;
import java.util.Collections;
import java.io.PrintWriter;

/**
 * You should develop a simple k-nearest-neighbour classifier using the
 * “tiny image” feature. The “tiny image” feature is one of the simplest
 * possible image representations. One simply crops each image to a square
 * about the centre, and then resizes it to a small, fixed resolution
 * (we recommend 16x16). The pixel values can be packed into a vector by
 * concatenating each image row. It tends to work slightly better if the
 * tiny image is made to have zero mean and unit length. You can choose
 * the optimal k-value for the classifier.
 */

public class Run
{
    // Tiny image side length
    private static final int TINY_SIZE = 16;

    /*
     * We found 15 to be the optimal k value after trying all k values 1-40
     * 7 times for 4 different training split sizes. ALl values 14-18
     * performed significantly better than the remaining values.
     */
    private static final int K = 15;

    // Train and test
    public static void main(String[] args)
    {
        try
        {
            // Load the datasets
            final String path = "C:/Programs/Java/Computer-Vision-Team19/";
            GroupedDataset<String,VFSListDataset<FImage>,FImage> training =
                new VFSGroupDataset<>(path + "training", ImageUtilities.FIMAGE_READER);
            VFSListDataset<FImage> testing =
                new VFSListDataset<>(path + "testing", ImageUtilities.FIMAGE_READER);

            // Create feature extractor
            FeatureExtractor<DoubleFV, FImage> extractor = new TinyExtractor();

            // true ... Output predictions for testing set
            // false .. Output results for validation set to find the best k-value
            if (true)
            {
                // Create the annotator using the optimal k-value
                KNNAnnotator<FImage, String, DoubleFV> annotator =
                    new KNNAnnotator<>(extractor, DoubleFVComparison.EUCLIDEAN, K);

                // Train using all training images
                annotator.train(training);

                // Output to a text file
                PrintWriter writer = new PrintWriter("run1.txt", "UTF-8");

                // Iterate over testing images
                for (int i = 0; i < testing.size(); i++)
                {
                    // Print the filename and best guess
                    ScoredAnnotation<String> guess = Collections.max(annotator.annotate(testing.get(i)));
                    writer.println(testing.getID(i) + " " + guess.annotation);
                }

                writer.close();
            }
            else
            {
                // Use different split sizes
                for (int p = 80; p < 100; p += 5)
                {
                    System.out.printf("\n\n%d percent\n", p);

                    // Get 7 different results to reduce uncertainty
                    for (int i = 0; i < 7; i++)
                    {
                        System.out.println();

                        // Split the training dataset into training and validation subsets
                        GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String,FImage>(training, p, 0, 100 - p);

                        // Iterate over k-values
                        for (int k = 1; k <= 40; k++)
                        {
                            System.out.printf("K = %d\n", k);

                            // Create an annotator using the k-value to be tested
                            KNNAnnotator<FImage, String, DoubleFV> annotator =
                                new KNNAnnotator<>(extractor, DoubleFVComparison.EUCLIDEAN, k);

                            // Train using the training subset
                            annotator.train(splits.getTrainingDataset());

                            // Test using the validation subset
                            int correct = 0;
                            int total = 0;
                            for (String group : splits.getTestDataset().getGroups())
                            {
                                for (FImage image : splits.getTestDataset().get(group))
                                {
                                    ScoredAnnotation<String> guess = Collections.max(annotator.annotate(image));
                                    // System.out.println(group + " = " + guess.annotation);
                                    total++;
                                    if (group.equals(guess.annotation)) correct++;
                                }
                            }
                            System.out.printf("%d / %d correct\n", correct, total);
                        }
                    }
                }
            }
        }
        catch (Exception e)
        {
            // Display error information
            e.printStackTrace();
        }
    }

    private static class TinyExtractor implements FeatureExtractor<DoubleFV, FImage>
    {
        // Convert image to vectorised tiny image
        public DoubleFV extractFeature(FImage image)
        {
            // Crop the image to make it square
            int size = Math.min(image.width, image.height);
            FImage square = image.extractCenter(size, size);

            // Resize the square to the desired tiny dimensions
            square = square.process(new ResizeProcessor(TINY_SIZE, TINY_SIZE));

            // Normalise tiny image to zero mean and unit length
            double[] fv = ArrayUtils.reshape(ArrayUtils.convertToDouble(square.pixels));
            double[] clonefv = fv.clone();

            double mean = ArrayUtils.sumValues(fv) / fv.length;
            double sd = Math.sqrt((ArrayUtils.sumValuesSquared(ArrayUtils.subtract(clonefv, mean)) / clonefv.length));

            ArrayUtils.subtract(fv, mean);
            ArrayUtils.divide(fv, sd);

            // Return the tiny image as a vector
            return new DoubleFV(fv);
        }
    }
}
