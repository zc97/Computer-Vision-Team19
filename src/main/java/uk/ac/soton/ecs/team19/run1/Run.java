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

import org.openimaj.ml.annotation.ScoredAnnotation;
import java.util.Collections;

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

/*
 * This code appears to work.
 *
 * I have not worked out what k value is good.
 *
 * I have not implemented the zero mean and unit length part.
 */

public class Run
{
    // Tiny image side length
    private static final int TINY_SIZE = 16;

    // TODO write comment and pick optimal k-value
    private static final int K = 10;

    // TODO write comment
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

            // Create feature extractor and annotator
            FeatureExtractor<DoubleFV, FImage> extractor = new TinyExtractor();
            KNNAnnotator<FImage, String, DoubleFV> annotator =
                new KNNAnnotator<>(extractor, DoubleFVComparison.EUCLIDEAN, K);

            // Train
            annotator.train(training);

            // true ... Output predictions for testing set
            // false .. Output results for training set
            if (true)
            {
                // Iterate over testing images
                for (int i = 0; i < testing.size(); i++)
                {
                    // Print the filename and best guess
                    ScoredAnnotation<String> guess = Collections.max(annotator.annotate(testing.get(i)));
                    System.out.println(testing.getID(i) + " = " + guess.annotation);
                }
            }
            else
            {
                // Reuse training images for evaluation
                int correct = 0;
                int total = 0;
                for (String group : training.getGroups())
                {
                    for (FImage image : training.get(group))
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

            // TODO Zero mean and unit length?

            // Return the tiny image as a vector
            return new DoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(square.pixels)));
        }
    }
}
