package nlp.lm;

import java.io.*;
import java.util.*;

/**
 *
 * @author Ray Mooney
 * Version of bigram language model that models the sequence right to left instead
 * of left to right.  Just reverses sentences and then call normal BigramModel.
*/

public class BackwardBigramModel extends BigramModel {

    /** Accumulate unigram and backward bigram counts for these sentences */
    public void trainSentence (List<String> sentence) {
        List<String> reverseSentence = new ArrayList<String>(sentence);
        Collections.reverse(reverseSentence);
        super.trainSentence(reverseSentence);
    }

    /* Compute log probability of sentence given current backward model */
    public double sentenceLogProb (List<String> sentence) {
        List<String> reverseSentence = new ArrayList<String>(sentence);
        Collections.reverse(reverseSentence);
        return super.sentenceLogProb(reverseSentence);
    }

    /** Like sentenceLogProb but excludes predicting start-of-sentence when computing perplexity */
    public double sentenceLogProb2 (List<String> sentence) {
        List<String> reverseSentence = new ArrayList<String>(sentence);
        Collections.reverse(reverseSentence);
        return super.sentenceLogProb2(reverseSentence);
    }

    /** Returns vector of probabilities of predicting each token in the sentence
     *  including the start of sentence. Order starts from end of sentence and goes
     *  to start */
    public double[] sentenceTokenProbs (List<String> sentence) {
        List<String> reverseSentence = new ArrayList<String>(sentence);
        Collections.reverse(reverseSentence);
        return super.sentenceTokenProbs(reverseSentence);
    }


    /** Train and test a backward bigram model.
     *  Command format: "nlp.lm.BackwardBigramModel [DIR]* [TestFrac]" where DIR
     *  is the name of a file or directory whose LDC POS Tagged files should be
     *  used for input data; and TestFrac is the fraction of the sentences
     *  in this data that should be used for testing, the rest for training.
     *  0 < TestFrac < 1
     *  Uses the last fraction of the data for testing and the first part
     *  for training.
     */
    public static void main(String[] args) throws IOException {
        File[] files = new File[args.length - 1];
        for (int i = 0; i < files.length; i++)
            files[i] = new File(args[i]);
        double testFraction = Double.valueOf(args[args.length -1]);
        List<List<String>> sentences =  POSTaggedFile.convertToTokenLists(files);
        int numSentences = sentences.size();
        int numTest = (int)Math.round(numSentences * testFraction);
        List<List<String>> testSentences = sentences.subList(numSentences - numTest, numSentences);
        List<List<String>> trainSentences = sentences.subList(0, numSentences - numTest);

        System.out.println("# Train Sentences = " + trainSentences.size() +
                           " (# words = " + wordCount(trainSentences) +
                           ") \n# Test Sentences = " + testSentences.size() +
                           " (# words = " + wordCount(testSentences) + ")");
        // Create a bigram model and train it.
        BigramModel model = new BackwardBigramModel();
        System.out.println("Training...");
        model.train(trainSentences);
        // Test on training data using test and test2
        model.test(trainSentences);
        model.test2(trainSentences);
        System.out.println("Testing...");
        // Test on test data using test and test2
        model.test(testSentences);
        model.test2(testSentences);
    }

}
