package nlp.lm;

import java.io.*;
import java.util.*;

/**
 *
 * @author Ray Mooney
 * Combines both a forward and backward bigram model to build a language model.
 * Interpolates both models with equal weights.
*/

public class BidirectionalBigramModel {

    /** Forward bigram model */
    public BigramModel bigramModel;

    /** Backward bigram model */
    public BackwardBigramModel backwardBigramModel;

    /** Initialize bidirectional model by initializing both forward and backward
     * models */
    public BidirectionalBigramModel() {
        bigramModel = new BigramModel();
        backwardBigramModel = new BackwardBigramModel();
    }

    /** Train model by training both submodels */
    public void train (List<List<String>> sentences) {
        bigramModel.train(sentences);
        backwardBigramModel.train(sentences);
    }

    /** Use sentences as a test set to evaluate the model. Print out perplexity
     *  of the model for this test data */
    public void test (List<List<String>> sentences) {
        double totalLogProb = 0;
        double totalNumTokens = 0;
        for (List<String> sentence : sentences) {
            totalNumTokens += sentence.size() + 1;
            double sentenceLogProb = sentenceLogProb(sentence);
            totalLogProb += sentenceLogProb;
        }
        double perplexity = Math.exp(-totalLogProb / totalNumTokens);
        System.out.println("Perplexity = " + perplexity );
    }

    /* Compute log probability of sentence given current model */
    public double sentenceLogProb (List<String> sentence) {
        // Compute vector of token probabilities for forward and backward models
        double[] probs = bigramModel.sentenceTokenProbs(sentence);
        double[] backwardProbs = backwardBigramModel.sentenceTokenProbs(sentence);
        // Maintains sum of total sentence log prob
        double sentenceLogProb = 0;
        assert (probs.length == backwardProbs.length) && (probs.length == sentence.size()+1);
        // Add to sentence log prob for each token prediction log prob
        for (int i = 0; i < sentence.size(); i++) {
            // Prob of i'th token in sentence for forward model
            double probToken = probs[i];
            // Prob of i'th token in sentence for backward model
            // Counts back from end skipping the sentence-start prob at the end
            // since these are in reverse order
            double backwardProbToken = backwardProbs[backwardProbs.length - i - 2];
            // Average the forward and backward probs to get the interpolated prob for
            // this i'th token
            // System.out.println(probToken + " : " + backwardProbToken);
            double logProb = Math.log((probToken + backwardProbToken) / 2);
            sentenceLogProb += logProb;
        }
        // Find forward model's prediction prob for sentence end
        double endSentenceProb = probs[sentence.size()];
        // Find backward model's prediction prob for sentence start
        double startSentenceProb = backwardProbs[sentence.size()];
        // Average these two as the log prob for sentence boundary prediciton
        sentenceLogProb += Math.log((endSentenceProb + startSentenceProb) / 2);
        return sentenceLogProb;
    }

    /** Like test1 but excludes predicting end-of-sentence when computing perplexity */
    public void test2 (List<List<String>> sentences) {
        double totalLogProb = 0;
        double totalNumTokens = 0;
        for (List<String> sentence : sentences) {
            totalNumTokens += sentence.size();
            double sentenceLogProb = sentenceLogProb2(sentence);
            //      System.out.println(sentenceLogProb + " : " + sentence);
            totalLogProb += sentenceLogProb;
        }
        double perplexity = Math.exp(-totalLogProb / totalNumTokens);
        System.out.println("Word Perplexity = " + perplexity );
    }

    /** Like sentenceLogProb but excludes predicting end-of-sentence when computing perplexity */
    public double sentenceLogProb2 (List<String> sentence) {
        double[] probs = bigramModel.sentenceTokenProbs(sentence);
        double[] backwardProbs = backwardBigramModel.sentenceTokenProbs(sentence);
        double sentenceLogProb = 0;
        assert (probs.length == backwardProbs.length) && (probs.length == sentence.size()+1);
        for (int i = 0; i < sentence.size(); i++) {
            double probToken = probs[i];
            double backwardProbToken = backwardProbs[backwardProbs.length - i - 2];
            double logProb = Math.log((probToken + backwardProbToken) / 2);
            sentenceLogProb += logProb;
        }
        return sentenceLogProb;
    }



    /** Train and test a bidirectional bigram model.
     *  Command format: "nlp.lm.BidirectionalBigramModel [DIR]* [TestFrac]" where DIR
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
                           " (# words = " + BigramModel.wordCount(trainSentences) +
                           ") \n# Test Sentences = " + testSentences.size() +
                           " (# words = " + BigramModel.wordCount(testSentences) + ")");
        BidirectionalBigramModel model = new BidirectionalBigramModel();
        System.out.println("Training...");
        model.train(trainSentences);
        model.test(trainSentences);
        model.test2(trainSentences);
        System.out.println("Testing...");
        model.test(testSentences);
        model.test2(testSentences);
    }

}
