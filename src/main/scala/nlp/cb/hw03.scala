// Copyright 2013 Christopher Brown, MIT Licensed
package nlp.cb
import org.rogach.scallop._
import scala.util.Random
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

import java.io.{PrintWriter, OutputStream}
import edu.stanford.nlp.parser.lexparser.{LexicalizedParser, TrainOptions, TestOptions, Options, EnglishTreebankParserParams, TreebankLangParserParams}
import edu.stanford.nlp.io.{NumberRangeFileFilter, RegExFileFilter}
import edu.stanford.nlp.trees.{Tree, Treebank, DiskTreebank, MemoryTreebank}
import edu.stanford.nlp.ling.{HasTag, HasWord, TaggedWord}

class NullOutputStream extends OutputStream {
  override def write(b: Int) { }
  override def write(b: Array[Byte]) { }
  override def write(b: Array[Byte], off: Int, len: Int) { }
  override def close() { }
  override def flush() { }
}

class QuietEnglishTreebankParserParams() extends EnglishTreebankParserParams {
  override def pw(): PrintWriter = {
    return new PrintWriter(new NullOutputStream())
  }

  override def pw(o: OutputStream): PrintWriter = {
    return new PrintWriter(new NullOutputStream())
  }

  override def display() { }
}

class QuietTestOptions extends TestOptions {
  override def display() { }
}
class QuietTrainOptions extends TrainOptions {
  override def display() { }
}
class QuietOptions(par: TreebankLangParserParams) extends Options(par) {
  trainOptions = new QuietTrainOptions()
  testOptions = new QuietTestOptions()
  override def display() { }
}

object ActiveLearner {
  class TreebankWrapper(trees: Seq[Tree]) {
    def toTreebank = {
      val treebank = new MemoryTreebank()
      treebank.addAll(0, trees)
      treebank
    }
  }
  implicit def wrapTreebank(trees: Seq[Tree]) = new TreebankWrapper(trees)

  val LOGn2 = math.log(2.0)
  def log2(x: Double) = math.log(x) / LOGn2

  def main(args: Array[String]) {
    val opts = Scallop(args.toList)
      .opt[Int]("initial-labeled", required=true)
      .opt[Int]("iterations", required=true)
      .opt[Int]("sentences-per-iteration", required=true)
      .opt[String]("selection-method", required=true)
      .verify

    val penn = sys.env("PENN")

    val tlp = new QuietEnglishTreebankParserParams()
    val options = new QuietOptions(tlp)
    options.doDep = false
    options.doPCFG = true
    options.setOptions("-goodPCFG", "-evals", "tsv")
    options.testOptions.verbose = false

    // def step2 = {
    //   val trainTreebank = options.tlpParams.diskTreebank()
    //   // var trainTreebank = new DiskTreebank()
    //   trainTreebank.loadPath(opts[String]("train-dir"), new NumberRangeFileFilter(200, 270, true))

    //   val testTreebank = new MemoryTreebank()
    //   testTreebank.loadPath(opts[String]("test-dir"), new NumberRangeFileFilter(2000, 2100, true))

    //   val parser = LexicalizedParser.trainFromTreebank(trainTreebank, options)
    //   parser.parserQuery().testOnTreebank(testTreebank)
    // }

    // step3
    // Create an initial training set, an "unlabeled" training pool for active learning, and a test set. To create the initial training set, extract the first 50 sentences from section 00. For the unlabeled training set, concatenate sections 01-03 of WSJ. This will give you roughly 4500 additional potential training sentences (approximately 100,000 words). For testing, use WSJ section 20.

    val wsj_00 = new MemoryTreebank()
    wsj_00.loadPath(penn+"/parsed/mrg/wsj/00")
    wsj_00.textualSummary
    // the Stanford NLP is pretty awesome, because if I don't run the textualSummary, training the parser will break later
    val initial_count = opts[Int]("initial-labeled")
    println("wsj_00: " + wsj_00.size + " " + initial_count)
    val initial = wsj_00.toSeq.filter { tree =>
      // Stanford parser is so cool that the only way you can determine whether a single sentence
      // will break it is by trying to train a parser on a treebank of just that sentence.
      try {
        LexicalizedParser.trainFromTreebank(List(tree).toTreebank, options)
        true
      } catch {
        case e: java.lang.StringIndexOutOfBoundsException =>
          println("Cannot train on sentence: " + tree.yieldWords().map(_.word()).mkString(" "))
          false
      }
    }.take(initial_count)

    val unlabeled = new MemoryTreebank()
    unlabeled.loadPath(penn+"/parsed/mrg/wsj/01")
    unlabeled.loadPath(penn+"/parsed/mrg/wsj/02")
    unlabeled.loadPath(penn+"/parsed/mrg/wsj/03")
    unlabeled.textualSummary

    val test = new MemoryTreebank()
    test.loadPath(penn+"/parsed/mrg/wsj/20")
    test.textualSummary

    // step4
    val iterations = opts[Int]("iterations")
    val sentences_per_iteration = opts[Int]("sentences-per-iteration")
    // def step4(iterations: Int, : Int) {
    // Using the ParserDemo.java class as a example, develop a simple command line interface to the LexicalizedParser that includes support for active learning. Your package should train a parser on a given training set and evaluate it on a given test set, as with the bundled LexicalizedParser. Additionally, choose a random set of sentences from the "unlabeled" training pool whose word count totals approximately 1500 (this represents approximately 60 additional sentences of average length). Output the original training set plus the annotated versions of the randomly selected sentences as your next training set. Output the remaining "unlabeled" training instances as your next "unlabeled" training pool. Lastly, collect your results for this iteration, including at a minimum the following:

    def printResults(results: Seq[Map[String, Any]]) {
      val sep = List.fill(80)("-").mkString
      println(sep)
      val columns = List(
        ("iteration", "%d"),
        ("added", "%d"),
        ("total", "%d"),
        ("selection", "%s"),
        ("f1", "%.5f"),
        ("time", "%d")
      )
      val table = Table(columns, ", ")
      table.printHeader()
      for (result <- results)
        table.printLine(result)
      println(sep)
    }

    val results = ListBuffer[Map[String, Any]]()

    var next_initial = initial
    var next_unlabeled = unlabeled.toSeq

    val selection_method = opts[String]("selection-method") // "random" "length" "top" "entropy"
    val time_started = System.currentTimeMillis

    // Step 5: Execute 10-20 iterations of your parser for the random selection function, selecting approx 1500 words of additional training data each iteration. You may wish to write a simple test harness script that automates this for you. The random selection function represents a baseline that your more sophisticated sample selection functions should outperform.
    for (iteration <- 1 to iterations) {
      println("Iteration #" + iteration)
      val parser = LexicalizedParser.trainFromTreebank(next_initial.toTreebank, options)

      // we select the next sentences to train on from the beginning of the unlabeled_sorted list
      val unlabeled_sorted = selection_method match {
        case "random" =>
          Random.shuffle(next_unlabeled)
        case "length" =>
          // sortBy+reverse to make the longest first
          next_unlabeled.sortBy(_.yieldHasWord().size).reverse
        case "top" =>
          // sortBy+reverse to put the highest scores first
          next_unlabeled.sortBy { unlabeled_tree =>
            val parserQuery = parser.parserQuery()
            val sentence = unlabeled_tree.yieldHasWord()
            val best_score = if (parserQuery.parse(sentence)) {
              // parserQuery.getPCFGScore() -> lower = less likely!
              parserQuery.getPCFGScore()
            }
            else {
              0.0
            }
            // take the n - 1'th root as a way of normalizing
            math.pow(best_score, 1.0 / (sentence.size - 1))
          }.reverse
        case "entropy" =>
          val k = 20
          val tree_entropies = next_unlabeled.map { unlabeled_tree =>
            val parserQuery = parser.parserQuery()
            val sentence = unlabeled_tree.yieldWords()
            // println("sentence: " + sentence.map(_.word).mkString(" "))
            val tree_entropy = if (parserQuery.parse(sentence)) {
              val top_k_parses = parserQuery.getKBestPCFGParses(k)
              // val top_k_log_probs = top_parses.map(_.score)
              // top_parses(0).score is a log prob, so we exponentiate
              val top_k_probabilities = top_k_parses.map(_.score).map(math.exp)
              val p_sentence = top_k_probabilities.sum
              val top_k_normalized = top_k_probabilities.map(_/p_sentence)

              top_k_normalized.map(p => p * log2(p)).sum * -1
            }
            else {
              Double.PositiveInfinity
            }
            tree_entropy / sentence.size
          }
          // zip up with the entropies for sorting, and then drop them
          // we are seeking low entropy, so don't reverse
          next_unlabeled.zip(tree_entropies).sortBy(_._2).map(_._1)
      }

      val (unlabeled_selection, unlabeled_remainder) = unlabeled_sorted.splitAt(sentences_per_iteration)
      // Oh, wait,
      // val unlabeled_selection_reparsed = unlabeled_selection.map { unlabeled_tree =>
      //   parser.apply(unlabeled_tree.yieldHasWord())
      // }

      // update
      next_initial = next_initial ++ unlabeled_selection
      next_unlabeled = unlabeled_remainder

      // other than going through again and counting the unlabeled_section words,
      // we don't need it anymore. we only keep the reparses.
      var active_training_words = unlabeled_selection.map(_.yieldHasWord().size).sum
      val total_training_words = next_initial.map(_.yieldHasWord().size).sum

      val retrained_parser = LexicalizedParser.trainFromTreebank(next_initial.toTreebank, options)
      results += Map(
        "iteration" -> iteration,
        "added" -> active_training_words,
        "total" -> total_training_words,
        "selection" -> selection_method,
        "f1" -> retrained_parser.parserQuery().testOnTreebank(test),
        "time" -> (System.currentTimeMillis - time_started)
      )

      printResults(results)
    }

    println("Finished all iterations")
    // For reference, the TA's code took the following arguments: --trainBank <file>, --candidateBank <file>, --testBank <file>, --nextTrainBank <file>, --nextCandidatePool <file>, --selectionFunction <random|treeEntropy|...>. For each iteration, the first three arguments were files that were read, and the next two were filenames that were written to.

  }
}
