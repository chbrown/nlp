// Copyright 2013 Christopher Brown, MIT Licensed
package nlp.cb
import org.rogach.scallop._
import scala.util.Random
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer

import edu.stanford.nlp.parser.lexparser.{LexicalizedParser, Options}
import edu.stanford.nlp.io.{NumberRangeFileFilter, RegExFileFilter}
import edu.stanford.nlp.trees.{Tree, DiskTreebank, MemoryTreebank}
import edu.stanford.nlp.ling.{HasTag, HasWord, TaggedWord}

object ActiveLearner {
  // run-main nlp.cb.Tagger
  // --train-dir $PENN/parsed/mrg/wsj/ --test-dir $PENN/parsed/mrg/wsj/
  class TreebankWrapper(trees: Seq[Tree]) {
    def toTreebank = {
      val treebank = new MemoryTreebank()
      treebank.addAll(0, trees)
      treebank
    }
  }
  implicit def wrapTreebank(trees: Seq[Tree]) = new TreebankWrapper(trees)

  // def getInputSentence(t: Tree, options: Options, tagger: (Array[_ >: HasWord] => Array[_ >: HasWord])): Array[_ >: HasWord] = {
  //   if (options.testOptions.forceTags) {
  //     if (options.testOptions.preTag) {
  //       val words = t.yieldWords().toArray
  //       tagger.apply(words)
  //     } else if(options.testOptions.noFunctionalForcing) {
  //       val s = t.taggedYield()
  //       for (word <- s) {
  //         var has_tag = word.asInstanceOf[HasTag]
  //         has_tag.setTag(has_tag.tag().split("-").head)
  //       }
  //       s.toArray
  //     } else {
  //       t.taggedYield().toArray
  //     }
  //   } else {
  //     t.yieldWords().toArray
  //   }
  // }

  def main(args: Array[String]) {
    val opts = Scallop(args.toList)
      .opt[String]("train-dir")
      .opt[String]("test-dir")
      .verify

    val penn = sys.env("PENN")

    val options = new Options()
    options.doDep = false
    options.doPCFG = true
    options.setOptions("-goodPCFG", "-evals", "tsv")

    def step2 = {
      val trainTreebank = options.tlpParams.diskTreebank()
      // var trainTreebank = new DiskTreebank()
      trainTreebank.loadPath(opts[String]("train-dir"), new NumberRangeFileFilter(200, 270, true))

      val testTreebank = new MemoryTreebank()
      testTreebank.loadPath(opts[String]("test-dir"), new NumberRangeFileFilter(2000, 2100, true))

      val parser = LexicalizedParser.trainFromTreebank(trainTreebank, options)
      parser.parserQuery().testOnTreebank(testTreebank)
    }


    // step3
    // Create an initial training set, an "unlabeled" training pool for active learning, and a test set. To create the initial training set, extract the first 50 sentences from section 00. For the unlabeled training set, concatenate sections 01-03 of WSJ. This will give you roughly 4500 additional potential training sentences (approximately 100,000 words). For testing, use WSJ section 20.

    val wsj_00 = new MemoryTreebank()
    wsj_00.loadPath(penn+"/parsed/mrg/wsj/00")
    wsj_00.textualSummary
    // the Stanford NLP is pretty awesome, because if I don't run the textualSummary, training the parser will break later
    val initial = wsj_00.toSeq.take(50)

    val unlabeled = new MemoryTreebank()
    unlabeled.loadPath(penn+"/parsed/mrg/wsj/01")
    unlabeled.loadPath(penn+"/parsed/mrg/wsj/02")
    unlabeled.loadPath(penn+"/parsed/mrg/wsj/03")
    unlabeled.textualSummary

    val test = new MemoryTreebank()
    test.loadPath(penn+"/parsed/mrg/wsj/20")
    test.textualSummary

    def step4(iterations: Int, sentences_per_iteration: Int = 60) {
      // Using the ParserDemo.java class as a example, develop a simple command line interface to the LexicalizedParser that includes support for active learning. Your package should train a parser on a given training set and evaluate it on a given test set, as with the bundled LexicalizedParser. Additionally, choose a random set of sentences from the "unlabeled" training pool whose word count totals approximately 1500 (this represents approximately 60 additional sentences of average length). Output the original training set plus the annotated versions of the randomly selected sentences as your next training set. Output the remaining "unlabeled" training instances as your next "unlabeled" training pool. Lastly, collect your results for this iteration, including at a minimum the following:

      val results = ListBuffer[Map[String, Any]]()
      options.testOptions.verbose = false

      var next_initial = initial
      var next_unlabeled = unlabeled.toSeq

      // def activeLearn(initial: Seq[Tree], unlabeled: Seq[Tree], iteration: Int): (Seq[Tree], Seq[Tree]) = {
      for (iteration <- 1 to iterations) {
        val parser = LexicalizedParser.trainFromTreebank(next_initial.toTreebank, options)

        val (unlabeled_selection, unlabeled_remainder) = Random.shuffle(next_unlabeled).splitAt(sentences_per_iteration)
        var training_words = 0
        val unlabeled_selection_reparsed = unlabeled_selection.map { unlabeled_tree =>
          val words = unlabeled_tree.yieldHasWord()
          training_words += words.size
          parser.apply(words)
        }

        val retrained_parser = LexicalizedParser.trainFromTreebank(next_initial.toTreebank, options)
        results += Map(
          "Iteration" -> iteration,
          "Training words" -> training_words,
          "Sample selection function" -> "random",
          "PCFG F1 score" -> parser.parserQuery().testOnTreebank(test)
        )

        // update
        next_initial = next_initial ++ unlabeled_selection_reparsed
        next_unlabeled = unlabeled_remainder
      }

      println(List.fill(80)("=").mkString)
      val columns = List(
        ("Iteration", "%d"),
        ("Training words", "%d"),
        ("Sample selection function", "%s"),
        ("PCFG F1 score", "%.5f")
      )
      val table = Table(columns, ", ")
      table.printHeader()
      for (result <- results)
        table.printLine(result)
    }
    step4(20)


  }

  // def activeLearn(initial: Seq[Tree], unlabeled: Seq[Tree], count: Int, options: Options) = {
  //   val parser = LexicalizedParser.trainFromTreebank(initial.toTreebank, options)
  //   val (unlabeled_selection, unlabeled_remainder) = Random.shuffle(unlabeled).splitAt(count)
  //   var training_words = 0
  //   val unlabeled_selection_reparsed = unlabeled_selection.map { unlabeled_tree =>
  //     val words = unlabeled_tree.yieldHasWord()
  //     training_words += words.size
  //     parser.apply(words)
  //   }
  //   (initial ++ unlabeled_selection_reparsed, unlabeled_remainder, training_words)
  // }

}