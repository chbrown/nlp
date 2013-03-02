// Copyright 2013 Christopher Brown, MIT Licensed
package nlp.cb

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConversions._
import java.io.{FileReader, FileOutputStream, ObjectOutputStream}
// import java.util.logging.{Logger, Level}

import org.rogach.scallop._
import cc.mallet.fst.{SimpleTagger, TokenAccuracyEvaluator, CRFTrainerByLabelLikelihood, Transducer, HMMSimpleTagger, HMM, HMMTrainerByLikelihood}
import cc.mallet.types.{InstanceList, Instance, Sequence, FeatureSequence, FeatureVectorSequence, FeatureVector, LabelSequence, ArraySequence}
import cc.mallet.pipe.{Pipe}
import cc.mallet.pipe.iterator.LineGroupIterator

// run-main cc.mallet.fst.SimpleTagger --train true --model-file Atis3.model --training-proportion 0.8 --test lab /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/atis3.pos.mallet
// run-main cc.mallet.fst.HMMSimpleTagger --train true --model-file Atis3.model --training-proportion 0.8 --test lab /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/atis3.pos.mallet

// /Users/chbrown/Dropbox/ut/nlp/cs338-code/mallet-2.0.6/src/cc/mallet/fst
// /Users/chbrown/Dropbox/ut/nlp/homework/src/main/java/mallet-2.0.7/src/cc/mallet/fst

case class FileOut(filePath: String, echo: Boolean = false) {
  val fp = new java.io.File(filePath)
  val writer = new java.io.PrintWriter(fp)
  def println(line: String) {
    writer.println(line)
    if (echo) System.out.println(line)
  }
  def close() {
    writer.flush()
    writer.close()
  }
}

object Treebank2Mallet {
  // Instructions:
  // $ sbt
  // run-main Treebank2Mallet /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/
  val pairRegex = """(\S+)/(\S+)""".r

  def main(args: Array[String]) = {
    convertAllInDirectory(args.last)
  }

  def convertFile(treebankFile: java.io.File, malletFile: java.io.File) {
    // System.err.println("Reading Treebank file: " + treebankFile.getCanonicalPath)
    val malletWriter = new java.io.PrintWriter(malletFile)

    val lines = io.Source.fromFile(treebankFile).getLines.
      filterNot(_.startsWith("[ @")).filterNot(_.startsWith("*x*"))

    var collecting = false
    val sentence = ListBuffer[String]()
    def flush() {
      if (sentence.nonEmpty) {
        malletWriter.println(sentence.mkString("\n"))
        malletWriter.println("\n")
      }
      sentence.clear
    }
    while (lines.hasNext) {
      val line = lines.next
      if (line.startsWith("=======")) {
        collecting = true
        flush()
      }
      if (collecting) {
        val pairs = pairRegex.findAllMatchIn(line).map { pairMatch => (pairMatch.group(1), pairMatch.group(2)) }.toList
        sentence ++= pairs.map { case (token, tag) => token + " " + tag }
        if (pairs.size > 0 && pairs.last._2 == ".") flush()
      }
    }
    flush()

    malletWriter.close()
    // System.err.println("Writing Mallet file: " + malletFile.getCanonicalPath)
  }

  def convertAllInDirectory(directoryPath: String) {
    val directory = new java.io.File(directoryPath)
    for (treebankFile <- directory.listFiles) {
      if (treebankFile.getName.endsWith(".pos")) {
        val malletFile = new java.io.File(treebankFile.getCanonicalPath + ".mallet")
        // if (!malletFile.exists)
        convertFile(treebankFile, malletFile)
      }
    }
  }

  def getSentences(malletFile: java.io.File) = {
    io.Source.fromFile(malletFile).mkString.toLowerCase.split("\n{2,}")
  }
}

class OOVTokenAccuracyEvaluator(instanceLists: Array[InstanceList], descriptions: Array[String])
  extends TokenAccuracyEvaluator(instanceLists, descriptions) {
  def tokensForInstance(instance: Instance): Seq[String] = {
    val input = instance.getData().asInstanceOf[FeatureSequence]
    (0 until input.size).map { i => input.get(i).toString }
  }
  def oovEvaluateInstanceList(transducer: Transducer, instances: InstanceList, vocabulary: Set[String]): (Double, Double) = {
    // val transducer = trainer.getTransducer()
    val results = instances.map { instance =>
      val input = instance.getData().asInstanceOf[Sequence[_]]
      val tokens = tokensForInstance(instance)
      val gold_labels = instance.getTarget().asInstanceOf[LabelSequence]
      val output_labels = transducer.transduce(input).asInstanceOf[ArraySequence[_]]
      if (input.size != gold_labels.size || gold_labels.size != output_labels.size) println("WTF mismatch of input, gold, & output")
      val seq = (0 until input.size).map { i =>
        (tokens(i), gold_labels.get(i), output_labels.get(i))
      }
      val oov_seq = seq.filterNot { case(i, g, o) => vocabulary(i) }
      // println(seq.map(_._1).mkString(" ") + " -> " + oov_seq.map(_._1).mkString(" "))
      // println("  " + oov_seq.size + " of " + seq.size)

      (seq.filter { case (i, g, o) => g.equals(o) }.size, seq.size,
        oov_seq.filter { case (i, g, o) => g.equals(o) }.size, oov_seq.size)
    }
    // results = List((totals, corrects, oov_totals, oov_corrects))
    (results.map(_._1).sum.toDouble / results.map(_._2).sum, results.map(_._3).sum.toDouble / results.map(_._4).sum)
  }
}

class FVOOVTokenAccuracyEvaluator(instanceLists: Array[InstanceList], descriptions: Array[String])
  extends OOVTokenAccuracyEvaluator(instanceLists, descriptions) {
  override def tokensForInstance(instance: Instance): Seq[String] = {
    val input = instance.getData().asInstanceOf[FeatureVectorSequence]
    (0 until input.size).map { i =>
      val fv = input.get(i)
      fv.getAlphabet.lookupObject(fv.indexAtLocation(0)).toString
    }
  }
}


// run-main nlp.cb.MalletRunner --train-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/wsj/mini00 --test-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/wsj/mini01 --folds 10 --model hmm
// run-main nlp.cb.MalletRunner --train-dir /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis --train-proportion 0.8 --folds 10 --model hmm
// run-main MalletRunner --model-file Atis3.model --training-proportion 0.8 --mallet-file /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/atis/atis3.pos.mallet
object MalletRunner {
  def Instances(sentences: List[String], pipe: Pipe): InstanceList = {
    val instances = new InstanceList(pipe)
    sentences.map { sentence =>
      instances.add(pipe.pipe(new Instance(sentence, null, null, null)))
    }
    instances
  }
  def mean(xs: Seq[Double]): Double = xs.sum / xs.size

  def getTrainTestSentences(trainDirectory: String, testDirectory: Option[String],
    trainProportion: Option[Double], randomSeed: Int) = {
    Treebank2Mallet.convertAllInDirectory(trainDirectory)

    (testDirectory, trainProportion) match {
      case (_, Some(trainProportion)) =>
        // test on training data by setting some aside
        val sentences = new java.io.File(trainDirectory).listFiles
          .filter(_.getName.endsWith(".pos.mallet"))
          .flatMap(Treebank2Mallet.getSentences)
          .filterNot(_.trim.size == 0).toList

        // shuffle it up
        val trainingCount = trainProportion * sentences.size
        val rand = new util.Random(randomSeed)
        rand.shuffle(sentences).splitAt(trainingCount.toInt)
      case (Some(testDirectory), _) =>
        // the testing and training folders are totally separate
        val trainSentences = new java.io.File(trainDirectory).listFiles
          .filter(_.getName.endsWith(".pos.mallet"))
          .flatMap(Treebank2Mallet.getSentences)
          .filterNot(_.trim.size == 0).toList

        Treebank2Mallet.convertAllInDirectory(testDirectory)
        val testSentences = new java.io.File(testDirectory).listFiles
          .filter(_.getName.endsWith(".pos.mallet"))
          .flatMap(Treebank2Mallet.getSentences)
          .filterNot(_.trim.size == 0).toList

        (trainSentences, testSentences)
      case _ =>
        (List[String](), List[String]())
    }
  }

  def main(args: Array[String]) = {
    val opts = Scallop(args.toList)
      // .opt[String]("model-file")
      .opt[String]("train-dir", required=true)
      .opt[String]("test-dir")
      .opt[Double]("train-proportion")
      .opt[Int]("folds", default=() => Some(1))
      .opt[String]("model", default=() => Some("crf"))
      .verify

    println("# Corpus & Model & Folds & Training Accuracy & Test Accuracy & Training sentences & Testing sentences & OOV Accuracy & Time (sec)")

    val corpus = opts[String]("train-dir").trim.split("/").dropWhile(_ != "pos").mkString("/")
    val modelName = opts[String]("model").toLowerCase
    val folds = opts[Int]("folds")
    (0 until folds).foreach { i =>

      val started = System.currentTimeMillis
      val simplePipe = if (modelName == "crf")
        new SimpleTagger.SimpleTaggerSentence2FeatureVectorSequence()
      else
        new HMMSimpleTagger.HMMSimpleTaggerSentence2FeatureSequence()
      // this is pretty awesome and keeps the HMM from hitting a nullpointer exception on line 307.
      simplePipe.getTargetAlphabet.lookupIndex("0");

      val (trainSentences, testSentences) = getTrainTestSentences(opts[String]("train-dir"),
        opts.get[String]("test-dir"), opts.get[Double]("train-proportion"), 4288 + i)
      val trainInstances = Instances(trainSentences, simplePipe)
      // solidify the training vocabulary before we add the test instances
      val trainVocabulary = simplePipe.getDataAlphabet.toArray.map(_.asInstanceOf[String]).toSet
      val testInstances = Instances(testSentences, simplePipe)

      // val trainTestRatio = trainSentences.size.toDouble / testSentences.size

      val evaluator = new OOVTokenAccuracyEvaluator(Array(trainInstances, testInstances), Array("Training", "Testing"))

      // train, test, evaluator, orders, defaultLable, forbidden, allowed, connected, iterations, var, CRF
      val iterations = 500
      val transducer = if (modelName == "crf")
        SimpleTagger.train(trainInstances, testInstances, evaluator,
          Array(1), "0", "\\s", ".*", true, iterations, 10.0, null)
      else {
        // HMMSimpleTagger.train(trainInstances, testInstances, evaluator,
        //   Array(1), "0", "\\s", ".*", true, iterations, 10.0, null)
        val hmm = new HMM(trainInstances.getPipe, null);
        val startName = hmm.addOrderNStates(trainInstances, Array(1), null, "0", "\\s".r.pattern, ".*".r.pattern, true);
        for (i <- (0 until hmm.numStates))
          hmm.getState(i).setInitialWeight(Transducer.IMPOSSIBLE_WEIGHT)
        hmm.getState(startName).setInitialWeight(0.0)

        val hmmt = new HMMTrainerByLikelihood(hmm)
        val converged = hmmt.train(trainInstances, iterations)
        hmm
      }

      val (training, training_oov) = evaluator.oovEvaluateInstanceList(transducer, trainInstances, Set[String]())
      val (testing, testing_oov) = evaluator.oovEvaluateInstanceList(transducer, testInstances, trainVocabulary)
      // (training, testing, testing_oov)

      val ended = System.currentTimeMillis
      println("%s & %s & %d & %.5f & %.5f & %d & %d & %.5f & %.3f" format
        (corpus, modelName, folds, training, testing, trainSentences.size, testSentences.size, testing_oov, (ended - started) / 1000.0))

    }
    // val (training, testing, testing_oov) = (mean(results.map(_._1)), mean(results.map(_._2)), mean(results.map(_._3)))
    // val (training, testing, oov) = results.unzip( map(_._1).sum / results.size

    // save the model file, for whatever
    // val s = new ObjectOutputStream(new FileOutputStream(opts[String]("model-file")))
    // s.writeObject(crf)
    // s.close()
  }
}
