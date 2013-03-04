// Copyright 2013 Christopher Brown, MIT Licensed
package nlp.cb

import scala.collection.JavaConversions._
import java.io.{FileReader, FileOutputStream, ObjectOutputStream}

import org.rogach.scallop._
import cc.mallet.fst.{SimpleTagger, TokenAccuracyEvaluator, CRFTrainerByLabelLikelihood, Transducer, HMMSimpleTagger, HMM, HMMTrainerByLikelihood}
import cc.mallet.types.{InstanceList, Instance, Sequence, FeatureSequence, FeatureVectorSequence, FeatureVector, LabelSequence, ArraySequence}
import cc.mallet.pipe.Pipe

class OOVTokenAccuracyEvaluator(instanceLists: Array[InstanceList], descriptions: Array[String])
  extends TokenAccuracyEvaluator(instanceLists, descriptions) {
  def tokensForInstance(instance: Instance) = {
    val input = instance.getData().asInstanceOf[FeatureSequence]
    (0 until input.size).map { i => input.get(i).toString }
  }
  def oovEvaluateInstanceList(transducer: Transducer, instances: InstanceList, vocabulary: Set[String]): (Double, Double) = {
    // returns (all_mean, oov_mean)
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

      (seq.filter { case (i, g, o) => g.equals(o) }.size, seq.size,
        oov_seq.filter { case (i, g, o) => g.equals(o) }.size, oov_seq.size)
    }
    (results.map(_._1).sum.toDouble / results.map(_._2).sum, results.map(_._3).sum.toDouble / results.map(_._4).sum)
  }
}

class FVOOVTokenAccuracyEvaluator(instanceLists: Array[InstanceList], descriptions: Array[String])
  extends OOVTokenAccuracyEvaluator(instanceLists, descriptions) {
  override def tokensForInstance(instance: Instance) = {
    val input = instance.getData().asInstanceOf[FeatureVectorSequence]
    (0 until input.size).map { i =>
      val fv = input.get(i)
      fv.getAlphabet.lookupObject(fv.indexAtLocation(0)).toString
    }
  }
}

object MalletRunner {
  def Instances(sentences: List[String], pipe: Pipe) = {
    val instances = new InstanceList(pipe)
    sentences.map { sentence =>
      instances.add(pipe.pipe(new Instance(sentence, null, null, null)))
    }
    instances
  }
  def mean(xs: Seq[Double]): Double = xs.sum / xs.size

  val cap_min = 'A'
  val cap_max = 'Z'
  val features = List(
    (token: String) => if (token.endsWith("ing")) "GERUND",
    (token: String) => if (token.endsWith("s")) "PLURAL",
    (token: String) => if (token.size < 4) "SHORT",
    (token: String) => if (cap_min <= token(0) && token(0) <= cap_max) "UPPER"
  )

  def addExtras(sentence: String) = {
    // sentence is just a string with \n separators
    sentence.split("\n").map { line =>
      val lines = line.split(" ").toList match {
        case token :: tags =>
          token :: features.flatMap { feature =>
            feature(token) match {
              case tag: String => List(tag)
              case _ => List()
            }
          } ++ tags
        case _ => println("Could not find multiple tags in line")
          List()
      }
      lines.mkString(" ")
    }.mkString("\n")
  }

  def reverseLines(sentence: String) = {
    sentence.split("\n").reverse.mkString("\n")
  }

  def getTrainTestSentences(trainDirectory: String, testDirectory: Option[String],
    trainProportion: Option[Double], randomSeed: Int, extraFeatures: Boolean, reverse: Boolean) = {
    Treebank2Mallet.convertAllInDirectory(trainDirectory)

    val (trainSentences, testSentences) = (testDirectory, trainProportion) match {
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
    val mappers = (if (reverse) reverseLines _ else (x: String) => x) compose
      (if (extraFeatures) addExtras _ else (x: String) => x)
    (trainSentences.map(mappers), testSentences.map(mappers))
  }

  def main(args: Array[String]) = {
    val opts = Scallop(args.toList)
      .opt[String]("train-dir", required=true)
      .opt[String]("test-dir")
      .opt[Boolean]("extras")
      .opt[Boolean]("reverse")
      .opt[Double]("train-proportion")
      .opt[Int]("folds", default=() => Some(1))
      .opt[String]("model", default=() => Some("crf"))
      // .opt[String]("model-file")
      .verify

    val columns = List(
      ("Train", "%s"),
      ("Test", "%s"),
      ("Model", "%s"),
      ("Extras", "%s"),
      ("Folds", "%d"),
      ("Training Accuracy", "%.5f"),
      ("Test Accuracy", "%.5f"),
      ("Training sentences", "%d"),
      ("Testing sentences", "%d"),
      ("OOV Accuracy", "%.5f"),
      ("Train Time", "%.3f"),
      ("Test Time", "%.3f"),
      ("Total Time", "%.3f")
    )
    def cells(values: Map[String, Any], sep: String = " & ") {
      println(columns.map { case (key, formatter) =>
        formatter.format(values(key))
      }.mkString(sep))
    }
    // print headers
    println(columns.map(_._1).mkString(" & "))

    val modelName = opts[String]("model").toLowerCase
    val folds = opts[Int]("folds")
    val iterations = 500
    (0 until folds).foreach { i =>
      val time_started = System.currentTimeMillis

      val simplePipe = if (modelName == "crf")
        new SimpleTagger.SimpleTaggerSentence2FeatureVectorSequence()
      else
        new HMMSimpleTagger.HMMSimpleTaggerSentence2FeatureSequence()

      // this is pretty awesome and keeps the HMM from hitting a nullpointer exception on line 307.
      simplePipe.getTargetAlphabet.lookupIndex("0")

      val (trainSentences, testSentences) = getTrainTestSentences(opts[String]("train-dir"),
        opts.get[String]("test-dir"), opts.get[Double]("train-proportion"), 4288 + i, opts[Boolean]("extras"),
        opts[Boolean]("reverse"))
      // trainSentences.foreach { s => println("Sentence " + s) }
      val trainInstances = Instances(trainSentences, simplePipe)
      // solidify the training vocabulary before we add the test instances
      val trainVocabulary = simplePipe.getDataAlphabet.toArray.map(_.asInstanceOf[String]).toSet
      val testInstances = Instances(testSentences, simplePipe)

      // val trainTestRatio = trainSentences.size.toDouble / testSentences.size

      val evaluator = if (modelName == "crf")
        new FVOOVTokenAccuracyEvaluator(Array(trainInstances, testInstances), Array("Training", "Testing"))
      else
        new OOVTokenAccuracyEvaluator(Array(trainInstances, testInstances), Array("Training", "Testing"))

      // train, test, evaluator, orders, defaultLable, forbidden, allowed, connected, iterations, var, CRF
      val transducer = if (modelName == "crf") {
        SimpleTagger.train(trainInstances, testInstances, null,
          Array(1), "0", "\\s", ".*", true, iterations, 10.0, null)
      }
      else {
        HMMSimpleTagger.train(trainInstances, testInstances, null,
          Array(1), "0", "\\s", ".*", true, iterations, 10.0, null)
        // val hmm = new HMM(trainInstances.getPipe, null)
        // val startName = hmm.addOrderNStates(trainInstances, Array(1), null, "0", "\\s".r.pattern, ".*".r.pattern, true);
        // for (i <- (0 until hmm.numStates))
        //   hmm.getState(i).setInitialWeight(Transducer.IMPOSSIBLE_WEIGHT)
        // hmm.getState(startName).setInitialWeight(0.0)

        // val hmmt = new HMMTrainerByLikelihood(hmm)
        // val converged = hmmt.train(trainInstances, iterations)
        // hmm
      }

      val time_trained = System.currentTimeMillis

      val (training, training_oov) = evaluator.oovEvaluateInstanceList(transducer, trainInstances, Set[String]())
      val (testing, testing_oov) = evaluator.oovEvaluateInstanceList(transducer, testInstances, trainVocabulary)

      val time_ended = System.currentTimeMillis

      val test = opts.get[String]("test-dir") match { case Some(x) => x case _ => "" }
      val result = Map(
        "Train" -> opts[String]("train-dir").trim.split("/").dropWhile(_ != "pos").mkString("/"),
        "Test" -> test.trim.split("/").dropWhile(_ != "pos").mkString("/"),
        "Model" -> modelName,
        "Extras" -> opts[Boolean]("extras"),
        "Folds" -> folds,
        "Training Accuracy" -> training,
        "Test Accuracy" -> testing,
        "Training sentences" -> trainSentences.size,
        "Testing sentences" -> testSentences.size,
        "OOV Accuracy" -> testing_oov,
        "Train Time" -> (time_trained - time_started) / 1000.0),
        "Test Time" -> (time_ended - time_trained) / 1000.0),
        "Total Time" -> (time_ended - time_started) / 1000.0)
      cells(result)
    }

    // save the model file, for whatever
    // val s = new ObjectOutputStream(new FileOutputStream(opts[String]("model-file")))
    // s.writeObject(crf)
    // s.close()
  }
}
