// Copyright 2013 Christopher Brown, MIT Licensed
package nlp.cb

import nlp.lm.POSTaggedFile // BigramModel
import java.io.File
import scala.collection.JavaConversions._
import scala.collection.mutable.{ListBuffer, Set => MutableSet, Map => MutableMap}

class UnigramModel(sentences: Iterable[List[String]]) {
  // the original includes counts for <S> and </S> as tokens in the total tokenCount variable. Weird.
  val padded_sentences = sentences.map("<S>" +: _ :+ "</S>")

  val all_unigram_counts = MutableMap("<S>" -> 0, "</S>" -> 0, "<UNK>" -> 0)
  val unigrams = padded_sentences.flatten.map { token =>
    all_unigram_counts.get(token) match {
      case Some(count) =>
        all_unigram_counts(token) = count + 1
        token
      case _ =>
        all_unigram_counts(token) = 0
        all_unigram_counts("<UNK>") = all_unigram_counts("<UNK>") + 1
        "<UNK>"
    }
  }.toList
  val unigram_counts = all_unigram_counts.filter(_._2 > 0)

  val word_count = unigrams.size
  val unigram_probabilities = unigram_counts.mapValues(_.toDouble / word_count)

  def sequenceLogProb(sequence: List[String]): Double = {
    val unigrams = sequence.map { token => if (unigram_counts.contains(token)) token else "<UNK>" }
    unigrams.map { unigram => Math.log(unigram_probabilities(unigram)) }.sum
  }

  def perplexity(sequences: Iterable[List[String]], word_count: Int): Double = {
    val total_log_prob = sequences.map(sequenceLogProb).sum
    Math.exp(-total_log_prob / word_count)
  }

  def fullSentencePerplexity(sentences: Iterable[List[String]]): Double = {
    perplexity(sentences.map("<S>" +: _ :+ "</S>"), sentences.map(_.size).sum + sentences.size)
  }
}

class BigramModel(sentences: Iterable[List[String]]) extends UnigramModel(sentences) {
  // this inherits UnigramModel, which means it runs the whole root body of
  // UnigramModel (the constructor, practically) before it runs this body:
  val bigrams = unigrams.sliding(2).map { grams =>
    (grams(0), grams(1))
  }.filterNot { case (a, b) => a == "</S>" && b == "<S>" }.toIterable
  val bigram_counts = bigrams.groupBy(x => x).mapValues(_.size)
}

class ForwardBigramModel(sentences: Iterable[List[String]],
  unigram_weight: Double = 0.1, bigram_weight: Double = 0.9)
  extends BigramModel(sentences) {

  override def sequenceLogProb(sequence: List[String]): Double = {
    val seq_unigrams = sequence.map { token => if (unigram_counts.contains(token)) token else "<UNK>" }
    val p_unigrams = seq_unigrams.tail.map { unigram => unigram_probabilities(unigram) }

    val seq_bigrams = seq_unigrams.sliding(2).map { grams => (grams(0), grams(1)) }.toList
    val p_bigrams = seq_bigrams.map { case (a, b) =>
      bigram_counts.getOrElse((a, b), 0).toDouble / unigram_counts(a)
    }

    (p_unigrams, p_bigrams).zipped.map { case (p_unigram, p_bigram) =>
      Math.log((p_unigram * unigram_weight) + (p_bigram * bigram_weight))
    }.sum
  }
}

class BackwardBigramModel(sentences: Iterable[List[String]],
  unigram_weight: Double = 0.1, bigram_weight: Double = 0.9)
  extends BigramModel(sentences) {

  override def sequenceLogProb(sequence: List[String]): Double = {
    val seq_unigrams = sequence.map { token => if (unigram_counts.contains(token)) token else "<UNK>" }
    val p_unigrams = seq_unigrams.dropRight(1).map { unigram => unigram_probabilities(unigram) }

    val seq_bigrams = seq_unigrams.sliding(2).map { grams => (grams(0), grams(1)) }.toList
    val p_bigrams = seq_bigrams.map { case (a, b) =>
      bigram_counts.getOrElse((a, b), 0).toDouble / unigram_counts(b)
    }

    (p_unigrams, p_bigrams).zipped.map { case (p_unigram, p_bigram) =>
      Math.log((unigram_weight * p_unigram) + (bigram_weight * p_bigram))
    }.sum
  }
}

class BidirectionalBigramModel(
  sentences: Iterable[List[String]],
  unigram_weight: Double = 0.1,
  bigram_weight: Double = 0.9,
  forward_bigram_weight: Double = 0.5,
  backward_bigram_weight: Double = 0.5)
  extends BigramModel(sentences) {

  override def sequenceLogProb(sequence: List[String]): Double = {
    val seq_unigrams = sequence.map { token => if (unigram_counts.contains(token)) token else "<UNK>" }
    val p_unigrams = seq_unigrams.tail.dropRight(1).map { unigram => unigram_probabilities(unigram) }

    val seq_bigrams = seq_unigrams.sliding(2).map { grams => (grams(0), grams(1)) }.toList
    val p_forward_bigrams = seq_bigrams.map { case (a, b) =>
      bigram_counts.getOrElse((a, b), 0).toDouble / unigram_counts(a)
    }
    val p_backward_bigrams = seq_bigrams.map { case (a, b) =>
      bigram_counts.getOrElse((a, b), 0).toDouble / unigram_counts(b)
    }

    (p_unigrams, p_forward_bigrams, p_backward_bigrams).zipped.map { case (p_unigram, p_forward_bigram, p_backward_bigram) =>
      Math.log(
        (unigram_weight * p_unigram) +
        (bigram_weight *
          (
            (forward_bigram_weight * p_forward_bigram) +
            (backward_bigram_weight * p_backward_bigram)
          )
        )
      )
    }.sum
  }
}

class TrigramModel(
  sentences: Iterable[List[String]],
  unigram_weight: Double = 0.1,
  // bigram_weight: Double = 0.3,
  trigram_weight: Double = 0.9)
  extends BigramModel(sentences) {

  val trigrams = unigrams.sliding(3).map { grams =>
    (grams(0), grams(1), grams(2))
  }.filter {
    case ("</S>", "<S>", _) => false
    case (_, "</S>", "<S>") => false
    case _ => true
  }.toIterable
  val trigram_counts = trigrams.groupBy(x => x).mapValues(_.size)

  override def sequenceLogProb(sequence: List[String]): Double = {
    val seq_unigrams = sequence.map { token => if (unigram_counts.contains(token)) token else "<UNK>" }
    val p_unigrams = seq_unigrams.tail.dropRight(1).map { unigram => unigram_probabilities(unigram) }

    val seq_bigrams = seq_unigrams.sliding(2).map { grams => (grams(0), grams(1)) }.toList
    val p_forward_bigrams = seq_bigrams.map { case (a, b) =>
      bigram_counts.getOrElse((a, b), 0).toDouble / unigram_counts(a)
    }
    // val p_backward_bigrams = seq_bigrams.map { case (a, b) =>
      // bigram_counts.getOrElse((a, b), 0).toDouble / unigram_counts(b)
    // }
    val seq_trigrams = seq_unigrams.sliding(3).map { grams => (grams(0), grams(1), grams(2)) }.toList
    val p_trigrams = seq_trigrams.map { case (a, b, c) =>
      trigram_counts.getOrElse((a, b, c), 0).toDouble / unigram_counts(b)
    }

    (p_unigrams, p_forward_bigrams, p_trigrams).zipped.map { case (p_unigram, p_forward_bigram, p_trigram) =>
      Math.log(
        (unigram_weight * p_unigram) +
        // (bigram_weight * p_forward_bigram) +
        (trigram_weight * p_trigram)
      )
    }.sum
  }
}

object NgramEvaluator {
  def printRaystyleReport(model_name: String, model: UnigramModel, wrapper: List[String] => List[String],
    train_sentences: Iterable[List[String]], test_sentences: Iterable[List[String]]) {
    println(s"\n\n$model_name")
    println(s"# Train Sentences = ${train_sentences.size} (# words = ${train_sentences.map(_.size).sum})")
    println(s"# Test Sentences = ${test_sentences.size} (# words = ${test_sentences.map(_.size).sum})")
    List(("Train", train_sentences), ("Test", test_sentences)).foreach { case(name, sentences) =>
      var sent_perplexity = model.fullSentencePerplexity(sentences)
      var nonterm_perplexity = model.perplexity(sentences.map(wrapper), sentences.map(_.size).sum)
      println(s"""${name}ing...
Perplexity = $sent_perplexity
Word Perplexity = $nonterm_perplexity""")
    }
  }

  def printCsvReport(model_name: String, model: UnigramModel, wrapper: List[String] => List[String],
    train_sentences: Iterable[List[String]], test_sentences: Iterable[List[String]]) {
    List(("train", train_sentences), ("test", test_sentences)).foreach { case(name, sentences) =>
      var sent_perplexity = model.fullSentencePerplexity(sentences)
      var nonterm_perplexity = model.perplexity(sentences.map(wrapper), sentences.map(_.size).sum)
      println(s"$model_name, $name, ${sentences.size}, ${sentences.map(_.size).sum}, $sent_perplexity, $nonterm_perplexity")
    }
  }

  def main(args: Array[String]) = {
    // run-main nlp.cb.NgramEvaluator /Users/chbrown/Dropbox/ut/nlp/data/penn-treebank3/tagged/pos/brown/ 0.1

    val files = args.slice(0, args.length - 1).map(file => new File(file))
    // sentences is a List[List[String]]
    val sentences = POSTaggedFile.convertToTokenLists(files).map(_.toList).toList
    // we can quickly test out a backward model by using the transformation below:
    // val sentences = ...map(_.reverse.toList)...

    val test_fraction = args.last.toDouble
    val test_count = (sentences.size * test_fraction).toInt + 1

    // the given model code takes the training sentences from the first part, test from the second
    val (train_sentences, test_sentences) = sentences.splitAt(sentences.size - test_count)

    // Create all the models, train them, and evaluate.
    // println("model, dataset, sentences, words, sentence_perplexity, nonterminating_perplexity")
    List(
      ("UnigramModel", new UnigramModel(train_sentences), (s: List[String]) => s),
      ("ForwardBigramModel", new ForwardBigramModel(train_sentences), (s: List[String]) => "<S>" +: s),
      ("BackwardBigramModel", new BackwardBigramModel(train_sentences), (s: List[String]) => s :+ "</S>"),
      ("BidirectionalBigramModel", new BidirectionalBigramModel(train_sentences), (s: List[String]) => "<S>" +: s :+ "</S>"),
      ("TrigramModel", new TrigramModel(train_sentences), (s: List[String]) => "<S>" +: s :+ "</S>")
    ).foreach { case (model_name, model, wrapper) =>
      printRaystyleReport(model_name, model, wrapper, train_sentences, test_sentences)
      // printCsvReport(model_name, model, wrapper, train_sentences, test_sentences)
    }
  }
}
