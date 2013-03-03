// Copyright 2013 Christopher Brown, MIT Licensed
package nlp.cb

import scala.collection.mutable.ListBuffer

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
    // .toLowerCase
    io.Source.fromFile(malletFile).mkString.split("\n{2,}")
  }
}

