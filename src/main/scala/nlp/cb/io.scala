// Copyright 2013 Christopher Brown, MIT Licensed
package nlp.cb

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

case class Table(columns: List[(String, String)], separator: String = " & ") {
  def join(values: Map[String, Any]) = {
    columns.map { case (key, fmtstring) =>
      fmtstring.format(values(key))
    }.mkString(separator)
  }
  def printHeader() {
    println(columns.map(_._1).mkString(separator))
  }
  def printLine(values: Map[String, Any]) {
    println(join(values))
  }
}
