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
