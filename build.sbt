name := "homework"

version := "0.3.3"

scalaVersion := "2.10.1"

resolvers ++= Seq(
  "Sonatype-snapshots" at "http://oss.sonatype.org/content/repositories/snapshots",
  "apache" at "https://repository.apache.org/content/repositories/releases",
  "gwtwiki" at "http://gwtwiki.googlecode.com/svn/maven-repository/",
  "repo.codahale.com" at "http://repo.codahale.com",
  "repo.scalanlp.org" at "http://repo.scalanlp.org/repo",
  "opennlp.sourceforge.net" at "http://opennlp.sourceforge.net/maven2"
)

libraryDependencies ++= Seq(
  "commons-lang" % "commons-lang" % "2.6",
  "com.codahale" % "jerkson_2.9.1" % "0.5.0",
  "junit" % "junit" % "4.8.2",
  "bsh" % "bsh" % "2.0b4",
  "jgrapht" % "jgrapht" % "0.6.0",
  "jwnl" % "jwnl" % "1.3.3",
  "trove" % "trove" % "2.0.4",
  "mtj" % "mtj" % "0.9.9",
  "org.jdom" % "jdom" % "1.1",
  "org.rogach" %% "scallop" % "0.8.0",
  "edu.stanford.nlp" % "stanford-corenlp" % "1.3.4",
  "edu.stanford.nlp" % "stanford-parser" % "2.0.4"
)

scalacOptions ++= Seq("-unchecked", "-deprecation")

resourceDirectory in Compile <<= baseDirectory { _ / "src" }

//javaOptions += "-Xmx2G"
