name := "homework"

version := "0.0.2"

scalaVersion := "2.10.0"

resolvers ++= Seq(
  "Sonatype-snapshots" at "http://oss.sonatype.org/content/repositories/snapshots",
  "apache" at "https://repository.apache.org/content/repositories/releases",
  "gwtwiki" at "http://gwtwiki.googlecode.com/svn/maven-repository/",
  "repo.codahale.com" at "http://repo.codahale.com",
  "maven.ontotext.com" at "http://maven.ontotext.com/archiva/repository/public",
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
  "org.rogach" %% "scallop" % "0.8.0"
)

scalacOptions ++= Seq("-deprecation")

resourceDirectory in Compile <<= baseDirectory { _ / "src" }

//javaOptions += "-Xmx2G"
