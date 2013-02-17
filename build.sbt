//import AssemblyKeys._

//assemblySettings

//seq(com.github.retronym.SbtOneJar.oneJarSettings: _*)

name := "homework"

version := "0.0.1"

scalaVersion := "2.10.0"

resolvers ++= Seq(
  "Sonatype-snapshots" at "http://oss.sonatype.org/content/repositories/snapshots",
  "apache" at "https://repository.apache.org/content/repositories/releases",
  "gwtwiki" at "http://gwtwiki.googlecode.com/svn/maven-repository/",
  "repo.codahale.com" at "http://repo.codahale.com"
)

libraryDependencies ++= Seq(
  "commons-lang" % "commons-lang" % "2.6",
  "com.codahale" % "jerkson_2.9.1" % "0.5.0"
//  "org.scalanlp" %% "breeze-math" % "0.1",
//  "org.scalanlp" %% "breeze-learn" % "0.1",
//  "org.scalanlp" %% "breeze-process" % "0.1",
//  "org.scalanlp" %% "breeze-viz" % "0.1"
)

scalacOptions ++= Seq("-deprecation") // , "-Ydependent-method-types", "-unchecked"

//jarName in assembly := "tacc-hadoop-assembly.jar"

//mainClass in assembly := None

//test in assembly := {}

//mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
//  {
//    case x => {
//      val oldstrat = old(x)
//      if (oldstrat == MergeStrategy.deduplicate) MergeStrategy.first
//      else oldstrat
//    }
//  }
//}

//mainClass in oneJar := None
