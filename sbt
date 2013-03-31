#!/bin/sh
java -Xmx10G -XX:MaxPermSize=512M -jar sbt-launch.jar "$@"
