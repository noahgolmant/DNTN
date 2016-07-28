package loader

import scala.collection.mutable
import scala.io.Source

case class Relation(leftEntity: Int, relation: Int, rightEntity: Int)

/**
  * Created by noah on 7/27/16.
  */
class DataLoader(entityFile: String,
                 relationsFile: String,
                 dataFile: String) {

  private def loadDictionary(filename: String): Map[String, Int] = {
    val fileLines = Source.fromFile(filename).getLines()
    fileLines.zipWithIndex.toMap
  }

  /* Map from an entity string name to its numeric index */
  val entityDict = loadDictionary(entityFile)
  /* Map from a relations name to its numeric index */
  val relationsDict = loadDictionary(relationsFile)

  /* Main data - sequence of training relations of form (leftEntity, relation, rightEntity) */
  val data: Seq[Relation] = {
    Source.fromFile(dataFile).getLines.map(line => {
      val triple = line.split("\\s+")
      assert(triple.length == 3, "Expected sample to have 3 entries: " + line)

      Relation(
        entityDict.get(triple(0)).get,
        relationsDict.get(triple(1)).get,
        entityDict.get(triple(2)).get
      )
    }).toSeq
  }

  /* Map to get an entity by its index */
  val wordIndices: Map[Int, String] = entityDict.map(f => (f._2, f._1))
}
