package pipelines

import loader.DataLoader
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import activation.ActivationEnum._

object DNTN extends Logging {
  val appName = "DNTN"

  def run(sc: SparkContext, conf: DataConfig) {
    logInfo("Loading training data")
    val trainingData = sc.parallelize(new DataLoader(conf.entitiesLocation, conf.relationsLocation, conf.trainLocation).data)

    logInfo("Sample training data:")
    trainingData.take(5).foreach(f => logInfo(f.toString))
  }

  case class DataConfig(
    trainLocation: String = "data/Wordnet/train.txt", // train data file
    testLocation: String = "data/Wordnet/test.txt",  // test data file
    entitiesLocation: String = "data/Wordnet/entities.txt", // entities dictionary file (entity -> ID)
    relationsLocation: String=  "data/Wordnet/relations.txt", // relations dictionary file (relation -> ID)
    embeddingSize: Int = 100,   // size of a word vector
    sliceSize: Int = 3,         // number of slices in a tensor
    numIterations: Int = 500,   // number of optimization iterations
    batchSize: Int = 20000,     // training batch sample size
    corruptionSize: Int = 10,   // size of corruption samples (for negative samples)
    activationFunction: ActivationEnum = TANH,  // neuron activation function
    regParam: Double = 0.0001,  // lambda regularization parameter
    batchIterations: Int = 5    // optimization iterations per batch
  )

  def parse(args: Array[String]): DataConfig = new OptionParser[DataConfig](appName) {
    head(appName, "0.1")
    //opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    //opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    //opt[Int]("commonFeatures") action { (x,c) => c.copy(commonFeatures=x) }
  }.parse(args, DataConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[2]") // This is a fallback if things aren't set via spark submit.

    val sc = new SparkContext(conf)

    val appConfig = parse(args)
    run(sc, appConfig)

    sc.stop()
  }

}
