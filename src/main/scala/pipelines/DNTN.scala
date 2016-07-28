package pipelines

import loader.DataLoader
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import activation.ActivationEnum._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{Gaussian, Uniform}
import breeze.numerics.{sigmoid, tanh}
import org.apache.spark.rdd.RDD
import pipelines.DNTN.DataConfig

class DNTN(sc: SparkContext, dataLoader: DataLoader, conf: DataConfig) {

  /** 3rd order tensor representation for weight matrix slices */
  type Tensor = Seq[DenseMatrix[Double]]

  val data = sc.parallelize(dataLoader.data)

  final val RANDOM_INIT_RANGE = 1e-4
  final val RANDOM_TENSOR_INIT_RANGE = 1.0 / math.sqrt(2 * conf.embeddingSize)
  final val numWords = dataLoader.wordIndices.size
  final val numEntities = dataLoader.entityDict.size
  final val numRelations = dataLoader.relationsDict.size

  /* Randomly initialize the word embedding vectors */
  val embeddingMatrix: DenseMatrix[Double] = {
    val randomDistribution = Uniform(-RANDOM_INIT_RANGE, RANDOM_INIT_RANGE)
    DenseMatrix.rand(numWords, conf.embeddingSize, randomDistribution)
  }

  /*
   Collection of weight tensor slices that form the tensor
   (W in the paper)
  */
  val W: RDD[Tensor] = {
    val randomDistribution = Uniform(-RANDOM_TENSOR_INIT_RANGE, RANDOM_TENSOR_INIT_RANGE)
    sc.parallelize((0 to numRelations).map(relationIndex => {
      (0 to conf.sliceSize).map(sliceIndex => {
        DenseMatrix.rand(conf.embeddingSize, conf.embeddingSize, randomDistribution)
      })
    }))
  }

  /*
    Standard neural net weight matrices applied to concatenated entity vectors (2 * embeddingSize by numRelations)
    (V in the paper)
   */
  val V: RDD[DenseMatrix[Double]] = {
    sc.parallelize((0 to numRelations).map(relationIndex => {
      DenseMatrix.zeros[Double](conf.sliceSize, 2 * conf.embeddingSize)
    }))
  }

  /* bias vector */
  val b: RDD[DenseVector[Double]] = {
    sc.parallelize(0 to numRelations).map(_ => {
      DenseVector.zeros[Double](conf.sliceSize)
    })
  }

  /* scoring vector U that is dotted with the output of the activation function */
  val U: RDD[DenseVector[Double]] = {
    sc.parallelize(0 to numRelations).map(_ => {
      DenseVector.ones[Double](conf.sliceSize)
    })
  }

  /**
    * Apply nonlinearity function elementwise to vector
    */
  def activation(vector: DenseVector[Double], funcType: ActivationEnum = TANH) : DenseVector[Double] = {
    if (funcType == TANH) {
      tanh(vector)
    } else {
      sigmoid(vector)
    }
  }

  /**
    * Apply differential of the nonlinear activation function to the vector
    */
  def activationDifferential(vector: DenseVector[Double], funcType: ActivationEnum = TANH) : DenseVector[Double] = {
    if (funcType == TANH) {
      1.0 - (vector :* vector)
    } else {
      vector :* (1.0 - vector)
    }
  }
}

object DNTN extends Logging {
  val appName = "DNTN"

  def run(sc: SparkContext, conf: DataConfig) {
    logInfo("Loading training data")
    val trainingDataLoader = new DataLoader(conf.entitiesLocation, conf.relationsLocation, conf.trainLocation)
    val dntn = new DNTN(sc, trainingDataLoader, conf)


    logInfo("Matrix!")
    logInfo(dntn.embeddingMatrix(::, 0).toString)
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
