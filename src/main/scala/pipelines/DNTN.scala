package pipelines

import loader.DataLoader
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import activation.ActivationEnum._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Uniform
import breeze.numerics.{sigmoid, tanh}
import org.apache.spark.rdd.RDD
import pipelines.DNTN.DataConfig

class DNTN(sc: SparkContext, dataLoader: DataLoader, conf: DataConfig) extends Serializable {

  /** 3rd order tensor representation for weight matrix slices */
  type Tensor = Seq[DenseMatrix[Double]]

  /** representation of model to train per relation type - word embedding matrix is global */
  case class RelationModel(W: Tensor, V: DenseMatrix[Double], b: DenseVector[Double], U: DenseVector[Double])

  case class RelationModelDecodeInfo(WShape: (Int, Int, Int), VShape: (Int, Int), bShape: Int, UShape: Int)

  val data = sc.parallelize(dataLoader.data)

  final val RANDOM_EMBEDDING_INIT_RANGE = 1e-4
  final val RANDOM_TENSOR_INIT_RANGE = 1.0 / math.sqrt(2 * conf.embeddingSize)
  final val numWords = dataLoader.wordIndices.size
  final val numEntities = dataLoader.entityDict.size
  final val numRelations = dataLoader.relationsDict.size

  /* Randomly initialize the word embedding vectors */
  var embeddingMatrix: DenseMatrix[Double] = {
    val randomEmbedDistribution = Uniform(-RANDOM_EMBEDDING_INIT_RANGE, RANDOM_EMBEDDING_INIT_RANGE)
    DenseMatrix.rand(numWords, conf.embeddingSize, randomEmbedDistribution)
  }

  val initModels: RDD[RelationModel] = sc.parallelize((1 to numRelations).map(relationIndex => {
    val randomTensorDistribution = Uniform(-RANDOM_TENSOR_INIT_RANGE, RANDOM_TENSOR_INIT_RANGE)

    // weight tensor
    val W = (1 to conf.sliceSize).map(sliceIndex => {
      DenseMatrix.rand(conf.embeddingSize, conf.embeddingSize, randomTensorDistribution)
    })
    // standard feedforward layer weights applied to concatenated entity vectors
    val V = DenseMatrix.zeros[Double](conf.sliceSize, 2 * conf.embeddingSize)
    // bias vector
    val b = DenseVector.zeros[Double](conf.sliceSize)
    // scoring vector dotted with output of activation function
    val U = DenseVector.ones[Double](conf.sliceSize)
    RelationModel(W, V, b, U)
  }), numSlices = numRelations)

  /**
    * Roll all model parameters to be trained into a flattened param vector, theta
    */
  def rollToTheta(model: RelationModel) : (DenseVector[Double], RelationModelDecodeInfo) = {
    // flatten the tensor: horizontally concat all tensor slices, then flatten them all into one vector
    val rolledW: DenseVector[Double] = model.W.map(_.toDenseVector).reduceLeft(DenseVector.vertcat(_, _))
    val rolledV = model.V.toDenseVector

    // supply parameter dimensions for reconstructing the model from theta
    val decodeInfo = RelationModelDecodeInfo((model.W.length, model.W.head.rows, model.W.head.cols),
      (model.V.rows, model.V.cols), model.b.length, model.U.length)
    (DenseVector.vertcat(rolledW, rolledV, model.b, model.U), decodeInfo)
  }

  /**
    * Unroll theta into the model for cost calculations
    */
  def unrollFromTheta(theta: DenseVector[Double], decodeInfo: RelationModelDecodeInfo) : RelationModel = {
    // get ranges [x, y) to initially partition the rolled parameter vector
    val WRange = 0 until (decodeInfo.WShape._1 * decodeInfo.WShape._2 * decodeInfo.WShape._3)
    val VRange = WRange.end until (WRange.end + decodeInfo.VShape._1 * decodeInfo.VShape._2)
    val bRange = VRange.end until (VRange.end + decodeInfo.bShape)
    val URange = bRange.end until (bRange.end + decodeInfo.UShape)

    // construct tensor: create matrix where each row represents a tensor slice, then transform
    // each slice into the matrix
    val WMat: DenseMatrix[Double] = new DenseMatrix[Double](decodeInfo.WShape._2 * decodeInfo.WShape._3, decodeInfo.WShape._1, theta(WRange).toArray)
    val W: Seq[DenseMatrix[Double]] = (0 until WMat.cols).map(i => {
      val col = WMat(::, i)
      new DenseMatrix[Double](decodeInfo.WShape._2, decodeInfo.WShape._3, col.toArray)
    })

    // construct other parameters and the final model
    val V = new DenseMatrix[Double](decodeInfo.VShape._1, decodeInfo.VShape._2, theta(VRange).toArray)
    val b = theta(bRange)
    val U = theta(URange)

    RelationModel(W, V, b, U)
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

    logInfo("Theta sample!")
    val model = dntn.initModels.first()
    val (theta, _) = dntn.rollToTheta(model)
    logInfo("Length: " + theta.length)
    logInfo(theta.toString)
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
