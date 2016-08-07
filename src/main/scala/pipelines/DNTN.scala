package pipelines

import loader.{DataLoader, Relation}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import activation.ActivationEnum._
import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.stats.distributions.Uniform
import breeze.stats.mean
import breeze.numerics.{sigmoid, tanh}
import org.apache.spark.rdd.RDD
import pipelines.DNTN.DataConfig

import scala.util.Random


case class RelationModel(W: Seq[DenseMatrix[Double]], V: DenseMatrix[Double], b: DenseVector[Double], U: DenseVector[Double])

class DNTN(sc: SparkContext, dataLoader: DataLoader, conf: DataConfig) extends Serializable {

  /** 3rd order tensor representation for weight matrix slices */
  type Tensor = Seq[DenseMatrix[Double]]

  /** representation of model to train per relation type - word embedding matrix is global */

  case class RelationModelDecodeInfo(WShape: (Int, Int, Int), VShape: (Int, Int), bShape: Int, UShape: Int)

  case class DataBatch(data: Seq[Relation], corruptEntities: Seq[Int])

  val data = sc.parallelize(dataLoader.data)

  final val RANDOM_EMBEDDING_INIT_RANGE = 1e-4
  final val RANDOM_TENSOR_INIT_RANGE = 1.0 / math.sqrt(2 * conf.embeddingSize)
  final val numWords = dataLoader.entityToWordIndices.size
  final val numEntities = dataLoader.entityDict.size
  final val numRelations = dataLoader.relationsDict.size

  /* Randomly initialize the word embedding vectors */
  var embeddingMatrix: DenseMatrix[Double] = {
    val randomEmbedDistribution = Uniform(-RANDOM_EMBEDDING_INIT_RANGE, RANDOM_EMBEDDING_INIT_RANGE)
    DenseMatrix.rand(numWords, conf.embeddingSize, randomEmbedDistribution)
  }

  val initModels: Seq[RelationModel] = (1 to numRelations).map(relationIndex => {
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
  })

  /**
    * Roll all model parameters to be trained into a flattened param vector, theta
    */
  def rollToTheta(models: Seq[RelationModel]) : (DenseVector[Double], Seq[RelationModelDecodeInfo]) = {
    // flatten the tensor: horizontally concat all tensor slices, then flatten them all into one vector
    val rolledPerRelation = models.map(model => {
      val rolledW: DenseVector[Double] = model.W.map(_.toDenseVector).reduceLeft(DenseVector.vertcat(_, _))
      val rolledV = model.V.toDenseVector

      // supply parameter dimensions for reconstructing the model from theta
      val decodeInfo = RelationModelDecodeInfo((model.W.length, model.W.head.rows, model.W.head.cols),
        (model.V.rows, model.V.cols), model.b.length, model.U.length)

      (DenseVector.vertcat(rolledW, rolledV, model.b, model.U), decodeInfo)
    })

    val theta = rolledPerRelation.map(_._1).reduceLeft(DenseVector.vertcat(_, _))
    val decoders = rolledPerRelation.map(_._2)
    (theta, decoders)
  }

  /**
    * Unroll theta into the model for cost calculations
    */
  def unrollFromTheta(theta: DenseVector[Double], decoders: Seq[RelationModelDecodeInfo]) : Seq[RelationModel] = {
    // get ranges [x, y) to initially partition the rolled parameter vector
    var i = 0
    decoders.map(decodeInfo => {
      val WRange = i until (i + decodeInfo.WShape._1 * decodeInfo.WShape._2 * decodeInfo.WShape._3)
      val VRange = WRange.end until (WRange.end + decodeInfo.VShape._1 * decodeInfo.VShape._2)
      val bRange = VRange.end until (VRange.end + decodeInfo.bShape)
      val URange = bRange.end until (bRange.end + decodeInfo.UShape)
      i = URange.end

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
    })

  }

  /**
    * Apply nonlinearity function elementwise to vector
    */
  def activation(vector: DenseMatrix[Double], funcType: ActivationEnum = TANH) : DenseMatrix[Double] = {
    if (funcType == TANH) {
      tanh(vector)
    } else {
      sigmoid(vector)
    }
  }

  /**
    * Apply differential of the nonlinear activation function to the vector
    */
  def activationDifferential(vector: DenseMatrix[Double], funcType: ActivationEnum = TANH) : DenseMatrix[Double] = {
    if (funcType == TANH) {
      1.0 - (vector :* vector)
    } else {
      vector :* (1.0 - vector)
    }
  }

  def generateBatch(batchSize: Int, numCorruptions: Int) : DataBatch = {
    val batchData = data.takeSample(true, batchSize)
    val corruptEntities = Seq.fill[Int](numCorruptions)(Random.nextInt(numEntities))
    DataBatch(batchData, corruptEntities)
  }

  private def tensorProduct(left: DenseMatrix[Double], middle: DenseMatrix[Double], right: DenseMatrix[Double]) : DenseVector[Double] = {
    val prod: DenseMatrix[Double] = left * (middle dot right)
    val columns = (0 until prod.cols).map(i => {
      prod(::, i)
    })
    columns.foldLeft(DenseVector.zeros[Double](prod.rows))(_ + _)
  }

  /**
    * Tensor network cost function. Return scalar cost and the gradient for the model.
    */
  def cost(theta: DenseVector[Double], decoders: Seq[RelationModelDecodeInfo], dataBatch: DataBatch, flip: Boolean) : (Double, DenseVector[Double]) = {
    val models = unrollFromTheta(theta, decoders)

    //val entityVectors = DenseMatrix.zeros[Double](conf.embeddingSize, numEntities)
    val entityGrad = DenseMatrix.zeros[Double](conf.embeddingSize, numEntities)
    val entityData = (0 until numEntities).map(i => {
      val wordIndices = dataLoader.entityToWordIndices.get(i)
      assert(wordIndices.isDefined, "expected word indices sequences for entity index")
      val embeddings = embeddingMatrix(::, wordIndices.get).toDenseMatrix
      mean(embeddings(*, ::))
    }).reduceLeft(DenseVector.vertcat(_, _)).toArray

    val entityMatrix = new DenseMatrix[Double](conf.embeddingSize, numEntities, entityData)

    var cost = 0.0

    for (i <- 0 until numRelations) {
      // get all instances of this type of relation
      val relations = dataBatch.data.filter(_.relation == i)

      // get the indices of the entities to left and right of each instance of this relation
      var leftEntities = relations.map(_.leftEntity)
      var rightEntities = relations.map(_.rightEntity)

      // get all entity embedding vectors
      val leftEntityMatrix = entityMatrix(::, leftEntities).toDenseMatrix
      val rightEntityMatrix = entityMatrix(::, rightEntities).toDenseMatrix
      val corruptEntityMatrix = entityMatrix(::, dataBatch.corruptEntities).toDenseMatrix

      // get entity embedding vectors for negative sampling
      var leftEntityMatrixNeg = leftEntityMatrix
      var rightEntityMatrixNeg = corruptEntityMatrix
      var leftEntitiesNeg = leftEntitiesNeg
      var rightEntitiesNeg = dataBatch.corruptEntities

      // randomly swap where corrupt entities placed (left or right)
      if (!flip) {
        leftEntityMatrixNeg = corruptEntityMatrix
        rightEntityMatrixNeg = rightEntityMatrix
        leftEntitiesNeg = dataBatch.corruptEntities
        rightEntitiesNeg = rightEntities
      }

      // initialize pos/neg sampling preactivations
      // each row is the preactivation vector for a slice 1..k of the tensor W
      val preactivationPos: DenseMatrix[Double] = DenseMatrix.zeros[Double](conf.sliceSize, relations.length)
      val preactivationNeg: DenseMatrix[Double] = DenseMatrix.zeros[Double](conf.sliceSize, relations.length)

      val model = models(i)

      // add W tensor contribution to activation
      for (j <- 0 to conf.sliceSize) {
        preactivationPos(j, ::) := tensorProduct(leftEntityMatrix, model.W(j), rightEntityMatrix).t
        preactivationNeg(j, ::) := tensorProduct(leftEntityMatrixNeg, model.W(j), rightEntityMatrixNeg).t
      }

      // add standard feedforward model + bias to activation
      preactivationPos += model.b.t + (model.V.t dot DenseMatrix.horzcat(leftEntityMatrix, rightEntityMatrix))
      preactivationNeg += model.b.t + (model.V.t dot DenseMatrix.horzcat(leftEntityMatrixNeg, rightEntityMatrixNeg))

      // apply the activation function to all preactivation vectors
      val activationPos = activation(preactivationPos.t, conf.activationFunction)
      val activationNeg = activation(preactivationNeg.t, conf.activationFunction)

      // get scores where each entry is the score g(model) for a particular slice of the tensor W
      val scorePos: DenseVector[Double] = model.U.t dot activationPos
      val scoreNeg: DenseVector[Double] = model.U.t dot activationNeg

      // sum the costs of the entries where the positive score + 1 is greater than the negative score
      val filter = (scorePos :+ 1.0 :> scoreNeg).map(if (_) 1.0 else 0.0)
      cost += filter * (scorePos - scoreNeg :+ 1.0)

      // initialize gradients for the W tensor and V feedforward layer weights
      var WGrad = model.W.indices.map (i => DenseMatrix.zeros[Double](model.W.head.rows, model.W.head.cols))
      var VGrad = DenseMatrix.zeros[Double](model.V.rows, model.V.cols)

      // use total number of activations wrong to calculate gradient of model params
      val numWrong = filter.reduceLeft(_ + _)

      

    }


    (1., DenseVector.zeros(5))
  }
}

object DNTN extends Logging {
  val appName = "DNTN"

  def run(sc: SparkContext, conf: DataConfig) {
    logInfo("Loading training data")
    val trainingDataLoader = new DataLoader(conf.entitiesLocation, conf.relationsLocation, conf.trainLocation, conf.initEmebsLocation)
    val dntn = new DNTN(sc, trainingDataLoader, conf)

    logInfo("Theta sample!")
    val (theta, _) = dntn.rollToTheta(dntn.initModels)
    logInfo("Length: " + theta.length)
    logInfo(theta.toString)
  }

  case class DataConfig(
    trainLocation: String = "data/Wordnet/train.txt", // train data file
    testLocation: String = "data/Wordnet/test.txt",  // test data file
    entitiesLocation: String = "data/Wordnet/entities.txt", // entities dictionary file (entity -> ID)
    relationsLocation: String=  "data/Wordnet/relations.txt", // relations dictionary file (relation -> ID)
    initEmebsLocation: String = "data/Wordnet/initEmbed.mat",
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
