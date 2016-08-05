package pipelines

import loader.DataLoader
import org.apache.spark.SparkContext
import org.scalatest.FunSuite
import pipelines.DNTN.DataConfig

/**
  * Created by noah on 8/3/16.
  */
class DNTNSuite extends FunSuite {

  test("Model parameters roll and unroll correctly") {
    val sc = new SparkContext("local[6]", "DNTNSuite")
    val conf = new DataConfig()
    val trainingDataLoader = new DataLoader(conf.entitiesLocation, conf.relationsLocation, conf.trainLocation)
    val dntn = new DNTN(sc, trainingDataLoader, conf)

    val model = dntn.initModels.first()

    val (theta, shape) = dntn.rollToTheta(model)

    val unrolledModel = dntn.unrollFromTheta(theta, shape)

    assert(model.V == unrolledModel.V, "feedforward weight matrices V should be equal")
    assert(model.U == unrolledModel.U, "activation vectors U should be equal")
    assert(model.b == unrolledModel.b, "bias vectors should be equal")
    assert(model.W.zip(unrolledModel.W).forall(p => p._1 == p._2), "tensor slices should be equal")
  }

}
