/**
  * Created by noah on 7/27/16.
  */
package object activation {
  object ActivationEnum extends Enumeration {
    type ActivationEnum = Value
    val TANH, SIGMOID = Value
  }
}
