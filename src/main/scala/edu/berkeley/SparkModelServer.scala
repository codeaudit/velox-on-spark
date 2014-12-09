package edu.berkeley

import java.nio.ByteBuffer
import scala.util._
import java.io.IOException
import java.nio.ByteBuffer
import scala.collection.immutable.HashMap
import java.nio.ByteBuffer
import java.io.ByteArrayOutputStream
import scala.collection.immutable.TreeMap
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext._
import breeze.linalg._

object ModelServingSpark {

  def main(args: Array[String]) {

    val sparkMaster = args(0)
    val numUsers = args(1).toInt
    val numItems = args(2).toInt
    val modelDim = args(3).toInt
    val percentOfItems = args(4).toDouble
    val sparkHome = "/root/spark"
    println("Starting spark context")

    val sc = new SparkContext(sparkMaster, "SparkTestApp", sparkHome,
        SparkContext.jarOfObject(this).toSeq)
    val modelServer = new ModelServer[Long](sc, numUsers, numItems, modelDim, percentOfItems)



    System.exit(0)

  }


  // implement updates with and without partial sums


}

class ModelServer(sc: SparkContext, numUsers: Int, numItems: Int, modelDim: Int, percentOfItems: Double) {

  val maxScore = 10

  var trainingDataRDD: RDD[(Long, Map[Long, Score])] = createTrainingDataRDD

  var modelsRDD: RDD[(Long, Array[Double])] = createModelsRDD

  // potential optimization: Make features a map, not RDD and make a broadcast variable
  var featuresRDD: RDD[(Long, Array[Double])] = createFeaturesRDD

  var partialSumsRDD: RDD[(Long, (DenseMatrix[Double], DenseVector[Double]))] = initializePartialSums(sc)


  // convenience RDDs for initalizing datasets
  val usersRDD = sc.parallelize((0 until numUsers)).cache()
  val itemsRDD = sc.parallelize((0 until numItems)).cache()

  def createModelsRDD: RDD[(Long, Array[Double])] = {
    usersRDD.map(id => {
      val rand = new Random
      (id, randomArray(rand, modelDim))
    })
  }

  def createTrainingDataRDD: RDD[(Long, Map[Long, Score])] = {
    usersRDD.map(id => {
      val rand = new Random
      val userObsMap = (0 until (numItems*percentOfItems/100.0).toInt)
        .map(x => (x.toLong, rand.nextDouble * maxScore)).toMap
      (id, userObsMap)
    })

  }

  def createFeaturesRDD: RDD[(Long, Array[Double])] = {
    itemsRDD.map(id => {
      val rand = new Random
      (id, randomArray(rand, modelDim))
    })
  }

  def initializePartialSums: RDD[(Long, (DenseMatrix[Double], DenseVector[Double]))] = {
    usersRDD.map(id => {
      val fakeSums = (DenseMatrix.rand[Double](numFeatures, numFeatures),
        DenseVector.rand[Double](numFeatures))
      (id, fakeSums)
    })
  }

  private def randomArray(rand: Random, size: Int) : Array[Double] = {
    val arr = new Array[Double](size)
    var i = 0
    while (i < size) {
      arr(i) = rand.nextGaussian
      i += 1
    }
    arr
  }

  /**
   * Update using partial sum
   */
  def updateModel(uid: Long, item: Long, score: Double) {
    val partialResults = lookup(uid, partialSumsRDD)
    val partialFeaturesSum = partialFeaturesSum._1
    val partialScoresSum = partialFeaturesSum._2
    val newScore = Map(item -> score)
    val itemFeatures = lookup(item, features)

    val (newWeights, newPartialResult) = UpdateMethods.updateWithBreeze(
      partialFeaturesSum, partialScoresSum, itemFeatures, newScore, modelDim)
    partialSumsRDD = updateRDD(uid, newPartialResult, partialSumsRDD)
    modelsRDD = updateRDD(uid, newWeights, modelsRDD)
    trainingDataRDD = updateRDD(uid, item, score, trainingDataRDD)
  }

  def lookup[T](id: Long, rdd: RDD[(Long, T)]): T = {
    (rdd.filter(_._1 == id).collect())(0)._2
  }

  def updateRDD[T](uid: Long, value: T, rdd: RDD[(Long, T)]): RDD[(Long, T)] = {

    rdd.map({ case(key, oldVal) => {
      if (key == uid) {
        (key, value)
      } else {
        (key, oldVal)
      }
    })
  }

  def appendScore[T](uid: Long, mapKey: T, score: Double, rdd: RDD[(Long, Map[T, Double])])
      : RDD[(Long, Map[T, Double])] = {

    rdd.map({ case(key, map) => {
      if (key == uid) {
        (key, map + (mapkey -> score))
      } else {
        (key, map)
      }
    })
  }


  def makePrediction(user: Long,
                     item: Long,
                     models: RDD[(Long, Array[Double])],
                     features: RDD[(Long, Array[Double])]): Double = {

    val userModel = lookup(models, user)
    val itemFeatures = lookup(features, item)
    var i = 0
    var prediction = 0.0
    while (i < user.size) {
      prediction += userModel(i)*itemFeatures(i)
      i += 1
    }
    prediction
  }

}


object UpdateMethods {

  val lambda = 1.0


  /**
   * Use breeze for matrix ops, does no caching or anything smart.
   */
  def updateWithBreeze[T](
      partialFeaturesSum: DenseMatrix[Double],
      partialScoresSum: DenseVector[Double],
      newItems: Map[T, FeatureVector],
      newScores: Map[T, Double],
      k: Int): (WeightVector, (DenseMatrix[Double], DenseVector[Double])) = {



    var i = 0

    val observedItems = newItems.keys.toList

    while (i < observedItems.size) {
      val currentItem = observedItems(i)
      // TODO error handling
      val currentFeaturesArray = newItems.get(currentItem) match {
        case Some(f) => f
        case None => throw new Exception(
          s"Missing features in online update -- item: $currentItem")
      }
      // column vector
      val currentFeatures = new DenseVector(currentFeaturesArray)
      val product = currentFeatures * currentFeatures.t
      partialFeaturesSum += product

      val obsScore = newScores.get(currentItem) match {
        case Some(o) => o
        case None => throw new Exception(
          s"Missing rating in online update -- item: $currentItem")
      }

      partialScoresSum += currentFeatures * obsScore
      i += 1
    }


    val partialResult = (partialFeaturesSum.copy, partialScoresSum.copy)

    val regularization = DenseMatrix.eye[Double](k) * (lambda*k)
    partialFeaturesSum += regularization
    val newUserWeights = partialFeaturesSum \ partialScoresSum
    (newUserWeights.toArray, partialResult)

  }

}
