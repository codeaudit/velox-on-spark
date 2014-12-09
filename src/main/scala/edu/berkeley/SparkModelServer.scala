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
    val numReqs = args(4).toInt
    val percentObs = args(5).toDouble
    val percentOfItems = args(6).toDouble
    
    val sparkHome = "/root/spark"
    println("Starting spark context")

    val sc = new SparkContext(sparkMaster, "SparkTestApp", sparkHome,
        SparkContext.jarOfObject(this).toSeq)
    val modelServer = new ModelServer(sc, numUsers, numItems, modelDim, percentOfItems)
    val requestor = new Requestor(numUsers, numItems, percentObs)

    val nanospersec = math.pow(10, 9)
    var numPreds = 0
    var numObs = 0


    val startTime = System.nanoTime
    var i = 0
    while (i < numReqs) {

      requestor.getRequest.fold(
        oreq => {
          modelServer.updateModel(oreq.uid, oreq.context, oreq.score)
          numObs += 1
          println(s"Making observation for user ${oreq.uid} and item ${oreq.context}")
        },
        preq => {
          val result = modelServer.makePrediction(preq.uid, preq.context)
          numPreds += 1
          println(s"Prediction result: $result")
        }
      )

      if (i % 10 == 0) {
        println(s"Finished $i/$numReqs queries")

      }

      i += 1
    }
    modelServer.countRDDs()

    val stopTime = System.nanoTime
    val elapsedTime = (stopTime - startTime) / nanospersec
    val pthruput = numPreds.toDouble / elapsedTime.toDouble
    val othruput = numObs.toDouble / elapsedTime.toDouble
    val totthruput = numReqs.toDouble / elapsedTime.toDouble

    val outstr = (s"duration: ${elapsedTime}\n" +
                  s"num_pred: ${numPreds}\n" +
                  s"num_obs: ${numObs}\n" +
                  s"pred_thru: ${pthruput}\n" +
                  s"obs_thru: ${othruput}\n" +
                  s"total_thru: ${totthruput}\n")

    println(outstr)


    System.exit(0)

  }


  // implement updates with and without partial sums


}

class ModelServer(sc: SparkContext, numUsers: Int, numItems: Int, modelDim: Int, percentOfItems: Double) {

  private val maxScore = 10

  private var trainingDataRDD: RDD[(Long, Map[Long, Double])] = createTrainingDataRDD

  private var modelsRDD: RDD[(Long, Array[Double])] = createModelsRDD

  // potential optimization: Make features a map, not RDD and make a broadcast variable
  private var featuresRDD: RDD[(Long, Array[Double])] = createFeaturesRDD

  private var partialSumsRDD: RDD[(Long, (DenseMatrix[Double], DenseVector[Double]))] = initializePartialSums


  // convenience RDDs for initalizing datasets
  private val usersRDD = sc.parallelize((0 until numUsers)).cache()
  private val itemsRDD = sc.parallelize((0 until numItems)).cache()

  private def createModelsRDD: RDD[(Long, Array[Double])] = {
    usersRDD.map(id => {
      val rand = new Random
      (id, randomArray(rand, modelDim))
    })
  }

  private def createTrainingDataRDD: RDD[(Long, Map[Long, Double])] = {
    usersRDD.map(id => {
      val rand = new Random
      val userObsMap = (0 until (numItems*percentOfItems/100.0).toInt)
        .map(x => (x.toLong, rand.nextDouble * maxScore)).toMap
      (id, userObsMap)
    })

  }

  private def createFeaturesRDD: RDD[(Long, Array[Double])] = {
    itemsRDD.map(id => {
      val rand = new Random
      (id, randomArray(rand, modelDim))
    })
  }

  private def initializePartialSums: RDD[(Long, (DenseMatrix[Double], DenseVector[Double]))] = {
    usersRDD.map(id => {
      val fakeSums = (DenseMatrix.rand[Double](modelDim, modelDim),
        DenseVector.rand[Double](modelDim))
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

  def countRDDs() {
    trainingDataRDD.count()
    modelsRDD.count()
    partialSumsRDD.count()
    featuresRDD.count()
  }

  def makePrediction(user: Long, item: Long): Double = {

    val userModel = lookup(user, modelsRDD)
    val itemFeatures = lookup(item, featuresRDD)
    var i = 0
    var prediction = 0.0
    while (i < userModel.size) {
      prediction += userModel(i)*itemFeatures(i)
      i += 1
    }
    prediction
  }

  /**
   * Update using partial sum
   */
  def updateModel(uid: Long, item: Long, score: Double) {
    val partialResults = lookup(uid, partialSumsRDD)
    val partialFeaturesSum = partialResults._1
    val partialScoresSum = partialResults._2
    val newScore = Map(item -> score)
    val itemFeatures = Map(item -> lookup(item, featuresRDD))


    val (newWeights, newPartialResult) = UpdateMethods.updateWithBreeze(
      partialFeaturesSum, partialScoresSum, itemFeatures, newScore, modelDim)
    partialSumsRDD = updateRDD(uid, newPartialResult, partialSumsRDD)
    modelsRDD = updateRDD(uid, newWeights, modelsRDD)
    trainingDataRDD = appendScore(uid, item, score, trainingDataRDD)
  }

  private def lookup[T](id: Long, rdd: RDD[(Long, T)]): T = {
    (rdd.filter(_._1 == id).collect())(0)._2
  }

  private def updateRDD[T](uid: Long, value: T, rdd: RDD[(Long, T)]): RDD[(Long, T)] = {

    rdd.map({ case(key, oldVal) => {
        if (key == uid) {
          (key, value)
        } else {
          (key, oldVal)
        }
      }
    })
  }

  private def appendScore[T](uid: Long, mapKey: T, score: Double, rdd: RDD[(Long, Map[T, Double])])
      : RDD[(Long, Map[T, Double])] = {

    rdd.map({ case(key, map) => {
        if (key == uid) {
          (key, map + (mapKey -> score))
        } else {
          (key, map)
        }
      }
    })
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
      newItems: Map[T, Array[Double]],
      newScores: Map[T, Double],
      k: Int): (Array[Double], (DenseMatrix[Double], DenseVector[Double])) = {



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
