/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.examples

import ai.djl.{ModelException}
import ai.djl.inference.Predictor
import ai.djl.modality.{Input, Output}
import ai.djl.ndarray.{NDList, NDManager}
import ai.djl.repository.zoo.{Criteria}
import ai.djl.translate.{TranslateException}
import org.slf4j.{Logger, LoggerFactory}
import java.io.{IOException}
import java.nio.file.{Files, Paths}



object MaskClassifications {

  private val logger = LoggerFactory.getLogger(classOf[MaskClassifications])

  @throws[IOException]
  @throws[ModelException]
  @throws[TranslateException]
  def main(args: Array[String]): Unit = {
    val criteria = Criteria.builder
      .setTypes(classOf[Input], classOf[Output])
      .optModelUrl(Urls.get("https://drive.google.com/file/d/1-PbG6BXa-lU4vvC3esd0mBoTGA3HilNH/view?usp=sharing"))
      .optEngine("Python")
      .build // Use Python engine for pre/post processing

    try {
      val python = criteria.loadModel
      val transformer = python.newPredictor
      try {
        Mask(transformer)
      } finally {
        if (python != null) python.close()
        if (transformer != null) transformer.close()
      }
    }
  }
  @throws[IOException]
  @throws[ModelException]
  @throws[TranslateException]
  private def Mask(transformer: Predictor[Input, Output]): Unit = {
    val bertCriteria = Criteria.builder
      .setTypes(classOf[NDList], classOf[NDList])
      .optModelPath(Paths.get("model/model.zip"))
      .optEngine("Tensorflow")
      .build

    try {
      val bert = bertCriteria.loadModel
      val predictor = bert.newPredictor
      try {
        val input = new Input("1,Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0")
        var output = transformer.predict(input)
        val manager = bert.getNDManager.newSubManager // use pytorch engine
        val ndList = output.getDataAsNDList(manager)
        val predictions = predictor.predict(ndList)
        var file = Paths.get("build/mask_classification")

        try {
          val os = Files.newOutputStream(file)
          try new NDList(ndList.head).encode(os)
          finally if (os != null) os.close()
        }

        file = Paths.get("build/mask_classification_output")
        try {
          val os = Files.newOutputStream(file)
          try predictions.encode(os)
          finally if (os != null) os.close()
        }

        val postProcessing = new Input("1,Q42,Douglas Adams,English writer and humorist,Male,United Kingdom,Artist,1952,2001.0,natural causes,49.0")
        postProcessing.add("data", predictions)
        postProcessing.add("input_ids", new NDList(ndList.head))
        postProcessing.addProperty("handler", "mask_classification_postprocessor")
        output = transformer.predict(postProcessing)
        val result = output.getData.getAsString
        logger.info(result)

      } finally {
        if (bert != null) bert.close()
        if (predictor != null) predictor.close()
      }
    }
  }

  case class MaskClassifications() {}
}

