package org.deeplearning4j.examples.feedforward.classification;

import java.io.DataInputStream;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileInputStream;

import org.apache.commons.io.FileUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 * "Linear" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class MLPClassifierLinear {


    public static void main(String[] args) throws Exception {
        int seed = 123;
        double learningRate = 0.1;
        int batchSize = 50;
        int nEpochs = 10;

        int numInputs = 37;
        int numOutputs = 2;
        int numHiddenNodes = 50;

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/SBI_lag_TI_rise_fall_train.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,37,2);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/SBI_lag_TI_rise_fall_eval.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,37,2);
        
////      //Load parameters from disk:
//        INDArray newParams;
//        try(DataInputStream dis = new DataInputStream(new FileInputStream("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/java_model/coefficients2.bin"))){
//            newParams = Nd4j.read(dis);
//        }

      //Load network configuration from disk:
//        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/java_model/conf.json")));
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.SGD)
                .list(2)  //number of hidden layers excluding only the input layer
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.NORMALIZED)
                        .activation("relu")
                        .build())
//                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
//                        .weightInit(WeightInit.NORMALIZED)
//                        .activation("relu")
//                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.NORMALIZED)
                        .activation("softmax").weightInit(WeightInit.NORMALIZED)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
//        model.setParameters(newParams);
//      model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener()));


        for ( int n = 0; n < nEpochs; n++) {
            model.fit( trainIter );
            double train_score = model.score();
            System.out.println("Epoch " + n + " complete"+"score"+model.score());
//            if (n%5==0){
//              model.fit(testIter);
//              System.out.println("Epoch " + n + "train_score"+ train_score+"val_score"+model.score());
//              }
            }
        
        
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);
            System.out.println(predicted);
            eval.eval(lables, predicted);

        }
        rrTest.reset();
        int nTestPoints = 904;
        DataSetIterator testIter_second = new RecordReaderDataSetIterator(rrTest,nTestPoints,37,2);
        DataSet ds = testIter_second.next();
        INDArray features_second = ds.getFeatureMatrix();
        INDArray predicted_second = model.output(features_second,false);
        System.out.println(predicted_second);
        
        

        //Print the evaluation statistics
        System.out.println(eval.stats());
        
//        //write the network parameters
//        //http://deeplearning4j.org/modelpersistence
//        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/java_model/coefficients2.bin")))){
//          Nd4j.write(model.params(),dos);
//        }
//      //Write the network configuration:
//        FileUtils.write(new File("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/java_model/conf.json"), model.getLayerWiseConfigurations().toJson());
//        
//        
    }
//
//        //------------------------------------------------------------------------------------
//        //Training is complete. Code that follows is for plotting the data & predictions only
//
//        //Plot the data:
//        double xMin = 0;
//        double xMax = 1.0;
//        double yMin = -0.2;
//        double yMax = 0.8;
//
//        //Let's evaluate the predictions at every point in the x/y input space
//        int nPointsPerAxis = 100;
//        double[][] evalPoints = new double[nPointsPerAxis*nPointsPerAxis][2];
//        int count = 0;
//        for( int i=0; i<nPointsPerAxis; i++ ){
//            for( int j=0; j<nPointsPerAxis; j++ ){
//                double x = i * (xMax-xMin)/(nPointsPerAxis-1) + xMin;
//                double y = j * (yMax-yMin)/(nPointsPerAxis-1) + yMin;
//
//                evalPoints[count][0] = x;
//                evalPoints[count][1] = y;
//
//                count++;
//            }
//        }
//
//        INDArray allXYPoints = Nd4j.create(evalPoints);
//        INDArray predictionsAtXYPoints = model.output(allXYPoints);
//
//        //Get all of the training data in a single array, and plot it:
//        rr.initialize(new FileSplit(new File("src/main/resources/classification/linear_data_train.csv")));
//        rr.reset();
//        int nTrainPoints = 1000;
//        trainIter = new RecordReaderDataSetIterator(rr,nTrainPoints,0,2);
//        DataSet ds = trainIter.next();
//        PlotUtil.plotTrainingData(ds.getFeatures(), ds.getLabels(), allXYPoints, predictionsAtXYPoints, nPointsPerAxis);
//
//
//        //Get test data, run the test data through the network to generate predictions, and plot those predictions:
//        rrTest.initialize(new FileSplit(new File("src/main/resources/classification/linear_data_eval.csv")));
//        rrTest.reset();
//        int nTestPoints = 500;
//        testIter = new RecordReaderDataSetIterator(rrTest,nTestPoints,0,2);
//        ds = testIter.next();
//        INDArray testPredicted = model.output(ds.getFeatures());
//        PlotUtil.plotTestData(ds.getFeatures(), ds.getLabels(), testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis);
//
//        System.out.println("****************Example finished********************");
//    }
}