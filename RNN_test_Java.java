package org.deeplearning4j.examples.feedforward.classification;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintWriter;

import org.apache.commons.io.FileUtils;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.CSVSequenceRecordReader;
import org.canova.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator.AlignmentMode;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class RNN_StockPrices_Test {
    public static void main(String[] args) throws IOException, InterruptedException {
        
        //load test data
        SequenceRecordReader featureReader = new CSVSequenceRecordReader(0, ",");
        SequenceRecordReader labelReader = new CSVSequenceRecordReader(0,",");
        featureReader.initialize(new NumberedFileInputSplit("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/myInput_%d.csv", 300,398));
        labelReader.initialize(new NumberedFileInputSplit("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/myLabels_%d.csv", 300,398));
        DataSetIterator testIter = new SequenceRecordReaderDataSetIterator(featureReader,labelReader,1,2,false,AlignmentMode.ALIGN_END);
        
        //load model configurations and parameters
        INDArray newParams;
    try (DataInputStream dis = new DataInputStream(new FileInputStream("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/coefficients13.bin"))) {
        newParams = Nd4j.read(dis);
    }
    MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/conf13.json")));
    MultiLayerNetwork model = new MultiLayerNetwork(confFromJson);
    model.init();
    model.setParameters(newParams);
        
//      //testing and writing the output on a text file
//      PrintWriter out = new PrintWriter("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/Prediction_RNN.txt");
//    int count = 0;
//    //while (testIter.hasNext()) {
//    while (count < 5) {
//        DataSet ds_predict = testIter.next();
//        INDArray features = ds_predict.getFeatureMatrix();
//        INDArray inMask = ds_predict.getFeaturesMaskArray();
//        INDArray outMask = ds_predict.getLabelsMaskArray();
//        INDArray predicted = model.output(features, false,inMask,outMask);
//        int[] prediction_value = model.predict(features);
//        out.println(predicted + "," + prediction_value[0]);
//        System.out.println("count" + count);
//        System.out.println(predicted.getRow(0));
//        System.out.println(predicted.getRow(1));
//        System.out.println(prediction_value[0]);
//        count = count + 1;
//        System.out.println(prediction_value[0])
//    }
//    out.close();
    DataSetIterator testIter2 = new SequenceRecordReaderDataSetIterator(featureReader,labelReader,1,2,false,AlignmentMode.ALIGN_END);
    Evaluation evaluation = new Evaluation();
    while (testIter2.hasNext()){
        DataSet ds_eval = testIter2.next();
        INDArray features = ds_eval.getFeatureMatrix();
        INDArray lables = ds_eval.getLabels();
        INDArray inMask = ds_eval.getFeaturesMaskArray();
        INDArray outMask = ds_eval.getLabelsMaskArray();
        INDArray predicted = model.output(features,false,inMask,outMask);
      evaluation.evalTimeSeries(lables,predicted,outMask);
    }
    
    
    System.out.println(evaluation.stats());
    }
}
