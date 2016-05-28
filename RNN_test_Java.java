package org.deeplearning4j.examples.feedforward.classification;

import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.commons.io.FileUtils;
import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.records.reader.impl.CSVSequenceRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator.AlignmentMode;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class RNN_StockPrices {

	public static void main(String[] args) throws IOException, InterruptedException {
	//Member variables declaration
		int nEpochs = 50;
		int vectorSize = 5;
		
	//Load the training data:
		SequenceRecordReader featureReader = new CSVSequenceRecordReader(0, ",");
		SequenceRecordReader labelReader = new CSVSequenceRecordReader(0,",");
		
		featureReader.initialize(new NumberedFileInputSplit("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/myInput_%d.csv", 0,250));
		labelReader.initialize(new NumberedFileInputSplit("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/myLabels_%d.csv", 0,250));
		
		SequenceRecordReader ValfeatureReader = new CSVSequenceRecordReader(0, ",");
		SequenceRecordReader VallabelReader = new CSVSequenceRecordReader(0,",");
		
		ValfeatureReader.initialize(new NumberedFileInputSplit("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/myInput_%d.csv",250,300));
		VallabelReader.initialize(new NumberedFileInputSplit("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/myLabels_%d.csv",250,300));
		
		DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(featureReader,labelReader,1,2,false,AlignmentMode.ALIGN_END);
		DataSetIterator ValIter = new SequenceRecordReaderDataSetIterator(ValfeatureReader,VallabelReader,50,2,false,AlignmentMode.ALIGN_END);
			
	//Set up network configuration
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    							.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
    							.updater(Updater.RMSPROP)
    							.regularization(true).l2(1e-5)
    							.weightInit(WeightInit.XAVIER)
    							.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
    							.learningRate(0.0001)
    							.list(2)		
    							.layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(20)
    							.activation("softsign").build())
    							.layer(1, new RnnOutputLayer.Builder().activation("softmax")
    									.lossFunction(LossFunctions.LossFunction.MCXENT).nIn(20).nOut(2).build())
    							.pretrain(false).backprop(true).build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
		
  //training on Dataset , calculate score on training and validation error
    System.out.println("Starting training");
    for( int n=0; n<nEpochs; n++ ){
        model.fit(trainIter);
        trainIter.reset();
        double train_score = model.score();
        double val_score = model.score(ValIter.next(), false);
        ValIter.reset();
        System.out.println("Epoch " + n + " TrainScore" + train_score + "ValScore"+ val_score);
        
      //write network parameters and configurations
        try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/Model_Configs/coefficients" +n+ ".bin"))))
        {
          Nd4j.write(model.params(),dos);
        }
      //Write the network configuration:
      FileUtils.write(new File("/Users/ROBIN/Desktop/MS_and_I_Res/HQ/Code/RNN_Java/Model_Configs/Model_Configs/conf" +n+".json"), model.getLayerWiseConfigurations().toJson());
    }    
	}

}
