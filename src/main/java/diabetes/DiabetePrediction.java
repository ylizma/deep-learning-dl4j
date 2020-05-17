package diabetes;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class DiabetePrediction {
    public static void main(String[] args) throws IOException, InterruptedException {
        double learningRate = 0.001;
        int numInputs = 8;
        int numHidden = 8;
        int numOutput = 1;
        int bashsize = 2;
        int classIndex = 8;
        int numEpoch = 300;
//Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
        UIServer uiServer = UIServer.getInstance();
        InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHidden)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(1, new OutputLayer
                        .Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR).
                        nIn(numHidden)
                        .nOut(numOutput)
                        .activation(Activation.SIGMOID)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new StatsListener(inMemoryStatsStorage));

        File traindataset = new ClassPathResource("datasets/diabetes/diabetes_train.csv").getFile();
        RecordReader recordReader = new CSVRecordReader(0, ',');
        recordReader.initialize(new FileSplit(traindataset));
        File testdataset = new ClassPathResource("datasets/diabetes/diabetes_test.csv").getFile();
        RecordReader recordReaderTest = new CSVRecordReader(0, ',');
        recordReaderTest.initialize(new FileSplit(testdataset));

        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, bashsize, classIndex, numOutput);
        DataSetIterator dataSetIteratorTest = new RecordReaderDataSetIterator(recordReaderTest, bashsize, classIndex, numOutput);

        System.out.println("------------------ data training ---------------");
        for (int i = 0; i < numEpoch; i++) {
            model.fit(dataSetIterator);
        }
        System.out.println(model.summary());
        System.out.println("--------- evaluate Model -----------------------");
        Evaluation evaluation = new Evaluation(numOutput);

        while (dataSetIteratorTest.hasNext()) {
            DataSet dataSets = dataSetIteratorTest.next();
            INDArray features = dataSets.getFeatures();
            INDArray labels = dataSets.getLabels();
            INDArray predicted = model.output(features);
            evaluation.eval(labels, predicted);
        }
        System.out.println(evaluation.stats());
        INDArray predictInput = Nd4j.create(new double[][]{
                {4,142,86,0,0,44,0.645,22}
        });
        INDArray predictedoutput = model.output(predictInput);
        int[] classes = predictedoutput.toIntVector();
        for (int i = 0; i < classes.length; i++) {
            System.out.println(classes[i]);
        }
    }
}
