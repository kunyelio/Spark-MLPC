package org.apache.spark.examples.mllib;

import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.Row;

import scala.Tuple2;

public class MultilayerPerceptronClassifierDemo {

	public static void main(String[] args) {
		// Set application name
		String appName = "MultilayerPerceptronClassifier";
		
		// Initialize Spark configuration & context
		SparkConf conf = new SparkConf().setAppName(appName)
				.setMaster("local[1]").set("spark.executor.memory", "1g");
		SparkContext sc = new SparkContext(conf);
		SQLContext sqlContext = new SQLContext(sc);

		// Load training and test data file from Hadoop and parse.
		String path = "hdfs://localhost:9000/user/konur/ED2010_2011_2012_SVM.txt";
		JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, path)
				.toJavaRDD();

		// Obtain 10 sets of training and test data. 12345 is the seed used to randomly split data.
		Tuple2<RDD<LabeledPoint>,RDD<LabeledPoint>>[] myTuple = MLUtils.kFold(data.rdd(), 10, 12345, data.classTag());
		
		
		// Train/validate the algorithm once for each set.
		for(int i = 0; i < myTuple.length; i++){
			JavaRDD<LabeledPoint> trainingData = (new JavaRDD<LabeledPoint>(myTuple[i]._1,data.classTag())).cache();
			JavaRDD<LabeledPoint> testData = new JavaRDD<LabeledPoint>(myTuple[i]._2,data.classTag());
			kRun(trainingData,testData,sqlContext);
		}
		sc.stop();
	}
	
	private static final void displayConfusionMatrix(Row[] rows){
		// #times label 0 correctly predicted
		int correctlyPredicted0 = 0;
		
		// #times label 1 correctly predicted
		int correctlyPredicted1 = 0;
		
		// #times label 1 wrongly predicted as label 0
		int wronglyPredicted0 = 0;
		
		// #times label 0 wrongly predicted as label 1
		int wronglyPredicted1 = 0;
		
		for(int i=0; i < rows.length; i++){
			Row row = rows[i];
			double label = row.getDouble(1);
			double prediction = row.getDouble(2);
			
			if(label == 0.0){
				if(prediction == 0.0){
					correctlyPredicted0++;
				}else{
					wronglyPredicted1++;
				}
			}else{
				if(prediction == 1.0){
					correctlyPredicted1++;
				}else{
					wronglyPredicted0++;
				}
			}
		}
		
		float fcorrectlyPredicted0 = correctlyPredicted0 * 1.0f;
		float fcorrectlyPredicted1 = correctlyPredicted1 * 1.0f;
		float fwronglyPredicted0 = wronglyPredicted0 * 1.0f;
		float fwronglyPredicted1 = wronglyPredicted1 * 1.0f;
		
		System.out.println("************");
		System.out.println(correctlyPredicted0 + "      " + wronglyPredicted1);
		System.out.println(wronglyPredicted0 + "      " + correctlyPredicted1);
		
		System.out.println("Class 0 precision: " + ((fcorrectlyPredicted0 == 0.0f)?0.0:(fcorrectlyPredicted0 / (fcorrectlyPredicted0 + fwronglyPredicted0))));
		System.out.println("Class 0 recall: " + ((fcorrectlyPredicted0 == 0.0f)?0.0:(fcorrectlyPredicted0 / (fcorrectlyPredicted0 + fwronglyPredicted1))));
		
		System.out.println("Class 1 precision: " + ((fcorrectlyPredicted1 == 0.0f)?0.0:(fcorrectlyPredicted1 / (fcorrectlyPredicted1 + fwronglyPredicted1))));
		System.out.println("Class 1 recall: " + ((fcorrectlyPredicted1 == 0.0f)?0.0:(fcorrectlyPredicted1 / (fcorrectlyPredicted1 + fwronglyPredicted0))));
		System.out.println("************");	
	}

	private static final void kRun(JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData, SQLContext sqlContext){
		DataFrame train = sqlContext.createDataFrame(trainingData, LabeledPoint.class);
		DataFrame test = sqlContext.createDataFrame(testData, LabeledPoint.class);

		// Input consists of 8 features; two hidden layers consist of 28, 25 computational units respectively;
		// Output is binary
		int[] layers = new int[] {8,  28, 25, 2};
		
		// Define the trainer.
		MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier()
		  .setLayers(layers)
		  .setBlockSize(128)
		  .setSeed(1234L)
		  .setMaxIter(150);
		// Obtain the trained model
		MultilayerPerceptronClassificationModel model = trainer.fit(train);	
		
		// Apply test data to model and obtain the output
		DataFrame testResult = model.transform(test);
		// Display performance metrics for the output
		displayConfusionMatrix(testResult.collect());

	}
}
