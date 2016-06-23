package com.imac;

import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;


// $example off$

public class JavaTfIdfExample {
	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("JavaTfIdfExample");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		SQLContext sqlContext = new SQLContext(jsc);

		// $example on$
		JavaRDD<Row> jrdd = jsc.parallelize(Arrays.asList(
				RowFactory.create(0.0, "Hi I heard about Spark"),
				RowFactory.create(0.0, "I wish Java could use case classes"),
				RowFactory.create(1.0, "Logistic regression models are neat")));
		
		StructType schema = new StructType(new StructField[] {
				new StructField("label", DataTypes.DoubleType, false,
						Metadata.empty()),
				new StructField("sentence", DataTypes.StringType, false,
						Metadata.empty()) });
		
		DataFrame sentenceData = sqlContext.createDataFrame(jrdd, schema);
		
		Tokenizer tokenizer = new Tokenizer().setInputCol("sentence")
				.setOutputCol("words");
		DataFrame wordsData = tokenizer.transform(sentenceData);
		
		for(Row v :wordsData.collect()){
			System.out.println("ROW : "+ v);
		}
		
		int numFeatures = 10;
		
		HashingTF hashingTF = new HashingTF().setInputCol("words")
				.setOutputCol("rawFeatures").setNumFeatures(numFeatures);
		DataFrame featurizedData = hashingTF.transform(wordsData);
		
		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
		IDFModel idfModel = idf.fit(featurizedData);
		DataFrame rescaledData = idfModel.transform(featurizedData);
		
		for (Row r : rescaledData.select("features", "label").take(3)) {
			Vector features = r.getAs(0);
			Double label = r.getDouble(1);
//			System.out.println(features);
//			System.out.println(label);
		}
		// $example off$

		jsc.stop();
	}
}