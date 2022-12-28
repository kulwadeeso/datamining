package org.annk.datamining.spark.classification;

import org.apache.log4j.PropertyConfigurator;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class IrisClassifier {
    public static void main(String[] args) {
        PropertyConfigurator.configure("log4j.properties");
        IrisClassifier app = new IrisClassifier();
        app.start();
    }

    private StringIndexerModel buildFeatureIndexer(Dataset<Row> df, String inputCol, String outputCol) {
        return new StringIndexer()
                .setInputCol(inputCol)
                .setOutputCol(outputCol)
                .fit(df);
    }
    private void start() {
        SparkSession spark = SparkSession.builder()
                .master("local")
                .appName("Classify Iris")
                .config("spark.ui.port", 5050)
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        Dataset<Row> df = spark.read().format("csv")
                .option("header", "true")
                .load("data/iris.csv");
        df = df.withColumn("sepal_length", df.col("sepal_length").cast("float"));
        df = df.withColumn("sepal_width", df.col("sepal_width").cast("float"));
        df = df.withColumn("petal_length", df.col("petal_length").cast("float"));
        df = df.withColumn("petal_width", df.col("petal_width").cast("float"));
        df.printSchema();
        // String features to Categorical
        StringIndexerModel labelIndexer = buildFeatureIndexer(df, "class_label", "indexedLabel");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[] {"sepal_length", "sepal_width", "petal_length", "petal_width"})
                .setOutputCol("features");

        Dataset<Row>[] splits = df.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("features")
                .setMaxDepth(5);

        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labelsArray()[0]);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{
                        labelIndexer, assembler, dt, labelConverter,
                });
        PipelineModel model = pipeline.fit(trainingData);

        Dataset<Row> predictions = model.transform(testData);

        predictions.select("predictedLabel", "class_label", "features").show((int)predictions.count(), false);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        DecisionTreeClassificationModel treeModel = (DecisionTreeClassificationModel) (model.stages()[2]);
        System.out.println("Learned classification tree model:\n" + treeModel.toDebugString());
    }

}
