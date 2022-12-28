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

public class MammalClassifier {

    public static void main(String[] args) {
        PropertyConfigurator.configure("log4j.properties");
        MammalClassifier app = new MammalClassifier();
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
                .appName("Classify Mammal")
                .config("spark.ui.port", 5050)
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        Dataset<Row> df = spark.read().format("csv")
                .option("header", "true")
                .load("data/mammal.csv");

        // String features to Categorical
        StringIndexerModel labelIndexer = buildFeatureIndexer(df, "class_label", "indexedLabel");
        StringIndexerModel bodyTempIndexer = buildFeatureIndexer(df, "body_temperature", "indexedBodyTemp");
        StringIndexerModel skinCoverIndexer = buildFeatureIndexer(df, "skin_cover", "indexedSkinCover");
        StringIndexerModel givesBirthIndexer = buildFeatureIndexer(df, "gives_birth", "indexedGivesBirth");
        StringIndexerModel aquaticIndexer = buildFeatureIndexer(df, "aquatic_creature", "indexedAquatic");
        StringIndexerModel aerialIndexer = buildFeatureIndexer(df, "aerial_creature", "indexedAerial");
        StringIndexerModel legsIndexer = buildFeatureIndexer(df, "has_legs", "indexedLegs");
        StringIndexerModel hibernatesIndexer = buildFeatureIndexer(df, "hibernates", "indexedHibernates");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[] {"indexedBodyTemp", "indexedSkinCover", "indexedGivesBirth",
                "indexedAquatic", "indexedAerial", "indexedLegs", "indexedHibernates"})
                .setOutputCol("features");

        Dataset<Row>[] splits = df.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("features")
                .setMaxDepth(4);

        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labelsArray()[0]);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{
                        labelIndexer, bodyTempIndexer, skinCoverIndexer, givesBirthIndexer,
                        aquaticIndexer, aerialIndexer,legsIndexer, hibernatesIndexer,
                        assembler, dt, labelConverter,
                });
        PipelineModel model = pipeline.fit(trainingData);

        Dataset<Row> predictions = model.transform(testData);

        predictions.select("predictedLabel", "class_label", "features").show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 -accuracy));

        DecisionTreeClassificationModel treeModel = (DecisionTreeClassificationModel) (model.stages()[9]);
        System.out.println("Learned classification tree model:\n" + treeModel.toDebugString());
    }
}
