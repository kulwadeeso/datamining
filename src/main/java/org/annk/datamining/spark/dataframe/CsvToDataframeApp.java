package org.annk.datamining.spark.dataframe;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import org.apache.log4j.PropertyConfigurator;

public class CsvToDataframeApp {

    public static void main(String[] args) {
        PropertyConfigurator.configure("log4j.properties");
        CsvToDataframeApp app = new CsvToDataframeApp();
        app.start();
    }
    private void start() {
        SparkSession spark = SparkSession.builder()
                .master("local")
                .appName("CSV to Dataset")
                .config("spark.ui.port", 5050)
                .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");

        Dataset<Row> df = spark.read().format("csv")
                .option("header", "true")
                .load("data/books.csv");
        df.show((int) df.count(), false);
        df.printSchema();
    }
}
