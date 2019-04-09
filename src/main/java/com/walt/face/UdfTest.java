package com.walt.face;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

/**
 * @author waltyou
 * @date 2018/12/11
 */
public class UdfTest {

    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("udf test");
        JavaSparkContext sc = new JavaSparkContext(conf);

        SQLContext sqlContext = new org.apache.spark.sql.SQLContext(sc);
        sqlContext.udf().register("strLen", (UDF1<String, Object>) String::length, DataTypes.IntegerType);
    }
}
