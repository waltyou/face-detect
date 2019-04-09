package com.walt.face.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;

/**
 * @author waltyou
 * @date 2018/12/18
 */
public class ResourceUtils {

    private static Logger logger = LoggerFactory.getLogger(ResourceUtils.class);

    public static String getResourceFilePath(String path) {
        InputStream inputStream = null;
        OutputStream outputStream = null;
        String[] arr = path.split("/");
        String filename = arr[arr.length - 1];
        String tempFilename = "/tmp/" + filename;
        if (new File(tempFilename).exists()){
            return tempFilename;
        }
        try {
            // read this file into InputStream
            inputStream = ResourceUtils.class.getResourceAsStream(path);
            if (inputStream == null) {
                System.out.println("empty streaming");
            }
            // write the inputStream to a FileOutputStream
            outputStream =
                    new FileOutputStream(tempFilename);
            int read;
            byte[] bytes = new byte[102400];
            while ((read = inputStream.read(bytes)) != -1) {
                outputStream.write(bytes, 0, read);
            }
            outputStream.flush();
            logger.info("Load XML file, Done!");
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (outputStream != null) {
                try {
                    outputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }

            }

        }
        return tempFilename;
    }
}
