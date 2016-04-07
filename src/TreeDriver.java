import java.io.*;
import java.util.Arrays;
import java.util.Base64;

import weka.core.Instances;
import homework2.*;

public class TreeDriver {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    /**
     * Sets the class index as the last attribute.
     * @param fileName
     * @return Instances data
     * @throws IOException
     */
    public static Instances loadData(String fileName) throws IOException{
        BufferedReader datafile = readDataFile(fileName);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }


    public static void main(String[] args) throws Exception {

        // read data file

        Instances testData = loadData("cancer_test.txt");
        Instances trainData = loadData("cancer_train.txt");
        // todo : build tree
        DecisionTree ourTree = new DecisionTree(trainData);

        //build the tree
        ourTree.buildClassifier(trainData);

        double avgError = ourTree.calcAvgError(trainData);
        double avgTestError = ourTree.calcAvgError(testData);

        //build tree and prune it.
        ourTree = new DecisionTree(trainData);
        ourTree.setPruningMode(true);
        ourTree.buildClassifier(trainData);

        double avgTrainErrorPruning = ourTree.calcAvgError(trainData);
        double avgTestErrorPruning = ourTree.calcAvgError(testData);

//		Run your code:
//		Load the train and the test data.
//				Build a decision tree on the train data without pruning and calculate the training error and the testing error.
//		Build a decision tree with pruning (i.e., set m_pruningMode to true) and calculate the errors again.
//				Which one had higher training error? Which one had higher test error? why?

        writeToTXT(avgError,avgTestError,avgTrainErrorPruning,avgTestErrorPruning);

//		The average train error of the decision tree is <avg_error>
//		The average test error of the decision tree is <avg_error>
//		The average train error the decision tree with pruning is <avg_error>
//				The average test error the decision tree with pruning is <avg_error>

    }

    /**
     * aux. method to output the required txt data on screen and into a text file.
     */
    public static void writeToTXT(double avgError, double avgTestError, double avgTrainErrorPruning, double avgTestErrorPruning) {
        try{
            String avgErrorTXT = ("The average train error of the decision tree is ");
            String avgTestErrorTXT = ("The average test error of the decision tree is ");
            String avgTrainErrorPruningTXT = ("The average train error of the decision tree with pruning is ");
            String avgTestErrorPruningTXT = ("The average test error of the decision tree with pruning is ");

            File file = new File("hw2.txt");

            // if file doesnt exists, then create it
            if (!file.exists()) {
                file.createNewFile();
            }

            // start writing into text file
            FileWriter fw = new FileWriter(file.getAbsoluteFile());
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(avgErrorTXT + Double.toString(avgError));
            bw.newLine();
            bw.write(avgTestErrorTXT + Double.toString(avgTestError));
            bw.newLine();
            bw.write(avgTrainErrorPruningTXT + Double.toString(avgTrainErrorPruning));
            bw.newLine();
            bw.write(avgTestErrorPruningTXT + Double.toString(avgTestErrorPruning));
            bw.newLine();

            bw.close();

            //output onto txt file done
        }
        catch(IOException e) {
            e.printStackTrace();

        }
    }

}

