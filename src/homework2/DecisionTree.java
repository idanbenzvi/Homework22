package homework2;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Enumeration;

import static weka.core.Utils.log2;


public class DecisionTree extends Classifier {

    public static double CHI_SQ_CONSTANT = 2.733;
    private boolean m_pruningMode = false;
    private Node m_rootNode;
    private int m_numOfOriginalAttributes;
    private double[][][] attrCounts;
    private Instances originalInstances;


    public DecisionTree(Instances trainingSet) {

        this.originalInstances = trainingSet;
        m_numOfOriginalAttributes = trainingSet.numAttributes()-1;
    }

    /**
     * Builds a decision tree from the training data as learned in class.
     * Note: BuildClassifier is separated from BuildTree in order to allow you to do extra preprocessing before calling
     * buildTree (plus, Weka requires this interface).
     * Also this is the only method we provide the signature for because the signature for this method is determined by WEKA
     *
     * @param arg0 - instances of the training data
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances arg0) throws Exception {
        buildTree(arg0);

        //prune tree post creation using chi sqaure significance test
        if(m_pruningMode)
            pruning();

        m_rootNode.print();
    }

    /**
     * Builds the decision tree on given data set using either a recursive or queue algorithm as learned in class.
     *
     * @param trainingData - training or a subset of training data if creating recursive solution
     * @return
     */
    public DecisionTree buildTree(Instances trainingData) {

        //define the attributelist as an array of integers, each one corresponding to another
        //attribute.
        ArrayList<Integer> attributeList = new ArrayList();

        // populate the list of attributes used later on for attribute
        // selection in the information gain splitting process
        for (int i = 0; i < trainingData.numAttributes() - 1; i++)
            attributeList.add(i);

        m_rootNode = growTree(trainingData, attributeList);

        return this;
    }

    private boolean checkUniformClass(Instances dataSet){

//        System.out.println("checking if dataset is uniform");

        boolean uniformClass = true;

        Enumeration<Instance> instances = dataSet.enumerateInstances();
        Instance firstInst = instances.nextElement();

        while(instances.hasMoreElements()){
            Instance curInst = (Instance) instances.nextElement();
            if((int)curInst.classValue()!=(int)firstInst.classValue()){
                uniformClass = false;
                break;
            }

        }

       return uniformClass;

    }

    private Double findMaxFreqClassValue(Instances dataSet) {

//        System.out.println("checking highest freq. of class value");
        //if only one attribute remains - we shall take the y that has is the majority of the Ys
        //find most common Y
        int[] classValueFreq = new int[dataSet.classAttribute().numValues()]; //in our cases 2 values

        // iterate all types of y values, and select the most common
        for (int i = 0; i < dataSet.numInstances() - 1; i++) {
            //add to frequency of given class value (assuming always there 2 values - not more
            classValueFreq[(int) dataSet.instance(i).classValue()]++;
        }

        //assign the maximal class value (the majority)
        double findMaxClassValue = 0;

        for (int z = 0; z < classValueFreq.length; z++) {
            if (findMaxClassValue < classValueFreq[z])
                findMaxClassValue = z;
        }

        return findMaxClassValue;
    }


    private Node growTree(Instances trainingData, ArrayList<Integer> attributeList) {
        //create node
        Node currNode = new Node(null,trainingData);

        //attach instances associated with this node as part of the node itself (used later on while pruning)
        currNode.setAssociatedInstances(trainingData);

        if (checkUniformClass(trainingData)) {
            currNode.setReturnValue(trainingData.instance(0).classValue()); // set the return value
            return currNode;
        }
        //stop condition #2 -
        //in case no more attributes remain - we need to remove the
        if(attributeList.size()==1)
        {
            //set the returrn value of the current node to the majority class value
            currNode.setReturnValue(findMaxFreqClassValue(trainingData));

            return currNode;
        }


        // DONT STOP !

        //neither stopping conditions have been met - commence splitting process by going over all attributes
        //reducing impurity as much as possible, as measured by the information gain (reducing entropy).

        //measure information gain, one attribute at a time (goal is to obtain the highest value possible, reducing
        //impurity)
        double[] informationGain = new double[m_numOfOriginalAttributes];
        int maxIGattIndex = 0;
        double curMaxIG = 0;

        // // TODO: 06/04/2016 - check this code - it is the weakeset link!!!
        //check which attribute produces the maximal information gain (as shown in class)
        //by calculating the measure for attribute as the splitting attribute.
        for (int i = 0; i < attributeList.size(); i++) {
//            System.out.println(Arrays.toString(informationGain));
//            System.out.println("checking attribute for max gain");
//            System.out.println(attributeList.get(i));
            informationGain[attributeList.get(i)] = calcInfoGain(trainingData, attributeList.get(i));

            if (informationGain[attributeList.get(i)] > curMaxIG) {
                curMaxIG = informationGain[attributeList.get(i)]; //todo: check this out !!!
                maxIGattIndex = attributeList.get(i);
//                System.out.println("the current max gain attribute is: " + maxIGattIndex +" "+attributeList.get(i));
            }
        }

        //the best information gain has been chosen - we can now create the
        // children nodes and recursively continue the process.
        // in order to avoid repeating classification using the attribute - we remove it from the list

        //split the instances in order to further grow the tree accordting to the attribute chosen (max IG att.)
        Instances[] splitData = splitAttInstances(trainingData, maxIGattIndex);

        for (int z = 0; z < splitData.length; z++) {
            //duplication is required as each time we will be using a different version of the attributlist (in each node)
            ArrayList<Integer> newAtt;
            newAtt = (ArrayList) attributeList.clone();

            newAtt.remove(Integer.valueOf(maxIGattIndex));

            //set attribute index of this node (by which we split the instances)
            currNode.attributeIndex = maxIGattIndex;
            currNode.attributeName = trainingData.instance(0).attribute(maxIGattIndex).name();

            //call the tree growing process recursively, removing the current feature from the possible attribute set.
            if(splitData[z].numInstances()!=0) {

                Node childNode = growTree(splitData[z], newAtt);

                //set child node attribute value (as will be required when classifying)
                //// TODO: 06/04/2016 check valid 
                childNode.setAttributeValue(splitData[z].instance(0).value(maxIGattIndex));

                //afer growing has been completed - attach each new node to its parent and set it as
                //the child for the current node.
                currNode.addChild(childNode);
                //declare the parent of the childnode just created as this node
                childNode.setParent(currNode);

            }
        }

        return currNode; // will return the current branch of the tree and from the 1st call will return the root.
    }

    /**
     * * Input: A subset of the training data, attribute index, extra variables you need.
     * * Description: The method calculates the information gain of splitting the input data according to the attribute.
     * * Output: The information gain (double).
     */
    public double calcInfoGain(Instances dataSet, int attributeIndex) {
        //calculate the gain acheived by choosing a specific attribute as the splitting attribute.

        // in order to calculate the information gain, we must evaluate each class value Y and its corresponding instances
        // for each attribute j. (as seen in class and in the recitation)
        // we will be calculating the information gain, based upon the entory we have measured for xj
        // I(A,B) = H(B) - sigma_b P(B=b)*H(A|B=b)

        // iterate over each attribute and asses the infromation gain obtained using the attribute as a splitting attribute

        double entropySum = 0;
        double curGain = 0;
        double calculatedEntropy;

        //split attribute to groups of instances according to the number of distinct possible values
        Instances[] splitter = splitAttInstances(dataSet, attributeIndex);

        //calculate the entropy of each attribute value and sum it
        for (int z = 0; z < splitter.length; z++) {
            calculatedEntropy = calcEntropy(splitter[z]);
            entropySum += ((double)splitter[z].numInstances() / (double) dataSet.numInstances()) * calculatedEntropy;
        }
            //compare to previous maximum
            curGain = calcEntropy(dataSet) - entropySum;

        return curGain;
    }

    /**
     * split instances according to a given attribute.
     * input: instances and attribute index
     * output: an array of instances - split according to the possible values of the attribute.
     *
     * @param instances
     * @param attIndex
     * @return
     */
    private Instances[] splitAttInstances(Instances instances, int attIndex) {

        Enumeration instEnum = instances.enumerateInstances();
        Instances[] attInstances = new Instances[instances.attribute(attIndex).numValues()];

        for (int j = 0; j < instances.attribute(attIndex).numValues(); j++) {
            attInstances[j] = new Instances(instances, instances.numInstances()); // init - this will be modified as we go
        }

        while (instEnum.hasMoreElements()) {
            Instance curInst = (Instance) instEnum.nextElement();
            attInstances[(int) curInst.value(attIndex)].add(curInst); // add the current instance to corresponding att. value
        }

        return attInstances;
    }

    /*
    Input: a set of probabilities
Description: Calculates the entropy of a random variable where all the probabilities of all of the possible values it can take are given as input.
Output: The entropy (double).
Note: using a log base e will work but you should use log base 2.
     */
    public double calcEntropy(Instances instances) {

        double[] classCounts = new double[instances.numClasses()];
        double entropy = 0;

        Enumeration instEnum = instances.enumerateInstances();

        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }

        for (int j = 0; j < instances.numClasses(); j++) {
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] / (double) instances.numInstances() * log2(classCounts[j] / (double) instances.numInstances());
            }
        }

        return entropy;
    }


    /**
        Description: Counts the total number of classification mistakes on the testing data set and divides that by the number of data instances.
        Output: Average error (double).
        Note: For example - if on a test set you classified 10 examples correctly, and 10 incorrectly, the output of this method should be 0.5.

     * @param instances
     * @return average error (double)
     */
    public double calcAvgError(Instances instances){

        Enumeration instEnum = instances.enumerateInstances();
        int[] classValues = new int[2];
        double classifyResult;
        int errors=0;
//        int counter= 0;

        //obtain number of positive and negative classifications (true values)
        while (instEnum.hasMoreElements()) {
            Instance curInst = (Instance) instEnum.nextElement();
            //compare with classification result
            classifyResult =classify(curInst);
            //compare the classifier result with true instance class value
            int classValue = (int) curInst.classValue();

//            counter++;
//            System.out.println(counter);
            if(classValue!=(int)classifyResult)
                errors++;
        }

        return (double) errors / (double) instances.numInstances();
    }

    /**
     * Description: Returns a classification for the instance.
     * output : the classification result - 0 / 1
     *
     * @param instance - an instance to classify according to our decision tree
     * @return
     */
    public double classify(Instance instance) {
        //Go down the decision tree until reaching a leaf ( a node with no children and a return value).

        Node currNode = m_rootNode;
        boolean noMatch = true;

        if(instance.toString().equals("40-49,ge40,40-44,15-17,yes,2,right,left_up,yes,no-recurrence-events"))
            System.out.println("wow");

        //repeat until reaching a leaf node (representing a a class value choice according to the tree's structure)
        while ((int)currNode.getReturnValue() == -1) {
                noMatch = true;
                //get the current node splitting attribute index
                int attIndex = currNode.attributeIndex;

                //get the child which corresponds to the attribute value of the given instance
                ArrayList<Node> children = currNode.getChildren();
                //find the child with the value
                for(Node i : children){
                    if(i.getAttributeValue()==instance.value(attIndex)) {
                        currNode = i;
                        noMatch = false;
                        break;
                    }
                }
                //if no match has been found - use the most common attribute value
                if(noMatch){
                     findMaxFreqClassValue(currNode.associatedInstances);
                    noMatch=true;
                    break;
                }

                //above procedure has failed - no match between attribute value and existing tree attribute values

        }

        //return the classification held by the leaf node
        return currNode.getReturnValue();
    }

    /**
     * Performs post pruning after the tree has been generated. The pruning process is done via selection of leaf nodes,
     * as representatives of an attribute class selection, evaluating their chi squared value and according to the
     * result - the branch is either kept, or cut off from the tree. In case the branch is cut off - all the child nodes
     * are cancelled and only the parent node is kept - letting the majority class value be the new return value.
     * After a node evaluated - the next node to be evaluated is its parent node. to avoid repetition - each node
     * that has been traversed in the tree is 'marked'.
     */
    public void pruning(){
        ArrayList<Node> allNodes = getAllNodes(m_rootNode); //obtain a list with all possible nodes. work only on non leaf nodes
        ArrayList<Node> leafsOnly = new ArrayList<Node>() ;

        //create a list of the leaf nodes
        for(Node i : allNodes){
            if(i.getReturnValue()!=-1)
            leafsOnly.add(i);
        }

        double chiSq= 0;
        double newValue = 0;
        ArrayList<Node> parentList = new ArrayList<Node>();

        // go over all child nodes
            for(Node i : leafsOnly) {
                Node currNode = i.getParent();
                while (currNode != null) {
                    //iterate until reaching a null parent
                    inspectPruning(currNode);

                    //next node to evaluate
                    currNode = currNode.getParent();
                }
            }



        //calc chi squared for each node in the resulting list

        //if non significant split - remove the branch under the node.
    }

    private void inspectPruning(Node node) {
    //check if node requires pruning - if it does - perform it. otherwise do nothing besides marking it as checked
        //look at the parent and see if it requires pruning
        Double chiSq = calcChiSquare(node.associatedInstances,node.attributeIndex);

        //if sig. do nothing - just mark as pruned
        if(chiSq < CHI_SQ_CONSTANT) {
            node.pruned = true;
        }
        else //prune
        {
            // remove node children, mark as pruned - children will have no reference from parent anymore (so sad...)
            node.prune();

            //Set most common class value as classifier
            node.setReturnValue(findMaxFreqClassValue(node.associatedInstances));
        }
    }

    private boolean awaitingInspection(ArrayList<Node> nodes){
        boolean stillRemain = false;

        for(Node i : nodes){
            if(i.pruned==false) {
                stillRemain = true;
                break;
            }
        }

        return stillRemain;
    }

    /**
     * Description: Calculates the chi square statistic of splitting the data according to this attribute as learned in class.
     * Output: The chi square score (double).
     * Note: This method is required only for the version of the algorithm with the pruning (section 6). When deciding
     * whether to prune or not to prune a branch you preform
     * the chi squared test and then you compare the number you get to the chi squared chart. The number you should
     * compare to for the cancer data is 2.733 . This number comes
     * from the chi squared chart in the row for 8 degrees of freedom (which is the number of attributes in the cancer
     * data minus 1) and the column for 0.95 confidence level. If you use the mushroom data you should use the number
     * 11.591 which is 0.95 confidence column and 21 dof.
     * Note on pruning: If you prune a branch and make a leaf node instead of it, what should that leaf node return?
     * It should return the majority class value of the data associated with that node.
     *
     * @param subsetInstances
     * @param attributeIndex
     * @return
     */
    public double calcChiSquare(Instances subsetInstances, int attributeIndex) {

        Instances[] instances = splitAttInstances(subsetInstances,attributeIndex);

        int py0=0; // instances count of y being 0
        int py1=0; // instances count of y being 1
        int pf =0; // instances of Df=f value, and y=1
        int nf =0 ; // instances of Df=f value, and y=0
        int E0=0 ;
        int E1=0 ;
        double chiSq =0;

        //count py0 and py1 for the E0 and E1 calculation, respectively
        for(int j = 0 ; j < subsetInstances.numInstances(); j++) {
            if (subsetInstances.instance(j).classValue() == 0)
                py0++;
            else
                py1++;
        }

        // perform the calculation according to recitation slide #28
        for (int f = 0 ; f < instances.length ; f++){

            //calculate nf and pf
            for(int z = 0 ; z < instances[f].numInstances() ; z++){
                if(instances[f].instance(z).classValue()==0 )
                    nf++;
                else
                    pf++;
            }

            //calculate E0 & E1
            E0 = instances[f].numInstances() * py0;
            E1 = instances[f].numInstances() * py1;

            //add to the chi sq. sum
            chiSq += (double) Math.pow((E0 - pf),2) / (double) pf + (double) Math.pow((E1 - nf),2) / (double) nf ;
        }


        // the chi square score as learned in class, returns the statistic and according to the chart and df - we select
        // the correct value for comparison (the expected values against the perceived values).
        return chiSq;
    }

    /**
     * returns all the leafs in the current tree, from the given node as root.
     * @param node
     */
    public ArrayList<Node> getAllNodes(Node node){
        ArrayList<Node> leafNodes = new ArrayList<Node>();

        leafNodes.add(node);
            for(Node i : node.getChildren()){
                if(i.getChildren().size()!=0)
                    leafNodes.addAll(getAllNodes(i));
        }

        return leafNodes;
    }

    public void setPruningMode(boolean mode){
        m_pruningMode = mode;
    }


/**
 * Internal class representing either an attribute by which the data is split, or when reaching a leaf node -
 * a classification according to a certain attribute value.
 */
private class Node {
    private ArrayList<Node> children;
    private Node parent;
    int attributeIndex;
    double attributeValue;
    String attributeName = "";
    private Double returnValue;
    Instances associatedInstances ;
    boolean pruned = false;


    /** Constructor - currently we only assign a parent and once the node is procssed by the algorithm we will be able
     * to set the rest of the parameters (attr, attrcountarray)
     *
     * @param parent
     */
    public Node(Node parent,Instances instances) {
        this.parent = parent;
        this.children = new ArrayList<Node>();
        this.returnValue = -1.0;
        this.associatedInstances = instances;
    }


    public void addChild(Node node) {
        this.children.add(node);
    }

    public void setParent(Node node) {
        this.parent = node;
    }

    public ArrayList<Node> getChildren() {
        return this.children;
    }

    public void setReturnValue(double value) {
        this.returnValue = value;
    }

    public Node getParent(){
        return this.parent;
    }

    public double getReturnValue() {
        return this.returnValue;
    }

    public void setAssociatedInstances(Instances instances){
        this.associatedInstances = instances;
    }

    public boolean isPruned() {
        return this.pruned;
    }

    public void setAttributeValue(double value){
        this.attributeValue = value;
    }

    public double getAttributeValue(){
        return this.attributeValue;
    }

    public void prune(){
        this.pruned=true;

        //nullify parent for the children (cut all ties from parent)
        for(Node i : this.children ){
            i.setParent(null);
        }

        this.children.clear(); //remove pointer reference to children nodes
    }

    @Override
    public String toString(){
        return "parent: "+ this.parent + " " + "num of children: " + this.children.size() + " return value : " + this.attributeName;
    }


    public void print() {
            print("", true);
        }

    private void print(String prefix, boolean isTail) {
            System.out.println(prefix + (isTail ? "└── " : "├── ") + this.attributeName +" "+ this.attributeIndex + " " + (int)this.attributeValue);
            for (int i = 0; i < children.size() - 1; i++) {
                children.get(i).print(prefix + (isTail ? "    " : "│   "), false);
            }
            if (children.size() > 0) {
                children.get(children.size() - 1).print(prefix + (isTail ?"    " : "│   "), true);
            }
        }


}
}
