package uk.soton.ecs.COMP3204.cw3.groupA;

import java.io.File;
import java.util.List;
import java.util.Scanner;
import java.io.FileWriter;
import java.io.IOException;
import java.io.BufferedWriter;
import java.nio.file.FileSystemException;

/**
 * @author team 14
 */
public class Main {
    final static String CURRENT_WORKING_DIRECTORY 		   = System.getProperty("user.dir");
    final static String TRAINING_PATH			           = CURRENT_WORKING_DIRECTORY+"/training";
    final static String TESTING_PATH  		 	   	   = CURRENT_WORKING_DIRECTORY+"/testing";
    final static Scanner scan 			   		   = new Scanner(System.in);
    private static Data data;

	/**
	 * The main routine to start 'Run1', 'Run2', 'Run3'
	 * @param args
	 * @throws IOException
	 */
    public static void main( String[] args ) throws IOException {
    	data = new Data(TRAINING_PATH, TESTING_PATH);
    	
    	System.out.println("==> START <== ");
    	System.out.println("Please Choose from [run1/run2/run3]:");
    	String result = scan.nextLine();
    	
    	if(!(result.equals("run1") || result.equals("run2") || result.equals("run3"))) {
    		System.err.println("[ERROR] There is no "+result+". [run1/run2/run3]");
    		System.exit(0);
    	}
    	
    	System.out.println("[*] Running "+result);
    	List<String> resultList = null;
    	
    	switch (result) {
    	  case "run1":  resultList = run1(data); break;
    	  case "run2":  resultList = run2(data); break;
    	  case "run3":	resultList = run3(data); break;
    	  default: System.err.println("[ERROR]"); System.exit(0);
    	}
    	
    	System.out.println("[*] Writing to file");
    	
    	switch(result) {
    		case "run1":  writeFile(data.getRun1File(), resultList); break;
    		case "run2":  writeFile(data.getRun2File(), resultList); break;
    		case "run3":  writeFile(data.getRun3File(), resultList); break;
    		default: System.err.println("[ERROR]"); System.exit(0);
    	}	
    }


	/**
	 * Run the classifier on the given dataset
	 * @param data the dataset
	 * @return
	 * @throws IOException
	 */
	public static List<String> run1(Data data) throws IOException {
    	KNNClassifier knn = new KNNClassifier(data);
    	return knn.run();
    }

	/**
	 * Run the classifier on the given dataset
	 * @param data the dataset
	 * @return
	 * @throws IOException
	 */
	public static List<String> run2(Data data) throws IOException {
    	KMeansClassifier kmc = new KMeansClassifier(data);
    	return kmc.run();
    }

	/**
	 * Run the classifier on the given dataset
	 * @param data the dataset
	 * @return
	 * @throws FileSystemException
	 * @throws IOException
	 */
	public static List<String> run3(Data data) throws FileSystemException, IOException{
    	BestClassifier bc = new BestClassifier(data);
		return bc.run();
    }

	/**
	 * Write a list of strings to a file
	 * @param resultFile the outputted file
	 * @param finalR the array of strings
	 * @throws IOException
	 */
	public static void writeFile(File resultFile, List<String> finalR) throws IOException {
    	File file = resultFile;
		FileWriter fr = new FileWriter(file, false);
		BufferedWriter br = new BufferedWriter(fr);
		
		String out;
		for(int j=0; j<finalR.size();j++){
			out = "";
			for(int p=0; p<finalR.size();p++){
				if(finalR.get(p).matches(j+".jpg(.*)")){
					out = finalR.get(p);
					break;
				}
			}

			if(out!="") {
				br.write(out);
				br.newLine();
			}
		}
		
		br.close();
		fr.close();
		
		System.out.println("\nCHECK: "+ resultFile.getPath()+"\n\n==> DONE <== ");
    }
}
