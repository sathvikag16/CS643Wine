# CS643Wine-Quality-Prediction-
CS 643 – Cloud Computing
Module 07 Assignment 01: Programming Assignment 2

This project is implemented with Spark data frames Api and MLib libraries, 
with this Native spark implementation application is automatically 
parallelized and distributed natively.

Wine prediction application is developed using Spark and MLib. Running it 
on AWS EMR cluster automatically parallelize and distribute job execution. 
Hadoop Distributed Files system is used for locating dataset files and for 
storing trained models.

This assignment is to develop parallel machine learning (ML) applications in 
AWS platform. We are using Apache Spark to train an ML model in parallel 
on multiple EC2 instances. We are also using Spark’s MLib to develop and 
use ML model in cloud. Along with that creating a Docker container for the 
ML model to simplify model deployment.

IMPORTANT LINKS: 
GitHub :  https://github.com/sathvikag16/CS643Wine-Quality-Prediction-

DockerHub: https://hub.docker.com/repository/docker/sathvikag16/wineprediction

STEPS ON HOW TO SET THE ENVIRONMENT FOR MODEL CREATION AND TRAINING USING AWS EMR:
1.	Login into AWS Account.

2.	Search for EMR in service > Open EMR.

3.	Click on the Create Cluster Button.

4.	Add Cluster Name. Here as we are using Spark we will Select Spark in Software Configuration. We need to run our program in 4 parallel EC2 instances so I created 5 instances 1 master and 4 slave. Select the keypair from the existing or you can also create one at that time and store at your local machine.

5. Click the button below to create the cluster.  

6. After that your cluster will be in the Starting Stage and you will see the screen as below.

7.	Click ssh to get login details for Master node of our EMR cluster > Save it.

8.	In the security and access summary click on the security group for master to change inbound rules for traffic as below.

After successful creation of AWS EMR Spark cluster, we can login into the cluster using terminal with the following Command. Command : ssh -i key.pem hadoop@user

Copy your Model file to the EWR cluster and run the file using the spark-submit command. Command : spark-submit ./sg2374-training.py

Output for this program is stored in my S3 bucket after the model is created we can run the Prediction program which gives us the Accuracy of the model same way using the Spark-Submit command.

STEPS TO RUN THE MODEL PREDICTION ON SINGLE EC2 INSTANCE WITHOUT DOCKER
1.	Create an EC2 instance.
2. Ssh into your ec2 instance.
3. Configure Spark in to your instance.
4. Pull the GitHub repository in your instance.
5. Create a Directory named testingdata by running the below command and put your test data set into that directory. Command : /mkdir testingdata
6. Now for the model prediction run the below command. Command : spark-submit --packages org.apache.hadoop:hadoop-aws:2.7.7 sg2374-testing.py

STEPS TO RUN THE MODEL PREDICTION ON SINGLE EC2 INSTANCE WITH DOCKER
1. Create an EC2 instance.
2. Ssh into your ec2 instance.
3. Install the docker into your EC2 instance if it is not there using the https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html
4. Add your test data file in your EC2 instance.
5. Run the Command : docker pull sathvikag16/wineprediction:latest
6. By running the above command the user will get the Docker image.
7. Now run the below command in the directory where your test data file is there.
8. Run the Command : sudo docker run -it -v "$(pwd)":/testingdata sathvikag16/wineprediction:latest
9. This will give the output of Model Accuracy and F1 score as shown below.
