# DNTN

A distributed neural tensor network on top of KeystoneML. Based on [Reasoning With Neural Tensor Networks For Knowledge Base Completion](http://www.socher.org/index.php/Main/ReasoningWithNeuralTensorNetworksForKnowledgeBaseCompletion) by Socher, Chen, Manning and Ng.

## To Run This Example:
Launch a KeystoneML cluster using the provided scripts according to [these instructions](http://keystone-ml.org/running_pipelines.html).

Once the cluster is up, ssh onto the master node and execute these commands:

```bash
# Build DNTN
cd /mnt
git clone https://github.com/noahgolmant/DNTN.git
cd DNTN
sbt/sbt assembly
~/spark-ec2/copy-dir target/

# Get the data
wget http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
tar -xvzf 20news-bydate.tar.gz

# Copy to HDFS
/root/ephemeral-hdfs/bin/hadoop fs -copyFromLocal 20news-bydate-train/ /data/
/root/ephemeral-hdfs/bin/hadoop fs -copyFromLocal 20news-bydate-test/ /data/

# Run the pipeline
export SPARK_HOME=/root/spark
export KEYSTONE_HOME=/root/keystone
KEYSTONE_MEM=4g ./bin/run-pipeline.sh \
  pipelines.DNTN \
  --trainLocation /data/20news-bydate-train \
  --testLocation /data/20news-bydate-test
```  
