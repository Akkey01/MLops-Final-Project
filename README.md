# MLops-Final-Project

## Intelligent Multimedia Processing (IMP) for Enterprises
Enterprises currently rely on manual searching through documents, audio, and video recordings, which is labor-intensive and inefficient. The IMP system automates the extraction and indexing of this multimedia data, allowing employees to directly query the information using natural language questions. Additionally, the system can automatically generate minutes of meetings from recorded meeting sessions, further increasing productivity and documentation accuracy. Key business metrics for evaluation include reduction in time spent searching for information, accuracy of retrieved answers, and quality of automatically generated meeting minutes.


### Contributors


| Name ( All team members)        | Responsible for                          | Link to their commits in this repo                                                   |
|---------------------------------|------------------------------------------|--------------------------------------------------------------------------------------|
| Akshat Mishra                   | data pipeline                            | https://github.com/Akkey01/MLops-Final-Project/commits/main/?author=Akkey01          |
| Siddhant Mohan                  | model training                           | https://github.com/Akkey01/MLops-Final-Project/commits/main/?author=siddhantmohan1110|
| Mihir Khare                     | model serving & monitoring               | https://github.com/Akkey01/MLops-Final-Project/commits/main/?author=Mihir-Khare429   |
| Nikita Gupta                    | continuous X pipeline                    | https://github.com/Akkey01/MLops-Final-Project/commits/main/?author=nairanikita      |



### System diagram
![Editor _ Mermaid Chart-2025-05-12-004857](https://github.com/user-attachments/assets/33c9cf1c-2de2-4d4a-95c0-8f38e863d673)


### Summary of outside materials


| DATESET      | Name                 |  Link to the DATASET                           |
|--------------|----------------------|------------------------------------------------|
| Data set 1   |os- AMI Meeting Corpus|https://groups.inf.ed.ac.uk/ami/download/       |
| Data set 2   |os-ICSI Meeting Corpus|https://groups.inf.ed.ac.uk/ami/icsi/download/  |
| Data set 3   | SLIDESPEECH          |https://www.openslr.org/144/                    |

| MODELS       | Name of Model used   |
|--------------|----------------------|
| Base model 1 | Llama 7b             |                                                |
| Base model 2 | Whisper              |
| Base model 3 | Sentence-Transformer |


### Summary of infrastructure requirements


| Requirement     | How many/when                                     | Justification       |
|-----------------|---------------------------------------------------|---------------------|
| `m1.medium` VMs | 3 for entire project duration                     |                     |
| `gpu_mi100`     | 4 hour block twice a week                         |                     |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |                     |
| etc             |                                                   |                     |

### Detailed design plan
Detailed Design Plan for Intelligent Multimedia Processing (IMP) System
Model Training and Training Platforms
## Unit 1
## Scale
#### Data
| **Dataset**                              | **Original Size** |
|------------------------------------------|-------------------|
| AMI Meeting Corpus                       | 56 GB             |
| ICSI Meeting Corpus                      | 1 GB              |
| Video on our drive                       | 20 MB             |

#### Model

| **Model**                | **Parameters**             | **Approx. Model Size**                      | **Inference Latency**                                                      |
|--------------------------|----------------------------|---------------------------------------------|----------------------------------------------------------------------------|
| **Llama 3B**             | 3 Billion                  | ~12-15GB (FP32; can be reduced via quantization)| ~0.5-1 second for typical query generation on a high-end GPU                |
| **Whisper**              | (Not typically measured)   | Lightweight (generally <1GB)                | Real-time or near real-time transcription (processing speed close to audio duration) |
| **longformer-base-4096** | ~149 Million               | ~600MB to 1GB                                | ~0.5-1 second per forward pass on GPU for sequences up to 4096 tokens        |


#### Deployment
We will deploy models on different configurations of CPUs and GPUs and compare performance on each, to obtain the most economical options for deployment. We will also utilize staging, canary and production environments to comprehensively test the service.

## Value Proposition
We are reimagining Atlassian’s Confluence (similar to Google Drive but more advanced in terms of features) by integrating a Retrieval-Augmented Generation (RAG) agent that empowers users to extract precise, context-rich insights from a vast repository of enterprise knowledge. This innovative approach transforms static documentation into a dynamic, interactive platform where queries yield targeted, actionable information, enhancing both collaboration and decision-making. By seamlessly merging advanced machine learning with robust knowledge management, our solution elevates the user experience and drives operational excellence across the organization.
#### 1. Enhancing Operational Efficiency
##### Automation:
The IMP system brings automation to tasks that once required countless hours of manual work. Imagine the challenge of sifting through thousands of client documents, compliance records, or recorded presentations—now, this system can handle that load effortlessly. It minimizes manual labor and drastically reduces the chance for human error.

##### Rapid Insights:
Leveraging cutting-edge deep learning for natural language processing, the system doesn’t just process data—it understands it. It quickly highlights essential details, spots irregularities, and raises alerts on potential risks, ensuring that nothing critical slips through the cracks.

##### Scalability:
As the volume of unstructured data grows with the client base and operations, the IMP system scales alongside the business. Whether it’s a sudden influx of documents or a surge in multimedia content, the system remains robust, ensuring Atlassian’s services stay reliable and responsive.

##### Enhanced Multimedia Data Handling:
Beyond traditional documents, the IMP system is specially designed to process various multimedia formats—like videos, images, and audio recordings. This means whether the data comes in the form of a detailed video briefing or a series of promotional images, the system can extract meaningful insights, further empowering the user's decision-making process.

#### 2. Strategic Business Advantages
##### Cost Reduction:
By automating the heavy lifting of data processing, the system helps slash operational costs. Less time spent on manual reviews means more resources can be allocated to strategic initiatives and high-value client engagements.

##### Improved Decision Making:
Faster, more accurate insights mean Atlassian’s consultants can deliver recommendations that are both timely and data-driven. This leads to better decisions that directly enhance client satisfaction and drive business growth.

##### Competitive Differentiation:
In today’s fast-paced consulting landscape, staying ahead means embracing innovation. By integrating advanced machine learning capabilities, Atlassian not only optimizes its operations but also reinforces its image as a forward-thinking leader in the industry.

#### Current status quo(non-ML)  used in the business : 
Confluence currently operates using traditional, non-ML approaches to manage and organize content. Here’s a breakdown of the status quo:
##### Manual Content Creation & Curation:
Content in Confluence is primarily created and curated by users. Teams rely on manually drafted pages, blogs, and documents, often following pre-defined templates. This requires significant human input for content creation, categorization, and updating.
##### Keyword-Based Search & Indexing:
The search functionality in Confluence is based on traditional indexing and keyword matching. Users search for content using specific keywords or tags, with results ranked by relevance using conventional algorithms rather than contextual or semantic analysis.
##### Static Organization & Tagging:
Content is organized using manually assigned labels, spaces, and hierarchies. There is no automated content categorization or dynamic reorganization based on usage patterns or content similarity, which means that maintaining an up-to-date structure is a largely manual effort.
##### Standard Collaboration Tools:
While Confluence offers robust collaboration features like version control, commenting, and page sharing, the recommendations and insights provided (such as related pages or recent updates) are driven by rule-based logic rather than personalized, machine learning–driven insights.

#### Business metrics 
To evaluate the success of the Intelligent Multimedia Processing (IMP) system, consider the following business metrics across key areas:
##### 1. User Engagement & Adoption
Adoption Rate – Percentage of enterprise users actively using IMP within a given period.
Active Users – Number of unique users leveraging IMP daily, weekly, or monthly.
Query Volume – Number of queries performed within the system, indicating usage frequency.
##### 2. Efficiency & Productivity Gains
Reduction in Retrieval Time – Average time saved in accessing relevant information compared to traditional methods.
Automation Rate – Percentage of manual document parsing and multimedia reviewing tasks replaced by IMP.
Query Success Rate – Percentage of queries that return relevant and useful results.
User Satisfaction Score – Feedback from users on ease of use and effectiveness.
##### 3. Business Impact & ROI
Cost Savings – Reduction in labor costs due to decreased manual document processing.
Operational Efficiency Gain – Productivity improvements quantified in work hours saved per employee.
Return on Investment (ROI) – Revenue or cost savings generated compared to implementation costs.
##### 4. Data Quality & Accuracy
Accuracy of Retrieval – Percentage of correctly retrieved multimedia content based on user queries.
Error Rate – Number of incorrect, incomplete, or irrelevant results returned.
##### 5. Future Scalability & Expansion
Multilingual Performance – Effectiveness of language support when new languages are integrated.
Enterprise Expansion Rate – Growth in the number of enterprises or business units adopting IMP.
Infrastructure Utilization – Performance metrics such as GPU load, query processing time, and storage efficiency.
Tracking these metrics will help measure IMP’s effectiveness, adoption, and business value while guiding future improvements.

## Outside Material
##### AMI Meeting Corpus

| **Aspect**                 | **Details**                                                                                                                                                                                                                                                                                               |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Name of Dataset**        | AMI Meeting Corpus                                                                                                                                                                                                                                                                                        |
| **Creator(s)**             | The AMI Consortium – a collaboration between universities and research labs across Europe (e.g., University of Edinburgh, IDIAP Research Institute, TNO, etc.)                                                                                                                                       |
| **Date of Creation**       | Initial release in 2005, with subsequent updates                                                                                                                                                                                                                                                         |
| **Purpose of Collection**  | Designed for research in multimodal conversational understanding, such as automatic speech recognition (ASR), speaker diarization, topic segmentation, summarization, etc.                                                                                                                            |
| **Conditions of Collection** | Meetings were scripted and unscripted, held in controlled environments, and involved multiple speakers with consent. Audio, video, and transcriptions were all collected.                                                                                                                            |
| **Academic Documentation** | Yes – documented in the paper: Carletta, J. (2007). “Unleashing the killer corpus: Experiences in creating the multi-everything AMI Meeting Corpus.” *Language Resources and Evaluation*.                                                                                                              |
| **Privacy Concerns**       | Minimal – participants gave informed consent; however, since real people were recorded, anonymity and ethical usage are still important.                                                                                                                                                                |
| **Fairness/Ethics Concerns** | The corpus may lack demographic diversity (e.g., accents, gender balance), which can introduce bias in downstream models trained using this data.                                                                                                                                                         |
| **Preprocessing Notes**    | Various levels of pre-processing available: raw audio/video, speaker-annotated audio, aligned transcripts, ASR outputs, and topic annotations. Some processed versions include segmented/chunked data.                                                                                                   |
| **License**                | Distributed under a Creative Commons Attribution Non-Commercial Share-Alike (CC BY-NC-SA) license.                                                                                                                                                                                                       |
| **Permissible Use Cases**  | Non-commercial academic and research purposes. Commercial use is prohibited without additional permissions.                                                                                                                                                                                              |
| **Where to Access**        | [AMI Corpus Access](https://groups.inf.ed.ac.uk/ami/corpus/)                                                                                                                                                                                                                                              |

---

##### ICSI Meeting Corpus

| **Aspect**                 | **Details**                                                                                                                                                                                                                                                                                              |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Name of Dataset**        | ICSI Meeting Corpus                                                                                                                                                                                                                                                                                       |
| **Creator(s)**             | International Computer Science Institute (ICSI), Berkeley, CA, USA                                                                                                                                                                                                                                      |
| **Date of Creation**       | Recorded between 2000 and 2002                                                                                                                                                                                                                                                                             |
| **Purpose of Collection**  | Designed to support research in automatic speech recognition (ASR), speaker diarization, meeting summarization, dialogue analysis, and other multimodal processing tasks.                                                                                                                             |
| **Conditions of Collection** | Natural, real meetings held by ICSI research teams (mostly speech and audio researchers). Audio recorded with multiple microphones per meeting for rich multi-channel capture.                                                                                                                       |
| **Academic Documentation** | Yes – see: Janin, A., Ang, J., et al. (2003). “The ICSI Meeting Corpus.” IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP).                                                                                                                                              |
| **Privacy Concerns**       | Participants provided consent for recording and research use. Still, meetings contain real speech from identifiable individuals, so ethical use respecting privacy is necessary.                                                                                                                    |
| **Fairness/Ethics Concerns** | Like many corpora, may reflect demographic bias (limited accents, cultural diversity, or gender balance), potentially affecting generalization or fairness in trained models.                                                                                                                      |
| **Preprocessing Notes**    | Distributed with transcripts, speaker segmentations, and channel-separated audio. Some versions include automatic annotations or phonetic alignments.                                                                                                                                                   |
| **License**                | Available under a Linguistic Data Consortium (LDC) license. Access typically requires an LDC membership or specific agreement.                                                                                                                                                                             |
| **Permissible Use Cases**  | Permitted for research and educational purposes under the LDC license. Commercial use may require additional licensing or permissions from ICSI or LDC.                                                                                                                                                  |
| **Where to Access**        | Through the Linguistic Data Consortium ([LDC Access](https://www.ldc.upenn.edu)), catalog ID: LDC2004S02                                                                                                                                                                                                 |




## Multiple Models: System Design Explanation

The system is composed of multiple machine learning models that work together—not just in parallel—to accomplish a complex goal: automated meeting understanding and response generation (e.g., query answering, meeting minutes generation). Each model contributes a unique capability that is critical for the system’s end-to-end function:

### Models Overview

| **Model**                 | **Role in the Pipeline**                                                                                                                                       | **Why It’s Necessary**                                                                                                                                                                                   |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Base Model 1: LLaMA 7B**| Used as a language model for generating responses and meeting minutes from retrieved contextual segments.                                                       | LLaMA is powerful for generation and reasoning based on context. It's essential for producing fluent, coherent text.                                                                                      |
| **Base Model 2: WHISPER**     | Performs speech-to-text transcription from audio input. This is the first model in the pipeline when the input is raw meeting audio.                              | Without this, the system cannot convert spoken language into usable text for downstream processing.                                                                                                       |
| **Base Model 3: Longformer**| Used to create text embeddings for long-form documents or transcripts, enabling semantic search over meeting content in the vector database (RAG engine).        | Enables contextual retrieval of relevant segments for LLaMA to reason over. Long context handling is essential.                                                                                            |

### Why All Three Models Are Required Together

- **WHISPER** converts audio files into readable, timestamped text.
- **Longformer** embeds the resulting (long) transcripts for semantic indexing in a vector database.
- **LLaMA 7B** is used in the RAG (Retrieval-Augmented Generation) loop, receiving the most relevant transcript segments and generating:
  - Natural language responses to queries.
  - Summarized meeting minutes.


### Model training and training platforms
## Strategy

Our training strategy employs distributed training techniques for the Longformer-base-4096 embeddings model and fine-tuning of Llama 7B using LORA (Low-Rank Adaptation) to optimize performance while minimizing computational requirements. We'll implement a continuous training pipeline that automatically triggers retraining based on drift detection.
Relevant Parts of the Architecture

MLFlow for experiment tracking and model versioning
Ray clusters for distributed training
Longformer-base-4096, pretrained on long documents
Llama 7B with LORA optimizations
Model Registry for version control and deployment

Justification

Distributed Training: The ICSI Meeting Recorder Corpus (39GB) contains substantial audio data requiring efficient parallelized processing. Ray's distributed training framework enables us to scale across multiple nodes on Chameleon infrastructure.
Model Selection: Longformer-base-4096 (560MB) is specifically designed for long context (up to 4,096 tokens), making it ideal for meeting transcripts and lengthy documents while remaining small enough (560MB) for efficient deployment.
Parameter-Efficient Fine-tuning: LORA reduces the number of trainable parameters for Llama 7B by approximately 95%, enabling fine-tuning on limited resources while maintaining performance.

Relation to Lecture Material
This approach implements the distributed training patterns discussed in the "Training at Scale" lectures, specifically:

Data Parallelism through Ray's DDP implementation
Hyperparameter optimization via Ray Tune
Parameter-efficient fine-tuning techniques (LORA)
Experiment tracking best practices using MLFlow

## Specific Numbers

Training Infrastructure: 4 GPU nodes on Chameleon Cloud (each with NVIDIA T4)
Training Dataset: 39GB ICSI Meeting Corpus with approximately 75 hours of meeting recordings
Batch Size: 32 per GPU (128 global batch size)
Hyperparameter Tuning: 30 trials using Ray Tune with Bayesian optimization
Training Time: Approximately 12 hours for full embedding model training
LORA Configuration: Rank 8, alpha 16, reducing trainable parameters from ~7B to ~35M

## <!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, and which optional "difficulty" points you are attempting. -->

## Model serving and monitoring platforms
Strategy
We'll implement a multi-tier serving strategy with optimizations for both latency-sensitive and throughput-oriented workloads. FastAPI will serve as our API layer, with optimized inference paths for both CPU and GPU deployment options. Comprehensive monitoring will track model performance, drift, and system health.
Relevant Parts of the Architecture

FastAPI service layer
TorchServe for model deployment
Quantized models for CPU inference
GPU acceleration for batch processing
MLFlow dashboards for monitoring
Interactive data dashboard for visualization

Justification

Inference Optimization: Quantization reduces the Longformer model size by 75% (from 560MB to ~140MB) while maintaining 97%+ of accuracy, enabling faster CPU inference for low-latency requirements.
Multiple Serving Options: Different query patterns (interactive vs. batch processing) require different optimization strategies. Our architecture supports both GPU-accelerated batch processing and optimized CPU inference.
Comprehensive Monitoring: Early detection of model drift is crucial for maintaining accuracy in enterprise environments where data patterns evolve.

Relation to Lecture Material
This implements the serving concepts from "Model Serving" lectures:

Model optimization techniques (quantization, operator fusion)
Batch vs. real-time inference patterns
Zero-downtime deployment strategies
A/B testing for model deployments

Specific Numbers

Latency Target: <200ms for query processing (from request to embedding generation)
Throughput: 50 queries per second per node
Model Size Reduction: From 560MB to ~140MB through INT8 quantization
Monitoring Frequency: Performance metrics collected every 30 seconds
Alerting Threshold: Alert if accuracy drops below 92% or latency exceeds 500ms
Canary Deployment: 10% traffic to new model versions for validation


#### Data pipeline
Strategy
Our data pipeline focuses on efficient processing of multimedia inputs, with separate paths for documents, audio, and video. We'll implement both batch processing for historical data and streaming capabilities for real-time meeting analysis. A comprehensive data dashboard will provide insights into data quality and processing efficiency.
Relevant Parts of the Architecture

Document, audio, and video processing pipelines
VOSK for speech-to-text conversion
Chunking module for text segmentation
Metadata extraction and storage
Chameleon persistent storage
Interactive data dashboard

Justification

Specialized Processing Paths: Different media types require specialized processing techniques. Our pipeline handles each appropriately while converging to a common text representation.
Persistent Storage Strategy: Enterprise data requires secure, persistent storage with proper versioning and access controls.
Streaming Processing: Real-time meeting analysis requires low-latency processing of audio streams, necessitating an optimized streaming pipeline.

Relation to Lecture Material
This implements the data engineering concepts from lectures:

ETL pipeline design
Data quality monitoring
Offline vs. online data processing
Feature store concepts (for embeddings and metadata)

Specific Numbers

Storage Allocation: 150GB persistent storage on Chameleon
Used Storage: ~ 63 GB
Chunking Configuration: Semantic chunks of 300-500 tokens with 50-token overlap
Data Retention: Till 15th MAy 2025
Dashboard Refresh: Real-time metrics with 60-second refresh interval

#### Continuous X
Strategy
We'll implement a comprehensive CI/CD pipeline integrated with continuous training and monitoring to ensure both code and model quality. GitOps principles using ArgoCD will enable declarative infrastructure and application deployment, with PythonChi providing Infrastructure as Code capabilities.
Relevant Parts of the Architecture

GitHub repository for version control
CI/CD pipeline for testing and deployment
PythonChi for IaaC
Docker containerization
Kubernetes orchestration
ArgoCD for GitOps
Staged deployment patterns

Justification

GitOps Approach: Declarative infrastructure ensures consistency between environments and enables rapid rollback if needed.
Continuous Training: Automating the retraining process based on drift detection ensures model accuracy over time without manual intervention.
Staged Deployments: Progressive rollout of updates minimizes risk and enables validation before full deployment.

Relation to Lecture Material
This implements DevOps and MLOps concepts from lectures:

CI/CD pipeline design
Infrastructure as Code principles
Container orchestration patterns
GitOps deployment methodology
Continuous training workflows

Specific Numbers

Test Coverage: Minimum 85% code coverage for all components
Deployment Frequency: Support for multiple deployments per day if needed
Rollback Time: <5 minutes for reverting to previous stable version
Infrastructure Provisioning: <15 minutes from commit to complete environment setup
Deployment Stages: Development → Testing → Staging → Production with automated promotion
Model Validation: Automated evaluation against benchmark dataset (95% accuracy threshold) before promotion



