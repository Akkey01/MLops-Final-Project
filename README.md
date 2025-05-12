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

| MODELS       | Name of Model used   |
|--------------|----------------------|
| Base model 1 | Flan-T5-Large        |                                                
| Base model 2 | Whisper              |
| Base model 3 | Sentence-Transformer |


### Summary of infrastructure requirements


| Requirement     | How many/when                                     | Justification       |
|-----------------|---------------------------------------------------|---------------------|
| `m1.xlarge` VMs | 3 for entire project duration                     | Docker Image of the entire system was 20GB|
| `gpu_a100`      | 4 hour block twice a week                         |                                           |
| Floating IPs    | 2 for entire project duration, 1 for sporadic use |                                           |
| etc             |                                                   |                                           |

### Detailed design plan
Detailed Design Plan for Intelligent Multimedia Processing (IMP) System
![mlops 2nd](https://github.com/user-attachments/assets/c73aa631-9e6c-4be6-a19a-0ec475899586)

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
| **Flan-T5-Large**        | 783 M                      | ~1.02GB                                     | ~0.5-1 second for typical query generation on a high-end GPU                |
| **Whisper**              | (Not typically measured)   | Lightweight (generally <1GB)                | Real-time or near real-time transcription (processing speed close to audio duration) |
| **Sentence Transformer** | ~149 Million               | ~600MB to 1GB                               | ~0.5-1 second per forward pass on GPU for sequences up to 4096 tokens        |


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
| **Base Model 1: Flan-T5-Large**| Used as a language model for generating responses and meeting minutes from retrieved contextual segments.                                                       | Flan-T5-Large is powerful for generation and reasoning based on context. It's essential for producing fluent, coherent text.                                                                                      |
| **Base Model 2: WHISPER**     | Performs speech-to-text transcription from audio input. This is the first model in the pipeline when the input is raw meeting audio.                              | Without this, the system cannot convert spoken language into usable text for downstream processing.                                                                                                       |
| **Base Model 3: Sentence Transformer**| Used to create text embeddings for long-form documents or transcripts, enabling semantic search over meeting content in the vector database (RAG engine).        | Enables contextual retrieval of relevant segments for Flan-T5-Large to reason over. Long context handling is essential.                                                                                            |

### Why All Three Models Are Required Together

- **WHISPER** converts audio files into readable, timestamped text.
- **Sentence Transformer** embeds the resulting (long) transcripts for semantic indexing in a vector database.
- **Flan-T5-Large** is used in the RAG (Retrieval-Augmented Generation) loop, receiving the most relevant transcript segments and generating:
  - Natural language responses to queries.
  - Summarized meeting minutes.


### Model training and training platforms\


### Model training and training platforms
The model was trained using LoRA. The model flan-t5-large was selected due to it's suitability for summarization. We experimented with different values of batch size and observed that a small batch size of 2 performs the best. We also observed that the best performance is with a learning rate of 1e-4 and 5 epochs.

We utilized MLFlow and Ray for logging experiments and understanding the impact of different hyperparameters, and feeding training jobs,
## Strategy

Our training strategy employs distributed training techniques for the Sentence Transformer embeddings model and fine-tuning of Flan-T5-Large using LORA (Low-Rank Adaptation) to optimize performance while minimizing computational requirements. We'll implement a continuous training pipeline that automatically triggers retraining based on drift detection.
Relevant Parts of the Architecture

MLFlow for experiment tracking and model versioning
Ray clusters for distributed training
Sentence Transformer, pretrained on long documents
Flan-T5-Large with LORA optimizations
Model Registry for version control and deployment

Justification

Distributed Training: The ICSI Meeting Recorder Corpus (39GB) contains substantial audio data requiring efficient parallelized processing. Ray's distributed training framework enables us to scale across multiple nodes on Chameleon infrastructure.
Model Selection: Sentence Transformer (560MB) is specifically designed for long context (up to 4,096 tokens), making it ideal for meeting transcripts and lengthy documents while remaining small enough (560MB) for efficient deployment.
Parameter-Efficient Fine-tuning: LORA reduces the number of trainable parameters for Flan-T5-Large by approximately 95%, enabling fine-tuning on limited resources while maintaining performance.

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
Quantized models for CPU inference
GPU acceleration for Faster inferencing
Prometheus dashboards for metrics
Interactive Grafana dashboard for visualization
Airflow for testing suites

Justification

Inference Optimization: Quantization reduces the Sentence Transformer model size by 75% (from 560MB to ~140MB) while maintaining 97%+ of accuracy, enabling faster CPU inference for low-latency requirements.
Multiple Serving Options: Different query patterns (interactive vs. batch processing) require different optimization strategies. Our architecture supports both GPU-accelerated batch processing and optimized CPU inference.
Comprehensive Monitoring: Early detection of model drift is crucial for maintaining accuracy in enterprise environments where data patterns evolve.

Relation to Lecture Material
This implements the serving concepts from "Model Serving" lectures:

Model optimization techniques (ONNX)
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


## Data pipeline
#### 1. Persistent Storage

1. *Block-storage volume (KVM)*  
   - A dedicated Chameleon block volume is attached at /mnt/block on your VM.  
   - All containers in docker-compose.yml mount it as /data:
     yaml
     volumes:
       - /mnt/block:/data
     
   - *Raw* and *processed* files persist across container restarts.

2. *Object-storage container (CHI-TACC Swift)*  
   - A Swift container named object-persist-project39 holds final artifacts.  
   - Rclone is configured in ~/.config/rclone/rclone.conf under the [chi_tacc] profile.  
   - load-data.sh pushes /data/processed/*.jsonl into swift:object-persist-project39/processed.
#### 2. Offline Data Pipeline

All offline (training) data lives under the same block volume and is moved to object storage when ready.  

1. **Extract (extract-data.sh)**  
   - Downloads 56 GB of AMI signals via your wget.txt manifest.  
   - Grabs both manual & automatic NXT annotations:  
     - ami_public_manual_1.6.2.zip → raw/ami/manual_annotations/…  
     - ami_public_auto_1.5.1.zip   → raw/ami/automatic_annotations/…  
   - Unpacks any existing amicorpus.tgz if present.

2. **Transform (transform-data.sh)**  
   - Walks *all* raw subfolders (raw/ami, raw/meetingbank), supports:  
     - .trs, .txt, .json, NXT XML, PDF, DOCX, PNG/JPG (OCR), etc.  
   - Emits clean JSON-lines per category:
     - manual.jsonl  
     - automatic.jsonl  
     - transcripts.jsonl  
   - Collates metadata (meetingID, file path, etc.) for each record.

3. **Load (load-data.sh)**  
   - Verifies local /data/processed contents & sizes.  
   - Uses rclone copy --checksum --progress to push to:  
     
     swift:object-persist-project39/processed
     
   - Lists remote contents before & after to confirm success.

#### 3. **Data pipelines**

1. **Data Source**
   - **ICSI Meeting Signals**  
  - Audio `.wav` files for meeting IDs  
    - Downloaded via `etl/extract/icsi.sh` from  
      `https://groups.inf.ed.ac.uk/ami/ICSIsignals/NXT/<MID>.interaction.wav`  
  - Manifest & license text from  
    `https://groups.inf.ed.ac.uk/ami/download/temp/icsiBuild-15735-Sun-May-11-2025.manifest.txt`  
    and `CCBY4.0.txt`

- **Custom Video Inputs** (replacing OpenSLR)  
  - Two MP4 files hosted on Google Drive  
    - IDs `1bbmmYdlnkYwrkoULIa-IEhb6g80i4HwW` and `1Sc2OemI3c7blKMFAQnGnbSTgAE7IZz-6`  
  - Downloaded via `gdown` in `extract-video` service
 
2. **Offline ETL Pipeline**
   -All steps run **in Docker**; code lives under `etl/`.


cd etl
docker compose run --rm extract-icsi      # 👉 Raw ICSI .wav + metadata onto /mnt/block/raw/icsi
docker compose run --rm extract-video     # 👉 Raw videos onto /mnt/block/raw/video
docker compose run --rm convert-video     # 👉 ffmpeg: MP4 → 16 kHz mono WAV
docker compose build whisper-builder      # 👉 Build Whisper image (caches weights)
docker compose run --rm transcribe-icsi   # 👉 Whisper transcribes ICSI → transcripts_icsi.jsonl
docker compose run --rm transcribe-video  # 👉 Whisper transcribes video audio → transcripts_video.jsonl
docker compose run --rm build-chunks      # 👉 chunk_text → train/val/prod JSONL in /mnt/block/processed
docker compose run --rm embed-index       # 👉 Sentence-Transformer embed + FAISS index → /mnt/block/faiss_base
docker compose run --rm push-object       # 👉 rclone pushes faiss_base → Swift `object-persist-project39/faiss_base`

---
3. **Data Pipeline and Online data** 
Simulates production inference traffic:

cd streaming
docker compose up -d simulator
simulate_requests.py reads prod_seed_chunks.jsonl

Sends each chunk as a POST to your RAG API (RAG_ENDPOINT_URL) at configurable rate
Logs {"timestamp", "latency", "status"} to /mnt/block/metrics/stream_metrics.jsonl

Characteristics of simulated data

Rate: 0.2 requests/sec (adjustable via --rate)

Distribution: exact meeting-IDs held out for “prod_seed” in splits.yaml

Realism: uses actual chunks derived from raw transcripts

4. **Interactive Dashboard**
Implemented in Streamlit, two tabs:


cd rag_app
streamlit run app.py  # reads RAG_ENDPOINT_URL from .env
Chat: query RAG inference API, view Q/A history

Dashboard: reads /mnt/block/metrics/*.jsonl, displays:

Latency-over-time line chart

### Usage

1. Provision storage (if you include provision/ scripts).  
2. Clone & cd into this directory.
3. Ensure your Swift credentials are in ~/.config/rclone/rclone.conf.  
4. Run:
   
   docker compose up --rm extract-data
   docker compose up --rm transform-data
   docker compose up --rm load-data

Specific Numbers

Storage Allocation: 150GB persistent storage on Chameleon
Used Storage: ~ 63 GB
Chunking Configuration: Semantic chunks of 300-500 tokens with 50-token overlap
Data Retention: Till 15th MAy 2025
Dashboard Refresh: Real-time metrics with 60-second refresh interval

## Continuous X
Implemented a comprehensive CI/CD pipeline integrated with continuous training to ensure code and model quality. GitOps principles using ArgoCD enable declarative infrastructure and application deployment, with PythonChi providing Infrastructure as Code capabilities.

Relation to Lecture Material: This implements the devops concepts from "DevOps" lectures and Labs.

I have created the terraform_ansible.ipynb file in the root directory which runs as bash script to Provision infrastructure ,configuration ,resources and runs playbooks and deploys stages 
#### 1.GitHub repository for version control
Everything in version control: Terraform manifests, Ansible playbooks, Kubespray code, Argo Workflows, Argo CD manifests, Helm/Kustomize charts, and application source.  

#### 2.Terraform: Infrastructure-as-Code 
Declarative provisioning: All network, compute, and security configurations live in version control. Terraform manifests describe private networks, subnets, security groups, three Ubuntu instances, and a publicly routable floating IP.
Automated configuration: After infrastructure appears, Ansible playbooks declare every package, service, and Kubernetes component. Hosts are never updated by hand; changes flow from Git to the cluster.Infrastructure Provisioning: <15 minutes from commit to complete environment setup

#### 3. Cloud-Native Practices
Immutable infrastructure: Once VMs are provisioned, configuration drift is prevented. Updates require adjustments to code, not in-place edits.
Microservices architecture: The model-serving API, CI/CD controllers (Argo CD, Argo Workflows), and cluster services each run as isolated containers or small pods, communicating over well-defined APIs.
Containers as compute units: All application logic and tooling—even training steps—live in Docker images, enabling consistent environments across development, staging, and production.

#### 4. CI/CD 
The bash file in node1 clones the git repository on node1,builds the docker image  and runs the docker container.
GitOps-style deployments:Workflow outputs (new image tags) update manifest files in Git; Argo CD observes these changes and synchronizes them into the cluster.
The scripts for trigger worflows, staging, canary and production are added.







