# Project: Muffin vs. Chihuahua Model Investigation
**Team Name:** Semi-Motivated Team  
**Track:** Vis4AI — Phase I (AI-Aided Pair)

---

## 🎯 Overview & Goal

The goal of this phase is to analyze how a **ResNet-50** model behaves when classifying images from the classic **Muffin vs. Chihuahua** dataset. We treat the model as a black box—focusing on inputs, confidence levels, edge cases, and visual patterns to discover *why* and *when* the model makes mistakes.

---

## 🗺️ Project Roadmap (Phase I)

### Step 1: Baseline Inference & Error Extraction
* **Goal:** Run the dataset through ResNet-50 and capture output predictions alongside confidence scores.
* **Key Tasks:**
  * Extract true labels vs. predicted labels.
  * Compute confidence/softmax scores.
  * Isolate correct predictions, misclassifications, and high-confidence errors (e.g., predicted Muffin with 95% confidence on a Chihuahua).

### Step 2: Practical AI-Assisted Image Tagging
* **Goal:** Avoid manually labeling hundreds of images while still gaining rich insights into model behavior.
* **Key Tasks:**
  * Use a Vision-Language Model (VLM) / AI assistant to generate structured metadata tags for the dataset.
  * **Tags to extract:**
    * *Background/Context:* Plain, kitchen background, outdoor, cluttered.
    * *Key Visual Features:* Wrapper/cupcake liner present, close-up vs. full body/plate, dark vs. bright lighting.
    * *Texture/Anatomy Cues:* Visible eyes/nose, dark spots (blueberries vs. eyes), wrinkled texture.

### Step 3: Interactive Visual Analytics Dashboard
* **Goal:** Move beyond aggregate accuracy metrics by building an interactive visual analytics framework to evaluate model error clusters.
* **Key Tasks:**
  * Project high-dimensional ResNet-50 feature space representations into 2D using UMAP.
  * Develop an interactive dashboard featuring a stateful embedding scatter plot with hover diagnostics and linked lasso/box region selection.
  * Integrate dynamic UI components.

---

## 📝 Mandatory AI Collaboration Log (AI-Aided Pair)

*As per assignment rules, we document how GenAI supported our analytical process.*

### 1. Key Prompts & Analytical Impact
| Task / Goal | Key Prompt Used | Output / AI Suggestion | Action Taken (Accepted / Modified / Rejected) |
| :--- | :--- | :--- | :--- |
| Plan roadmap | *"I want to plan with you on how to best fulfill this courses task. Course description: ...  We chose this model and dataset: \<link\>. Furthermore we would like to use resnet-50 as a comparison. We thought about tagging the images and see if there are tags where the model struggles or has interesting behaviour but I am open for different approaches if you have some"* | *README you are currently reading* | *Modified: reprompted for markdown format and accepted afterwards* |
| Step 1 script | *"Looks good, then lets start with step 1. Can you give me a python script? I will make the environment work."* | step1_baseline_inference.py | Modified: needed to reprompt since training was not implemented, accepted afterwards |
| Step 2 implementation options | *"Ok how would we best tackle step 2? I have access to an llm AI agent that could look at each picture, but I think this is highly inefficient. What would be quick and easy alternatives for tagging the pictures?"* | 3 options: <ul><li> The Embedding & Clustering Shortcut (Recommended)</li><li> Simple Heuristics & Image Statistics </li><li> Batched / Sampled VLM Tagging</li></ul> | Accepted option 1 and 2 |
| Step 2 notebook | *"I think I want to do 1 and 2. For visualizing this a jupyter notebook would be best right?"*| step2_analysis.ipynb | Modified: some refactoring and reprompting for interesting visualizations, accepted afterwards|
| Step 3 Dashboard | *"Analyse step one and two implementations. is it ready for step three? Propose a detailed plan for step three."* | AI proposed an interactive visual analytics dashboard with linked brushing. | **Accepted & Expanded:** Shifted from simple HTML generation to a local web app (Dash/Plotly) to enable deep exploratory analysis. |
| Interactive State | *"can we make the confusion matrix adaptable to the selected samples?"* | AI identified callback conflict wiping selections and proposed decoupled state handling. | **Accepted:** Decoupled canvas dependencies so lasso bounds remain active while recalculating dynamic metrics. |

---

### 2. Observed AI Failure Patterns
* *Unconventional/Contextual Muffins:* Muffins that dont really look like muffins threw the model off. For example: nut bar wrapper with blueberry muffin flavour written on it (75% confident), muffin ingredients in bowls next to each other (54%) or small muffins in front of something else like tabaco pouches (67% for bigger and 79% for smaller version).
* *Carricature Dogs:* The model seems to struggle with drawn pictures of Chihuahuas, e.g. christmastree decoration (55%) or picture on dog food (63%, but very small resolution).
* *Out-of-Distribution (OOD) Dataset Noise:* For reasons unknown, the muffin test set contains stray images of plants (e.g., cannabis buds). Interestingly, the model produced its highest-confidence errors on these anomalies (80%–86%), likely mistaking dense textures or pistils for dog fur.
* *Oversimplified Visual Mockups:* When asked to generate visual previews for data points, the model initially defaulted to rigid placeholder graphics that failed to capture real-world data, risking misleading interpretations of model behavior.


### 3. Emerging Best Practices for AI in Model Analysis
* **Rule 1:** Refactoring code together with AI makes it a lot easier and faster. Most of the time models produce code that is nearly there, but needs some polishing.
* **Rule 2:** Do not take the very first version of a visualization. Refinement with the AI can make it way more interesting.
* **Rule 3:** Do not hesitate to push back. The AI often defaults to basic shortcuts, so giving direct, critical feedback is necessary to get useful outputs.
* **Rule 4:** Trust the raw data over the AI. AI tools are great for speeding up tagging and analysis, but you still need to double check their outputs against the actual dataset.

---

### 💡 How AI Broadened Analysis & Interaction (AI-Aided Scope)
* **Broader Analytical Scope:** Using AI assisted heuristic tagging and embedding clustering (Step 2) allowed us to analyze visual features across the entire dataset without manual time intensive annotations.


## 🚀 Setup & Execution

To set up the environment and run the analysis pipeline, follow these steps:

```bash
# 1. Clone the repository
git clone [https://github.com/MiraculousMira/ai-and-vis.git](https://github.com/MiraculousMira/ai-and-vis.git)
cd ai-and-vis

# 2. Install dependencies
pip install torch torchvision numpy pandas matplotlib seaborn pillow scikit-learn tqdm umap-learn plotly dash jupyter

# 3. Run Step 1: Baseline Inference
python step1_baseline_inference.py

# 4. Explore Step 2: Feature Tagging & Embeddings
jupyter notebook step2_analysis.ipynb

# 5. Launch the Step 3 Interactive Visual Analytics Dashboard
python step3_dashboard.py
```