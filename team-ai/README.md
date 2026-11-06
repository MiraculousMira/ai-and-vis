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

### Step 3: Interactive Visualization & Deep-Dives
* **Goal:** Move beyond a aggregate accuracy percentage by visualizing performance clusters.
* **Key Tasks:**
  * Project model feature embeddings into 2D (using UMAP or t-SNE).
  * Build an interactive scatter plot (e.g., Plotly) where hovering shows the image, predictions, confidence, and tags.
  * Investigate specific clusters: *What shared features explain why ResNet-50 struggles on certain sub-groups?*

---

## 📝 Mandatory AI Collaboration Log (AI-Aided Pair)

*As per assignment rules, we document how GenAI supported our analytical process.*

### 1. Key Prompts & Analytical Impact
| Task / Goal | Key Prompt Used | Output / AI Suggestion | Action Taken (Accepted / Modified / Rejected) |
| :--- | :--- | :--- | :--- |
| Plan roadmap | *"I want to plan with you on how to best fulfill this courses task. Course description: ...  We chose this model and dataset: \<link\>. Furthermore we would like to use resnet-50 as a comparison. We thought about tagging the images and see if there are tags where the model struggles or has interesting behaviour but I am open for different approaches if you have some"* | *README you are currently reading* | *Modified: reprompted for markdown format and accepted afterwards* |
| Step 1 script | "Looks good, then lets start with step 1. Can you give me a python script? I will make the environment work." | step1_baseline_inference.py | Modified: needed to reprompt since training was not implemented, accepted afterwards |

### 2. Observed AI Failure Patterns
* *Unconventional/Contextual Muffins:* Muffins that dont really look like muffins threw the model off. For example: nut bar wrapper with blueberry muffin flavour written on it (75% confident), muffin ingredients in bowls next to each other (54%) or small muffins in front of something else like tabaco pouches (67% for bigger and 79% for smaller version).
* *Carricature Dogs:* The model seems to struggle with drawn pictures of Chihuahuas, e.g. christmastree decoration (55%) or picture on dog food (63%, but very small resolution).
* *Out-of-Distribution (OOD) Dataset Noise:* For reasons unknown, the muffin test set contains stray images of plants (e.g., cannabis buds). Interestingly, the model produced its highest-confidence errors on these anomalies (80%–86%), likely mistaking dense textures or pistils for dog fur.

### 3. Emerging Best Practices for AI in Model Analysis
* **Rule 1:** Refactoring code together with AI makes it a lot easier and faster. Most of the time models produce code that is nearly there, but needs some polishing.
* **Rule 2:** ...

---

## 🚀 Setup & Execution

TODO
```bash
# Clone the repo
git clone https://github.com/your-username/muffin-chihuahua-vis4ai.git
cd muffin-chihuahua-vis4ai

# Install dependencies (Example)
pip install torch torchvision pandas plotly umap-learn
```