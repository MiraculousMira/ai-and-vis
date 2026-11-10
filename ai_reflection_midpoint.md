# AI-Assisted Reflection: Phase I (Vis4AI)

**Team Name:** semi motivated team  
**AI Users:** 
* k51850345, Gerald Mitterecker
* 12213736, Thomas Karasin

---

## 1. General Workflow & AI-Assisted Process
Planning and implementing this project with AI (Google Gemini) was very convenient and straightforward. First, a planning phase was conducted, resulting in a roadmap. Afterwards, each step of the roadmap was implemented one after another. Quite a lot of reprompting and refining needed to be done to get good-looking and interesting results. 

## 2. Key Analytical Decisions & AI Impact
* **Pivoting from Slow Loops to Efficient Statistics:** Initially, the AI suggested looping a Vision-Language Model (VLM) through every single test image for text tagging. Recognizing this as inefficient and resource-heavy for our workflow, we modified the approach. We decided instead to use zero-cost automated image heuristics (RGB color channels, brightness quartiles, and resolution ratios via Pillow/NumPy) combined with ResNet-50 feature embeddings and UMAP latent space clustering.
* **Iterative Code Refinement:** Code snippets for training, evaluation, and UMAP plotting were generated iteratively. For example, when default K-Means clustering failed to reveal error patterns, we prompted the AI to switch to UMAP dimensionality reduction with numbered error annotations.

## 3. Observed AI Failure Patterns & Limitations
* **Overcomplication & Tunnel Vision:** LLMs sometimes tend to overcomplicate things or suffer from tunnel vision in certain areas (e.g., suggesting heavy external web apps or complex pipelines when a simple Jupyter notebook cell would do). 
* **Blind Spot Acceptance Risk:** The AI occasionally produced code that assumed persistent dataframe states across cell re-runs (leading to missing column errors). This behavior had to be immediately called out and reprompted. We learned to never accept results blindly, always checking them since we are the ones responsible for handing them in.