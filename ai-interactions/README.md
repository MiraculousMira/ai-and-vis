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

### 2. Emerging Best Practices for AI in Model Analysis
* **Rule 1:** Refactoring code together with AI makes it a lot easier and faster. Most of the time models produce code that is nearly there, but needs some polishing.
* **Rule 2:** Do not take the very first version of a visualization. Refinement with the AI can make it way more interesting.
* **Rule 3:** Do not hesitate to push back. The AI often defaults to basic shortcuts, so giving direct, critical feedback is necessary to get useful outputs.
* **Rule 4:** Trust the raw data over the AI. AI tools are great for speeding up tagging and analysis, but you still need to double check their outputs against the actual dataset.