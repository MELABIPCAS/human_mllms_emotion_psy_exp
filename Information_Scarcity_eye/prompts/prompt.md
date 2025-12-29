### Table 6: Structured Prompts for MLLM Evaluation (Task II)

| Task Step      | Prompt Content                                                                                                                                                     |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Context        | The image shows two faces (or partial regions). Left is Neutral; Right is a Basic Emotion. Analyze the transition.                                         |
| Q1: Valence    | Analyze the valence of the emotion in the right image relative to the left. Output a number from 1.00 (Very Positive) to 5.00 (Very Negative).             |
| Q2: Category   | Identify the specific emotion type. Choose exactly one from: [Angry, Disgust, Fear, Happiness, Sadness, Surprise]. Do not provide explanations.            |
| Q3: Intensity  | Analyze the intensity of the emotion in the right image relative to the left. Output a number from 1.00 (Very Weak) to 5.00 (Very Strong).               |