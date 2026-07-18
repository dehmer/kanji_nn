Topic: Component Composition
Bot: ChatGPT
Date: 26 07 07

> Please consider the following context:
* LSTM model training for Kanji classification on KanjiVG vector data
* Input tensor (possible downsampled) columns: Δx, Δy, s(t) [0, 1] and pen-down/up (1, 0)
* Data augmentation on x, y and Δx, Δy
* And a few (hopefully) minor details

You're still with me?

> I just want to check a very specific idea with you. You may be familiar with KanjiVG, knowing that component composition (or decomposition) is encoded in KanjiVG SVG files through 'path groups'. Now check this out: What would you say, if I add an additional column to input tensor, holding the depth of the stroke's point (normalised to [0, 1]). This would give the model the opportunity to infer which points belong somehow together. Freaky idea or something to consider, and why?

It is a highly practical idea worth considering, not "freaky" at all. You are essentially proposing a structural positional encoding. Instead of forcing the LSTM to entirely memorize complex stroke hierarchies from temporal sequences alone, you are explicitly feeding it the tree depth from the XML/SVG hierarchy.
Here is an analysis of why this can work, along with the potential pitfalls to watch out for.
## Why It Can Work (The Pros)

* Explicit Hierarchical Context: Kanji are built from radicals (components). A stroke in a deep sub-group (e.g., depth 4) belongs to a fine-grained sub-component, while a stroke at depth 1 belongs to the main structure. Your feature gives the LSTM explicit boundaries of where components change.
* Overcoming LSTM Memory Horizon: While LSTMs handle sequential data well, they struggle with long-term hierarchical dependencies. If a Kanji has 20+ strokes, the LSTM might forget the radical context from the beginning. A depth feature acts as a continuous structural anchor.
* Aiding Downsampled Data: Since you mentioned your data might be downsampled, you are losing fine-grained temporal resolution. Injecting structural depth compensates for lost trajectory smoothness by preserving structural identity.

## Potential Bottlenecks (The Cons)

* Lack of Horizontal Context: Depth only tells the model how deep the component is, not which component it shares a level with. For example, two completely different radicals on opposite sides of a Kanji might both sit at depth 2. The model could get confused if it treats all "depth 0.5" strokes as structurally related.
* Ambiguity in Normalization: If Kanji A has a maximum depth of 2, and Kanji B has a maximum depth of 5, how do you normalize?
* If you normalize per Kanji (Local): Depth 2 in Kanji A becomes 1.0. Depth 2 in Kanji B becomes 0.4. The same structural level now has different values.
   * If you normalize globally (Global): You divide everything by the absolute maximum depth found in the entire dataset (e.g., 6). This is usually preferred.

## How to Improve the Idea
If you want to maximize the utility of this structural data, consider these alternatives to raw normalized depth:

   1. One-Hot Encoded Depth: Instead of a float between 0 and 1, use a small one-hot vector (e.g., 4 bins for depths 1, 2, 3, and 4+). Neural networks often handle discrete structural levels better as categorical embeddings than continuous floats.
   2. Component ID Transitions (Delta Depth): Instead of the raw depth, pass a binary flag (0 or 1) indicating "Did the path group change on this stroke?" This explicitly tells the LSTM, "You are entering a new structural component now," acting as a hard boundary trigger.

Would you like to explore how to extract and format these XML path group transitions using Python, or should we look at how to structure the embedding layer for this feature in your LSTM?

> I understand and consider the seconds point. For the first point: I don't know what One-Hot encoding is. I read it several times, but never bothered to ask (until now). Please explain.

One-Hot Encoding is a way to turn labels or categories into numbers so a machine learning model can understand them without getting confused. [1, 2, 3]
## The Problem it Solves
Imagine you have three categories: Cat, Dog, and Bird.
If you assign them simple numbers like:

* Cat = 1
* Dog = 2
* Bird = 3

Your LSTM model will look at those numbers and think, "Ah, a Bird (3) is worth three times more than a Cat (1). And a Dog (2) is the exact average of a Cat and a Bird." [4]
But a dog is not the average of a cat and a bird. The numbers introduce a fake math relationship (order and scale) that does not actually exist. [5, 6, 7, 8, 9]
## How One-Hot Encoding Works
Instead of using a single number that changes in value, you create a list of zeros and ones (a vector). The length of the list equals the total number of categories. Only one item in the list is "hot" (set to 1), while all the others are "cold" (set to 0). [10, 11, 12, 13, 14]
Using the same example:

* Cat = [1, 0, 0]
* Dog = [0, 1, 0]
* Bird = [0, 0, 1]

Now, every category is completely independent. No category is "larger" or "smaller" than another. They are just different directions in space. [15, 16, 17, 18]
## Applied to Your Kanji Project
If you use your normalized depth idea, a depth of 2 out of 4 becomes 0.5. The model might incorrectly assume depth 0.5 is halfway in meaning between depth 0.0 and 1.0.
If you use One-Hot Encoding for your SVG path depths (assuming a max depth of 4), it looks like this:

* Depth 1: [1, 0, 0, 0]
* Depth 2: [0, 1, 0, 0]
* Depth 3: [0, 0, 1, 0]
* Depth 4: [0, 0, 0, 1]

Instead of adding one single float column to your input tensor, you would add four columns (one for each position in the vector).
For a stroke point at Depth 2, your full input tensor row would look something like this:
[Δx, Δy, s(t), pen-state, 0, 1, 0, 0]
## Why this helps your LSTM

* Clear separation: The network can easily learn completely different rules for Depth 1 versus Depth 2.
* Better gradients: Neural networks generally find it much easier to learn from distinct binary switches (0 or 1) than trying to decipher the subtle difference between 0.4 and 0.6 on a single continuous scale.

Would you like to see a quick Python example of how to automatically convert your Kanji depths into these one-hot rows, or should we look at how this changes the total size of your input tensor?

[1] [https://www.digitalocean.com](https://www.digitalocean.com/community/tutorials/understanding-one-hot-encoding-in-machine-learning)
[2] [https://www.analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2023/12/how-to-do-one-hot-encoding/)
[3] [https://apxml.com](https://apxml.com/courses/intro-feature-engineering/chapter-3-encoding-categorical-features/one-hot-encoding)
[4] [https://medium.com](https://medium.com/@asmmorshedulhoque/a-beginners-guide-to-one-hot-encoding-in-nlp-c2c5f5cf57db)
[5] [https://builtin.com](https://builtin.com/articles/one-hot-encoding)
[6] [https://medium.com](https://medium.com/@asmmorshedulhoque/a-beginners-guide-to-one-hot-encoding-in-nlp-c2c5f5cf57db)
[7] [https://letsdatascience.com](https://letsdatascience.com/blog/categorical-encoding-a-practical-guide-to-one-hot-label-and-target-methods)
[8] [https://medium.com](https://medium.com/@abhaysingh71711/advanced-feature-engineering-part-2-feature-encoding-858dd4b8061a)
[9] [https://www.kaggle.com](https://www.kaggle.com/questions-and-answers/396146)
[10] [https://medium.com](https://medium.com/@asmmorshedulhoque/a-beginners-guide-to-one-hot-encoding-in-nlp-c2c5f5cf57db)
[11] [https://www.giskard.ai](https://www.giskard.ai/glossary/one-hot-encoding)
[12] [https://campus.datacamp.com](https://campus.datacamp.com/courses/introduction-to-deep-learning-with-pytorch/neural-network-architecture-and-hyperparameters-2?ex=8)
[13] [https://soulpageit.com](https://soulpageit.com/ai-glossary/one-hot-encoding-explained/)
[14] [https://towardsdatascience.com](https://towardsdatascience.com/inside-gpt-i-1e8840ca8093/)
[15] [https://medium.com](https://medium.com/@oyebamijimicheal10/a-comprehensive-comparison-between-one-hot-and-ordinal-encoding-6f899c4f08b3)
[16] [https://medium.com](https://medium.com/@amit25173/applying-one-hot-encoding-in-pandas-5a639cb3bc69)
[17] [https://medium.com](https://medium.com/ai-enthusiast/categorical-encoding-techniques-from-one-hot-to-target-encoding-c7e134eda7f2)
[18] [https://medium.com](https://medium.com/@debisreer/encoding-in-machine-learning-8e7b11857daa)

> Good explanation. Have one bit for every category and flip the one to 1 for actual category. See bit vector as, erm, a vector of 0 and 1. Still two strokes on the same level can belong to different components.

Exactly. You nailed the concept: a bit vector where only the active category is flipped to 1. [1]
You also spotted the exact flaw with raw depth: two strokes on the same depth level can belong to entirely different components (e.g., the left radical vs. the right radical, both at depth 2). To the model, they look identical structural-wise.
To fix this and uniquely identify distinct components, you can use Component ID Grouping combined with an Embedding Layer.
Here is how you solve it.
## Step 1: Assign a Unique ID to Each Component
Instead of tracking just depth, your XML parser tracks unique path groups per Kanji. Every time the SVG opens a new <g> (group) tag, it gets a unique ID number for that specific Kanji layout.
Look at a Kanji split into a Left and Right side, where both sides break down further:

* Root (Whole Kanji): ID 0
* Left Component: ID 1
   * Left-Top Part: ID 2
      * Left-Bottom Part: ID 3
   * Right Component: ID 4
   * Right-Top Part: ID 5

Now, strokes on the same depth level get completely different IDs. The left-top stroke gets ID 2, and the right-top stroke gets ID 5.
## Step 2: Don't One-Hot Encode IDs (Use Embeddings)
If Kanji A has 6 components and Kanji B has 12 components, you cannot easily one-hot encode them because the number of columns would keep changing.
Instead, you use an Embedding Layer (a standard tool in PyTorch or [TensorFlow](https://www.tensorflow.org/)) inside your model: [2, 3, 4]

   1. You pass the raw integer Component ID (e.g., 2 or 5) as a single extra column in your tensor.
   2. The network passes this integer into an Embedding Layer (think of it as a trainable lookup table).
   3. The layer automatically converts that single integer into a small, fixed-size vector of floats (e.g., a size of 4 or 8 numbers) that the model trains itself. [5, 6, 7, 8]

Integer ID (e.g., 2)  -->  [Embedding Layer]  -->  [0.12, -0.45, 0.89, -0.01]

## Why this is the ultimate fix
The model learns to cluster these vectors. It will naturally learn that IDs 2 and 3 are "close" because they both belong to the left side of the Kanji, while ID 5 is far away on the right side.
This gives your LSTM both horizontal context (which specific piece am I in?) and vertical context (how deep is it?) simultaneously.
How does your current parser extract data from the KanjiVG SVG files? I can show you how to inject these Group IDs into your preprocessing pipeline if you like.

[1] [https://people.engr.tamu.edu](https://people.engr.tamu.edu/choe/choe/mirror/kanerva.NIPS98-kanerva.pdf)
[2] [https://medium.com](https://medium.com/@smrati.katiyar/explaining-embedding-layer-in-pytorch-1f22b88c1a69)
[3] [https://colab.sandbox.google.com](https://colab.sandbox.google.com/github/securetorobert/docs/blob/master/site/en/tutorials/keras/intro_word_embeddings.ipynb)
[4] [https://pub.aimind.so](https://pub.aimind.so/mastering-the-embedding-layer-unveiling-the-art-and-science-of-text-representation-in-neural-d7fc52a94c46)
[5] [https://www.reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/18g6aze/mixtral8x7binstructv01_seems_good/)
[6] [https://news.ycombinator.com](https://news.ycombinator.com/item?id=36248633)
[7] [https://colab.sandbox.google.com](https://colab.sandbox.google.com/github/securetorobert/docs/blob/master/site/en/tutorials/keras/intro_word_embeddings.ipynb)
[8] [https://liu-nlp.ai](https://liu-nlp.ai/dl4nlp/units/NLP-2026-14.pdf)

> Ok. I take the general idea of structural encoding has potentially its merits, but I won't tackle it know. I let it simmer in the back of my head. Thanks for the assist.
