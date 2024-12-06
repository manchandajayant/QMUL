# 1.

## 1.1 Optimal Model

![alt text](<Screenshot 2024-12-05 at 4.41.35 PM.png>)

| \*  | Layer 1 (Neurons, Activation, Dropout) | Layer 2 (Neurons, Activation, Dropout) | Accuracy (%) |
| --- | -------------------------------------- | -------------------------------------- | ------------ |
| \*  | 64, ReLU, 0.4                          | 128, Tanh, 0.5                         | 49.30        |

---

## 1.2 Elements of the Algorithm

### State Encoding

Each genome corresponds to a network architecture defined as a list of layers. Each layer is a blueprint, specifying the number of neurons, the activation function, and the dropout rate. With this structure, modification to the genome happens through crossover and mutation, leading to modifications to the network's architecture with every iteration to improve predictions. In my blueprint, I use 2 layers, between 64 and 512 neurons, dropout rates between 0.2 and 0.5, and only ReLU and Tanh as activation functions. By running tests, I found these to be good for accuracy and convergence.

### Selection

During the selection process, the code first sorts the genomes by their fitness scores and assigns each one a rank. Then, it calculates each genome’s selection probability based on that rank. Finally, it uses these probabilities to randomly pick a set number of parents, ensuring that higher-ranked genomes have a better chance of being chosen. This approach was chosen so that higher-performing genomes have a better chance of contributing their traits to the next generation.

### Crossover

The crossover process performs a two-point crossover when the parent genomes have more than two layers. It selects two positions within the parent's genome, takes the initial segment from the first parent up to the first cutoff point, then includes the segment from the second parent until the second cutoff point, and adds the remaining layers from the first parent.

If the parent genome has two layers or fewer, it uses a single-point crossover, where it chooses one cutoff point and combines the former portion from the first parent and the latter portion from the second parent.

I wrote this to test multiple layer combinations, but since my optimal layer architecture only has 2 layers, it performs single-point crossover for my optimal case. I left the code for two-point crossover for testing with a higher number of layers.

### Mutation

During the mutation process, every genome layer is iterated over, and every parameter is updated individually based on the mutation rate. The neuron layer has an additional check with a higher threshold to ensure neuron count mutations happen more frequently. I found this to have more effect on variations than changing the activation or dropout rate.

---

## 1.3 The Number of Reproductions

| Run | Layer 1 (Neurons, Activation, Dropout) | Layer 2 (Neurons, Activation, Dropout) | Accuracy (%) |
| --- | -------------------------------------- | -------------------------------------- | ------------ |
| 1   | 64, ReLU, 0.3237                       | 128, Tanh, 0.4956                      | 45.68        |
| 2   | 128, ReLU, 0.1126                      | 512, Tanh, 0.3284                      | 41.23        |
| 3   | 128, Tanh, 0.4868                      | 256, ReLU, 0.2256                      | 43.45        |
| 4   | 256, ReLU, 0.2434                      | 128, ReLU, 0.4100                      | 47.35        |
| 5   | 64, Tanh, 0.4164                       | 512, ReLU, 0.4315                      | 47.08        |
| 6   | 64, ReLU, 0.5                          | 128, ReLU, 0.5                         | 45.68        |
| 7   | 512, ReLU, 0.2960                      | 64, Tanh, 0.2840                       | 44.29        |
| 8   | 512, ReLU, 0.3                         | 128, ReLU, 0.3                         | 46.52        |
| 9   | 64, ReLU, 0.3                          | 128, ReLU, 0.2                         | 52.92        |
| 10  | 256, ReLU, 0.4071                      | 512, ReLU, 0.5566                      | 49.86        |

The algorithm occasionally discovers better-performing architectures, but this might also be due to lucky weight initialization in those runs. The accuracy ranges from 41.23% to 52.92%, with the algorithm coming close to the optimal architecture in run 10, achieving 49.86%. Models using ReLU activation consistently perform well, appearing in 8 out of 10 runs. Moderate dropout values (0.2–0.5) balance regularization and performance. Simpler architectures in Layer 1, as seen in runs 1, 5, and 6, perform consistently near the optimum. Higher neuron counts with lower dropout rates tend to perform less well. Random initializations may contribute to occasional improvements, and fine-tuning the mutation rate could further improve performance.

From the above, a combination of fewer neurons in the first layer, ReLU as an activation function, and a slightly higher dropout rate could be considered optimal. With a higher number of neurons, a higher dropout rate seems to work well.

---

## 1.4 Hyper-parameters

_(No additional details provided in the original text)_

For this task, I concentrated on 2 main adjustments :

-   Number of Layers
-   Selection and Mutation Approaches

I started with a five-layer network and later reduced it to two layers. For selection, I switched from random choice to a rank-based probability method, ensuring fitter genomes had a higher chance of being chosen. For mutation, instead of uniformly mutating all parameters, I assigned separate probabilities to each parameter, leading to more guided variability.

Overall, the enhanced configuration not only achieved higher peak performances (e.g., 52.92% vs. 44.01% in the old configuration) but also demonstrated more consistently improved results across multiple runs. The average of the enhanced metrics was notably higher than that of the old configuration, indicating that the refined selection and mutation strategies, coupled with a simpler network, gave more effective solutions.

I ran for both the model 10 times, with the same population size, generations size, learning rate and EPOCHS

![alt text](hyper.png)

---

# 2.

## 2.1

Given the problem is a multi-class classification problem, the most appropriate loss function to use is the Cross Entropy function, as it directly measures the difference between the predicted values and true labels from the dataset.

![alt text](<WhatsApp Image 2024-12-06 at 12.55.30 AM.jpeg>)

### Interpretation of the Mathematical Properties

-   **Cross-entropy** measures how closely the model’s predicted probabilities match the true one-hot distribution. Minimizing it encourages the model to increase the probability of the correct class.
-   Taking the negative logarithm of the predicted probability for the correct class heavily penalizes confident yet incorrect predictions, driving the model to correct them.
-   The gradient of cross-entropy loss with respect to the logits is the predicted probability minus the true label. This provides a clear, proportional correction to the model’s predictions.
-   Implementing cross-entropy jointly with softmax ensures numerical stability and avoids issues like vanishing or exploding gradients, facilitating efficient training.

## 2.3

![alt text](<Screenshot 2024-12-05 at 10.31.46 PM.png>)

![alt text](cnn-loss-accuracy.png)

The model converges well on the training data but shows signs of overfitting during later epochs, as evidenced by the fluctuations and lack of consistent improvement in validation performance.

### Accuracy Trends

-   The training accuracy improves steadily and reaches a very high value by the end of training. The growth slows down significantly after around epoch 20, suggesting the model is nearing its capacity to fit the training data.
-   The validation accuracy increases quickly in the early epochs and stabilizes around 91%-92%, showing minor fluctuations near the end of training. This fluctuation may indicate slight overfitting.

### Loss Trends

-   The training loss decreases steadily, with a slower rate of improvement after epoch 20, moving towards convergence with some fluctuations at the end.
-   After around epoch 20, the validation loss starts to increase despite the training loss continuing to decrease. This is a sign of overfitting, indicating the model has learned patterns from the training set and is losing its ability to generalize.

---

# 3.

## 3.1

### Take Whole Apple

**Pre-conditions:**

-   Apple is available

**Effects:**

-   Apple is in the robot's hand
-   Apple is not available anymore

### Peel Apple

**Pre-conditions:**

-   Apple is clean
-   Apple is not peeled
-   Apple is in the robot's hand
-   Knife is in the robot's hand

**Effects:**

-   Apple is peeled

### Cut Apple

**Pre-conditions:**

-   Apple is peeled
-   Apple is not cut
-   Apple is in the robot's hand
-   Knife is in the robot's hand

**Effects:**

-   Apple is cut

### Put Down Knife

**Pre-conditions:**

-   Knife is in the robot's hand

**Effects:**

-   Knife is available
-   Knife is not in the robot's hand

### Add Chopped Apple to the Container (to serve guests)

**Pre-conditions:**

-   Apple is in the robot's hand
-   Container is empty

**Effects:**

-   Container contains the apple
-   Container is not empty
-   Apple is not in the robot's hand

---

# 3.2

**Take whole Apple (i.e. before it is peeled or chopped) :**

    (available apple)
    (is_clean apple)
    (not (in_hand apple))
    (not (is_peeled apple))
    (not (is_cut apple))
    (available knife)
    (not (in_hand knife))
    (empty container)

**Peel Apple :**

    (not (is_peeled apple))
    (in_hand apple)
    (in_hand knife)
    (not (is_cut apple))

**Cut Apple :**

    (is_peeled apple)
    (not (is_cut apple))
    (in_hand apple)
    (in_hand knife)

**Put down Knife :**

    (in_hand knife)
    (is_peeled apple)
    (is_cut apple)

**Add chopped apple to the container (in order to serve to your guests) :**

    (in_hand apple)
    (empty container)

---

# 3.3

**The action of peeling an apple**

<div style="background-color: #f3d3d3; padding: 10px; border-radius: 5px;">

    For all, if an apple is clean, not peeled and both the apple and the knife are in hand,
    then the action of peeling makes the apple peeled.

    In First Order Logic form :
    	- Taking Apple as x and knife as y
    		Definition :
    			is_clean(x): x is clean
    			is_peeled(x): x is peeled
    			in_hand(x): x is in hand
    			in_hand(y): y is in hand

    FOL statement :  ∀x∀y[(is_clean(x) ∧ ¬is_peeled(x) ∧ in_hand(x) ∧ in_hand(y)) => is_peeled(x)]
    	A ⟹ B
    	where
    	A := is_clean(x) ∧ ¬is_peeled(x) ∧ in_hand(x) ∧ in_hand(y)
    	And
    	B := is_peeled(x)
    	A => B is equivalent to ¬A V B
    	Negation : ∀x∀y[ ¬(is_clean(x) ∧ ¬is_peeled(x) ∧ in_hand(x) ∧ in_hand(y)) ∨ is_peeled(x)]

    Applying De Morgan’s Laws : ¬(A ∧ B ∧ C ∧ D) ≡ (¬A ∨ ¬B ∨ ¬C ∨ ¬D)
    	A = is_clean(x)
    	B= ¬is_peeled(x)
    	C = in_hand(x)
    	D = in_hand(y)

    Substituting :
    	= ∀x∀y[( ¬is_clean(x) ∨ is_peeled(x) ∨ ¬in_hand(x) ∨ ¬in_hand(y)) ∨ is_peeled(x)]

    	= ∀x∀y [ ¬is_clean(x) ∨ is_peeled(x) ∨ ¬in_hand(x) ∨ ¬in_hand(y)]

    CNF = ∀x∀y [ ¬is_clean(x) ∨ is_peeled(x) ∨ ¬in_hand(x) ∨ ¬in_hand(y)],
    where this statement is a disjunction of literals since every  sub statement is joined by an V (OR)

</div>

**The action of cutting an apple**

<div style="background-color: #f3d3d3; padding: 10px; border-radius: 5px;">

    For all, if an apple is clean, peeled, not peeled and both the apple and the knife are in hand,
    then the action of cutting the apple makes the apple cut
    In First Order Logic form :
    	- Taking the Apple as x and knife as y
    		Definition :
    			is_clean(x) : x is clean
    			is_peeled(x) : x is peeled
    			is_cut(x) : x is cut
    			in_hand(x): x is in hand
    			in_hand(y): y is in hand

    FOL statement :  ∀x∀y[(is_clean(x) ∧ is_peeled(x) ∧ ¬is_cut(x) ∧ in_hand(x) ∧ in_hand(y)) => is_cut(x)]
    	A ⟹ B
    	where
    	A := is_clean(x) ∧ is_peeled ∧ ¬is_cut(x) ∧ in_hand(x) ∧ in_hand(y)
    	And
    	B := is_cut(x)
    	A => B is equivalent to ¬A V B
    	Negation : ∀x∀y[ ¬(is_clean(x) ∧ is_peeled(x) ∧ ¬is_cut(x) ∧ in_hand(x) ∧ in_hand(y)) ∨ is_cut(x)]

    Applying De Morgan’s Laws : ¬(A ∧ B ∧ C ∧ D ∧ E) ≡ (¬A ∨ ¬B ∨ ¬C ∨ ¬D ∨ ¬E)
    	A = is_clean(x)
    	B = is_peeled(x)
    	C = ¬is_cut(x)
    	D = in_hand(x)
    	E = in_hand(y)

    	= ∀x∀y[( ¬is_clean(x) ∨ ¬is_peeled(x) ∨ is_cut(x) v ¬in_hand(x) ∨ ¬in_hand(y)) ∨ is_cut(x)]

    	= ∀x∀y [ ¬is_clean(x) ∨ ¬is_peeled(x) ∨ is_cut(x) ∨ ¬in_hand(x) ∨ ¬in_hand(y)]

    	CNF = ∀x∀y [ ¬is_clean(x) ∨ ¬is_peeled(x) ∨ is_cut(x) ∨ ¬in_hand(x) ∨ ¬in_hand(y)],
    	where this statement is a disjunction of literals since every sub statement is joined by an V (OR)

</div>

---

# 3.4

<div style="background-color: #f3d3d3; padding: 10px; border-radius: 5px;">

**Step 1: Detailed Sequence of Actions**

    - Initial conditions :
    	- Apple a is clean: is_clean(a)
    	- Apple a is not peeled: ¬is_peeled(a)
    	- Apple a is not cut: ¬is_cut(a)
    	- Container c is empty: empty(c)
    	- Apple a is available: available(a)
    	- Knife k is available: available(k)
    	- Container c is available: available(c)

    - Actions :
    	Take the Apple:
    		Precondition: available(a)
    		Effect: in_hand(a) and ¬available(a)

    	Take the Knife:
    		Precondition: available(k)
    		Effect: in_hand(k) and ¬available(k)

    	Peel the Apple:
    		Precondition: is_clean(a) ∧ ¬is_peeled(a) ∧ in_hand(a) ∧ in_hand(k)
    		Effect: is_peeled(a)

    	Cut the Apple:
    		Precondition: is_peeled(a) ∧ ¬is_cut(a) ∧ in_hand(a) ∧ in_hand(k)
    		Effect: is_cut(a)

    	Add the Apple to the Container: Precondition: in_hand(a) ∧ empty(c)
    		Effect: contains(c, a) ∧ ¬empty(c) ∧ ¬in_hand(a)

**Step 2: First order Logic**

    	Initial condition -
    		is_clean(a) ∧ ¬is_peeled(a) ∧ ¬is_cut(a) ∧ empty(c) ∧ available(a) ∧ available(k)

    	Take the apple from initial condition -
    		(available(a)) -> (in_hand(a) ∧ ¬available(a))

    	From having the apple in hand to taking the knife -
    		(in_hand(a) ∧ available(k)) -> (in_hand(a) ∧ in_hand(k) ∧ ¬available(k))

    	From having apple and knife in hand to peeling the apple -
    		(is_clean(a) ∧ ¬is_peeled(a) ∧ in_hand(a) ∧ in_hand(k)) -> is_peeled(a)

    	From a peeled but not cut apple, to cutting the apple -
    		(is_peeled(a) ∧ ¬is_cut(a) ∧ in_hand(a) ∧ in_hand(k)) -> is_cut(a)

    	From a cut apple in hand and empty container to adding the apple to the container -
    		(in_hand(a) ∧ empty(c)) -> (contains(c,a) ∧ ¬empty(c) ∧ ¬in_hand(a))

**Step 3: Using LTL**

    LTL Formula :
    	Implementing the formula with actions in sequence of order -
    	G( Initial -> X( TookApple -> X( TookKnife -> X( Peeled -> X( Cut -> X( Added ))))))

    	Substituting values :
    	G((is_clean(a) ∧ ¬is_peeled(a) ∧
    	¬is_cut(a) ∧ empty(c) ∧ available(a) ∧ available(k)) ->
    	X((in_hand(a) ∧ ¬available(a)) ->
    	X((in_hand(a) ∧ in_hand(k) ∧ ¬available(k)) ->
    	X(is_peeled(a) -> X(is_cut(a) -> X(contains(c,a) ∧
    	¬empty(c) ∧ ¬in_hand(a) ))))))

</div>
