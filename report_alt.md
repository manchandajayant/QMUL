# 1.

## 1.1 Optimal Model

![alt text](/Users/jayantmanchanda/Desktop/QMUL_STUDY_MATERIAL/AI/Assigment/ECS759P_CW2/Screenshot 2024-12-05 at 4.41.35 PM.png)

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


## 2.2

![alt text](<Screenshot 2024-12-05 at 10.31.46 PM.png>)

![alt text](cnn-loss-accuracy.png)

The model converges well on the training data but shows signs of overfitting during later epochs, as evidenced by the fluctuations and lack of consistent improvement in validation performance.

### Accuracy Trends

-   The training accuracy improves steadily and reaches a very high value by the end of training. The growth slows down significantly after around epoch 20, suggesting the model is nearing its capacity to fit the training data.
-   The validation accuracy increases quickly in the early epochs and stabilizes around 91%-92%, showing minor fluctuations near the end of training. This fluctuation may indicate slight overfitting.

### Loss Trends

-   The training loss decreases steadily, with a slower rate of improvement after epoch 20, moving towards convergence with some fluctuations at the end.
-   After around epoch 20, the validation loss starts to increase despite the training loss continuing to decrease. This is a sign of overfitting, indicating the model has learned patterns from the training set and is losing its ability to generalize.
