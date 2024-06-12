# Geometry Sharpness-Aware Minimization

In Sharpness-Aware Minimization, they compute gradients at two different positions within one update step, denoted as $\nabla L(w_t)$ and $\nabla L(w_t + \rho \frac{ \nabla L(w_t) }{ || \nabla L(w_t) || })$. This allows us to explore the geometry of the loss landscape through these gradients.

To validate the geometry of loss landscape, we use the ratio of the perturbed gradient to the current gradient :

$$\nabla L(w_t + \rho \frac{ \nabla L(w_t) }{ || \nabla L(w_t) || }) / \nabla L(w_t)
$$. 

The figure below illustrates the effectiveness of this metric in depicting the local convex minima.

![](checkpoint.png)

In the figure, `checkpoint1` and `checkpoint2` are notable because their perturbed gradients share the same sign as the current gradient. Experiments show that during training, over $60\%$ of parameters belong to `checkpoint1`. Conversely, `checkpoint2` starts at around $40\%$ in the early stage and decreases to $20\%$ in the later stage.

![](checkpoint2.png)
In contrast, checkpoint3 and checkpoint4 depict scenarios where the signs of the perturbed and current gradients differ. During training, the number of parameters associated with checkpoint3 and checkpoint4 is initially very low, at around $3\%$, but increases to $10\%$ and $6\%$ respectively in the later stages.

## Observations
These results indicate that the loss landscape geometry is predominantly of the `checkpoint1` type. There is a transition from `checkpoint2` to `checkpoint3` and `checkpoint4` as training progresses.

These observations suggest that the loss landscape in the late stages of training comprises multiple minima regions, as evidenced by the increase in `checkpoint3` and `checkpoint4`. Surprisingly, the number of `checkpoint1` parameters does not decrease during training, implying that a large number of parameters remains on the ridge of the landscape

## Further Experiments
A curious question arises: which checkpoint among the four contributes the most to finding the flat minima in SAM? This warrants further investigation to enhance our understanding and effectiveness of the SAM algorithm.

Experiment 1: We have ran the experiments to check that hypothesis and the results showed that if we only maintain the magnitude of all of the parameters belong to `checkpoint1` and replace others with magnitude of SGD. In this case, the finding flat minima ability of SAM still be remained while repeating this experiment with other `checkpoint` end up with the sharper minima.

Conclusion, the realistic ability of SAM is effective learning rate, but not lie on direction modification.

**Research Question**: The ratio of perturbed gradient and current gradient belongs to `checkpoint1` maintaion through how many steps:

Experiment 2: How many number of parameter belongs to `checkpoint1` in step 172 still belongs to `checkpoint1` in later steps. The results showed that at step 172, there are $6*10^6$ parameters and the overlap of these parameters with parameter of later step always maintain around $4*10^6$ even at the last step.

That is a very surprising observation that always exists $4*10^6 \sim 40\%$ parameters need the higher learning rate than the initial one. 

Experiment 3: Would these $4*10^6$ parameters be the same in many steps? Answer: No. The results showed that only after 5 steps the overlap parameters between these step reduce to $10^5$ and diminish soon.

**Research Question**: Think about effective learning rate. Why large batch training has lower generalization, what if the reason is about low learning rate?