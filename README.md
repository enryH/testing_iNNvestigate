# Test iNNvestigate Analyzer regarding DTD assumptions

## References:

On DTD:
> Montavon, G., Lapuschkin, S., Binder, A., Samek, W., Müller, K.R.: Explaining nonlinear classification decisions with deep Taylor decomposition. Pattern Recogn. 65, 211–222 (2017)

On Implementatin of LRP methods:
> Montavon G., Binder A., Lapuschkin S., Samek W., Müller KR. (2019) Layer-Wise Relevance Propagation: An Overview. In: Samek W., Montavon G., Vedaldi A., Hansen L., Muller KR. (eds) Explainable AI: Interpreting, Explaining and Visualizing Deep Learning. Lecture Notes in Computer Science, vol 11700. Springer, Cham

## Check conservation of decomposition 

- Test image is a 9 

### Having no bias constraint 
- some relevances is "leaking" as described by [Sebastian Lapuschkin](https://github.com/albermax/innvestigate/issues/91#issuecomment-414376522). 
- [Source](testing_dtd_wo_bias_constraint.py)

```python
# Create analyzer
analyzer = innvestigate.create_analyzer("deep_taylor", model_wo_sm)

# Applying the analyzer
analysis = analyzer.analyze(image)

# Check Conservation
scores = model_wo_sm.predict(image)
print("Maximum Score: {:.3f} with label {}".format(scores.max(), scores.argmax()))
print("sum of relevances assigned to inputs: {:.3f}".format(analysis.sum()))
try:
    assert abs(scores.max() - analysis.sum()) < 0.001
except AssertionError:
    print("not equal...")
# Biases are included and conversation property in DTD framework fails
```
```
Maximum Score: 11.711 with label 9
sum of relevances assigned to inputs: 11.561
not equal...
```

```python
# LRP-Alpha_1-Beta_0 without biases is z+ rule in DTD paper
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPAlpha1Beta0IgnoreBias
analyzer = LRPAlpha1Beta0IgnoreBias(model_wo_sm)

# Applying the analyzer
analysis = analyzer.analyze(image)

# Check Conservation
scores = model_wo_sm.predict(image)
print("Maximum Score: {:.3f} with label {}".format(scores.max(), scores.argmax()))
print("sum of relevances assigned to inputs: {:3f}".format(analysis.sum()))
assert abs(scores.max() - analysis.sum()) < 0.001
```

```
Maximum Score: 11.711 with label 9
sum of relevances assigned to inputs: 11.711
```

### Constrainig the bias in ReLUs to be negative
- relevance over inputs is larger as output score using default setup
- [Source](testing_dtd_w_bias_constraint.py)
```python
# Create analyzer
analyzer = innvestigate.create_analyzer("deep_taylor", model_wo_sm)

# Applying the analyzer
analysis = analyzer.analyze(image)

# Check Conservation
scores = model_wo_sm.predict(image)
print("Maximum Score: {:.3f} with label {}".format(scores.max(), scores.argmax()))
print("sum of relevances assigned to inputs: {:.3f}".format(analysis.sum()))
try:
    assert abs(scores.max() - analysis.sum()) < 0.001
except AssertionError:
    print("not equal...")
# Biases are included and conversation property in DTD framework fails
```
```
Maximum Score: 12.835 with label 9
sum of relevances assigned to inputs: 13.338
not equal...
```
```python
# LRP-Alpha_1-Beta_0 without biases is z+ rule in DTD paper
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPAlpha1Beta0IgnoreBias
analyzer = LRPAlpha1Beta0IgnoreBias(model_wo_sm)

# Applying the analyzer
analysis = analyzer.analyze(image)

# Check Conservation
scores = model_wo_sm.predict(image)
print("Maximum Score: {:.3f} with label {}".format(scores.max(), scores.argmax()))
print("sum of relevances assigned to inputs: {:3f}".format(analysis.sum()))
assert abs(scores.max() - analysis.sum()) < 0.001
```

```
Maximum Score: 12.835 with label 9
sum of relevances assigned to inputs: 12.835
```

