# Archived due to dropping the course

# DL-PCImageGen

By Wayne Zeng and Diego Arapovic (diego-arapovic <diego-arapovic@users.noreply.github.com>)

This repository is based on the works of @tommorse, specifically https://github.com/ModelDBRepository/218084, Whittington and Bogasz.

We aim to build a basic PC Network in order to then attempt the challenges proposed in "A Predictive-Coding Network That Is Both Discriminative and Generative" (https://pubmed.ncbi.nlm.nih.gov/32795234/).

## Proposal Feedback

The proposal is very clear, and the idea is valid. The literature review is complete but too brief. It would have been better to give a more detailed description of the cited papers, especially the main reference [2]. It follows that both the motivation and the method could have been elaborated further. One small note on the idea: the results in [2] are far from satisfying, so there’s definitely room for improvement even on MNIST. Instead of blindly following the recipe from [2], you might want to experiment (first on MNIST) with new solutions. Good luck!

## Main Issues to resolve

- Structure: The code is still written "in a matlab style". Should be importable as a pytorch nn module and just plug in and go -> to watch pytorch tutorials until this is worked out. 

## Notes from ModelDBRepo that still need to be implemented

example_code.m generates data for an XOR gate. Then trains a predictive coding network, as well as the equivalent MLP on the data.

DONE - f.m - calculates the an activation function. - This is already pre-implemented in PyTorch

DONE - f_b.m - calculates the an activation function as well as its derivitaive. - Ditto

TODO - w_init.m - initialises a set of random weights, for a given network structure - Don't really understand what they're doing in this file. Claim weights are random but they're clearly not.

(The following codes only accept one data point at a time)

DONE - test - makes a prediction for an ann/pc network + outputs rmse

DONE - rms_error - calculated rmse

DONE - learn_ann - performs back-propagation - from what I understand, this is just standard BP. 

DONE, to implement in pytorch - learn_pc - performs the learning for a predictive coding network - to research how to make own NN architecture in pytorch, maybe pytorch lightning

DONE, to implement in pytorch - infer_pc - performs the inference stage - tied in with learn_pc as the more important code

TODO - example_code to redo again.
